

#include "config.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "UCTNode.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GTP.h"
#include "GameState.h"
#include "Network.h"
#include "distance.h"
#include "Utils.h"
#include <iostream>

#define forEach4Nbr(v_origin,v_nbr,block)		\
	int v_nbr;									\
	v_nbr = v_origin + 1;			block;		\
	v_nbr = v_origin - 1;			block;		\
	v_nbr = v_origin + EBSIZE;		block;		\
	v_nbr = v_origin - EBSIZE;		block;

bool japanese_rule = false;

using namespace Utils;

UCTNode::UCTNode(int vertex, float score) : m_move(vertex), m_score(score) {
}

bool UCTNode::first_visit() const {
    return m_visits == 0;
}

SMP::Mutex& UCTNode::get_mutex() {
    return m_nodemutex;
}

bool UCTNode::create_children(std::atomic<int>& nodecount,
                              GameState& state,
                              float& eval,
                              float min_psa_ratio) {
    // check whether somebody beat us to it (atomic)
    if (!expandable(min_psa_ratio)) {
        return false;
    }
    // acquire the lock
    LOCK(get_mutex(), lock);
    // no successors in final state
    if (state.get_passes() >= 2) {
        return false;
    }
    // check whether somebody beat us to it (after taking the lock)
    if (!expandable(min_psa_ratio)) {
        return false;
    }
    // Someone else is running the expansion
    if (m_is_expanding) {
        return false;
    }
    // We'll be the one queueing this node for expansion, stop others
    m_is_expanding = true;
    lock.unlock();

    const auto raw_netlist = Network::get_scored_moves(
        &state, Network::Ensemble::RANDOM_SYMMETRY);

    // DCNN returns winrate as side to move
    m_net_eval = raw_netlist.winrate;
    const auto to_move = state.board.get_to_move();
    // our search functions evaluate from black's point of view
    if (state.board.white_to_move()) {
        m_net_eval = 1.0f - m_net_eval;
    }
    eval = m_net_eval;

    std::vector<Network::ScoreVertexPair> nodelist;

    auto legal_sum = 0.0f;
    for (auto i = 0; i < BOARD_SQUARES; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state.board.get_vertex(x, y);
        if (state.is_move_legal(to_move, vertex)) {
            nodelist.emplace_back(raw_netlist.policy[i], vertex);
            legal_sum += raw_netlist.policy[i];
        }
    }
    nodelist.emplace_back(raw_netlist.policy_pass, FastBoard::PASS);
    legal_sum += raw_netlist.policy_pass;

    if (legal_sum > std::numeric_limits<float>::min()) {
        // re-normalize after removing illegal moves.
        for (auto& node : nodelist) {
            node.first /= legal_sum;
        }
    } else {
        // This can happen with new randomized nets.
        auto uniform_prob = 1.0f / nodelist.size();
        for (auto& node : nodelist) {
            node.first = uniform_prob;
        }
    }

    link_nodelist(nodecount, nodelist, min_psa_ratio);
    return true;
}

void UCTNode::link_nodelist(std::atomic<int>& nodecount,
                            std::vector<Network::ScoreVertexPair>& nodelist,
                            float min_psa_ratio) {
    assert(min_psa_ratio < m_min_psa_ratio_children);

    if (nodelist.empty()) {
        return;
    }

    // Use best to worst order, so highest go first
    std::stable_sort(rbegin(nodelist), rend(nodelist));

    LOCK(get_mutex(), lock);

    const auto max_psa = nodelist[0].first;
    const auto old_min_psa = max_psa * m_min_psa_ratio_children;
    const auto new_min_psa = max_psa * min_psa_ratio;
    if (new_min_psa > 0.0f) {
        m_children.reserve(
            std::count_if(cbegin(nodelist), cend(nodelist),
                [=](const auto& node) { return node.first >= new_min_psa; }
            )
        );
    } else {
        m_children.reserve(nodelist.size());
    }

    auto skipped_children = false;
    for (const auto& node : nodelist) {
        if (node.first < new_min_psa) {
            skipped_children = true;
        } else if (node.first < old_min_psa) {
            m_children.emplace_back(node.second, node.first);
            ++nodecount;
        }
    }

    m_min_psa_ratio_children = skipped_children ? min_psa_ratio : 0.0f;
    m_is_expanding = false;
}

const std::vector<UCTNodePointer>& UCTNode::get_children() const {
    return m_children;
}


int UCTNode::get_move() const {
    return m_move;
}

void UCTNode::virtual_loss() {
    m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::virtual_loss_undo() {
    m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}

void UCTNode::update(float eval,float rollout) {
    m_visits++;
    accumulate_eval(eval);
    accumulate_rollouts(rollout);
    if(rollout==1)
        accumulate_rolloutwin();
}

bool UCTNode::has_children() const {
    return m_min_psa_ratio_children <= 1.0f;
}

bool UCTNode::expandable(const float min_psa_ratio) const {
    return min_psa_ratio < m_min_psa_ratio_children;
}

float UCTNode::get_score() const {
    return m_score;
}

void UCTNode::set_score(float score) {
    m_score = score;
}

int UCTNode::get_visits() const {
    return m_visits;
}

float UCTNode::get_nn_eval(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    auto virtual_loss = int{m_virtual_loss};
    auto visits = get_visits() + virtual_loss;
    
    assert(visits > 0);
    auto blackeval = get_blackevals();
    if (tomove == FastBoard::WHITE) {
        blackeval += static_cast<double>(virtual_loss);
    }
    auto score = static_cast<float>(blackeval / double(visits));
    if (tomove == FastBoard::WHITE) {
        score = 1.0f - score;
    }
    //myprintf(" v= %d ",get_visits());
    return score;
}


float UCTNode::get_rollouts(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    auto virtual_loss = int{m_virtual_loss};
    auto visits = get_visits() + virtual_loss;
    
    assert(visits > 0);
    auto blackrollouts = get_blackrollouts();
    if (tomove == FastBoard::WHITE) {
        blackrollouts += static_cast<double>(virtual_loss);
    }
    auto score = static_cast<float>(blackrollouts / double(visits));
    if (tomove == FastBoard::WHITE) {
        if(cfg_rollout_mode)
            score =  -score;
        else
            score = 1.0f - score;
    }
    //myprintf(" r= %d \n",get_visits());
    return score;
}

float UCTNode::get_rollout_winrate(int tomove) const {
    // Due to the use of atomic updates and virtual losses, it is
    // possible for the visit count to change underneath us. Make sure
    // to return a consistent result to the caller by caching the values.
    
    auto visits = get_visits();
    assert(visits > 0);
    auto blackrollouts = get_rolloutwin();
    auto score = static_cast<float>(blackrollouts / double(visits));
    if (tomove == FastBoard::WHITE) {
        score = 1.0 - score;
    }
    return score;
}

float UCTNode::get_net_eval(int tomove) const {
    if (tomove == FastBoard::WHITE) {
        return 1.0f - m_net_eval;
    }
    return m_net_eval;
}

double UCTNode::get_blackevals() const {
    return m_blackevals;
}

double UCTNode::get_blackrollouts() const {
    return m_blackrollouts;
}

int UCTNode::get_rolloutwin() const{
    return m_rolloutwin;
}

void UCTNode::accumulate_eval(float eval) {
    atomic_add(m_blackevals, double(eval));
}

void UCTNode::accumulate_rollouts(float rollout) {
    atomic_add(m_blackrollouts, double(rollout));
}

void UCTNode::accumulate_rolloutwin(){
    atomic_add(m_rolloutwin, 1);
}

UCTNode* UCTNode::uct_select_child(int color, bool is_root) {
    LOCK(get_mutex(), lock);

    // Count parentvisits manually to avoid issues with transpositions.
    auto total_visited_policy = 0.0f;
    auto parentvisits = size_t{0};
    for (const auto& child : m_children) {
        if (child.valid()) {
            parentvisits += child.get_visits();
            if (child.get_visits() > 0) {
                total_visited_policy += child.get_score();
            }
        }
    }

    auto numerator = std::sqrt(double(parentvisits));
    auto fpu_reduction = 0.0f;
    // Lower the expected eval for moves that are likely not the best.
    // Do not do this if we have introduced noise at this node exactly
    // to explore more.
    if (!is_root || !cfg_noise) {
        fpu_reduction = cfg_fpu_reduction * std::sqrt(total_visited_policy);
    }
    // Estimated eval for unknown nodes = original parent NN eval - reduction
    auto fpu_eval = get_net_eval(color) - fpu_reduction;

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value = std::numeric_limits<double>::lowest();

    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }

        auto winrate = fpu_eval;
        if (child.get_visits() > 0) {
            //if(cfg_use_rollout)
                winrate = child.get_nn_eval(color)*lambda+child.get_rollouts(color)*(1-lambda);
            //else
                //winrate = child.get_nn_eval(color);
        }
		
	    auto psa = child.get_score();
		
        auto denom = 1.0 + child.get_visits();
        auto puct = cfg_puct * psa * (numerator / denom);
        auto value = winrate + puct;
        assert(value > std::numeric_limits<double>::lowest());

        if (value > best_value) {
            best_value = value;
            best = &child;
        }
    }

    assert(best != nullptr);
    best->inflate();
    return best->get();
}

UCTNode* UCTNode::max_p_child() {
    LOCK(get_mutex(), lock);

    auto best = static_cast<UCTNodePointer*>(nullptr);
    auto best_value = std::numeric_limits<double>::lowest();

    for (auto& child : m_children) {
        if (!child.active()) {
            continue;
        }
        auto value = child.get_score();
        assert(value > std::numeric_limits<double>::lowest());
        if(value > 0.5){
            best = &child;
            break;
        }

        if (value > best_value) {
            best_value = value;
            best = &child;
        }
    }
    //myprintf("%.4f ",best_value);
    assert(best != nullptr);
    best->inflate();
    return best->get();
}

class NodeComp : public std::binary_function<UCTNodePointer&,
                                             UCTNodePointer&, bool> {
public:
    NodeComp(int color) : m_color(color) {};
    bool operator()(const UCTNodePointer& a,
                    const UCTNodePointer& b) {
        // if visits are not same, sort on visits
        if (a.get_visits() != b.get_visits()) {
            return a.get_visits() < b.get_visits();
        }

        // neither has visits, sort on prior score
        if (a.get_visits() == 0) {
            return a.get_score() < b.get_score();
        }

        // both have same non-zero number of visits
        //if(cfg_use_rollout)
            return (a.get_nn_eval(m_color)*lambda+a.get_rollouts(m_color)*(1-lambda)) < (b.get_nn_eval(m_color)*lambda+b.get_rollouts(m_color)*(1-lambda));
        //else
            //return a.get_nn_eval(m_color) < b.get_nn_eval(m_color);
    }
private:
    int m_color;
};

void UCTNode::sort_children(int color) {
    LOCK(get_mutex(), lock);
    std::stable_sort(rbegin(m_children), rend(m_children), NodeComp(color));
}

UCTNode& UCTNode::get_best_root_child(int color) {
    LOCK(get_mutex(), lock);
    assert(!m_children.empty());

    auto ret = std::max_element(begin(m_children), end(m_children),
                                NodeComp(color));
    ret->inflate();
    return *(ret->get());
}

size_t UCTNode::count_nodes() const {
    auto nodecount = size_t{0};
    nodecount += m_children.size();
    for (auto& child : m_children) {
        if (child.get_visits() > 0) {
            nodecount += child->count_nodes();
        }
    }
    return nodecount;
}

void UCTNode::invalidate() {
    m_status = INVALID;
}

void UCTNode::set_active(const bool active) {
    if (valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

bool UCTNode::valid() const {
    return m_status != INVALID;
}

bool UCTNode::active() const {
    return m_status == ACTIVE;
}

////////////////////////////////////////////
float UCTNode::calculateRollout(GameState& state) {
    //float rollout_value=0;
    FullBoard board_cpy;
    board_cpy=state.board;
    //rollout_value+=playout(board_cpy, lgr, state.m_komi);
    
    return Playout(board_cpy, state.m_komi);
}
int UCTNode::Playout(FullBoard& b, double komi) {

	int next_move;
	int prev_move = VNULL;
	int pl = b.my;

	while (b.move_cnt <= 720) {
		next_move = b.SelectMove();
		// 2��A���Ńp�X�̏ꍇ�A�I�ǁD
		// Break in case of 2 consecutive pass.
		if (next_move==PASS_AQ && prev_move==PASS_AQ) break;
		prev_move = next_move;
	}

	prev_move = VNULL;
	while (b.move_cnt <= 720) {
		next_move = b.SelectRandomMove();
		if (next_move==PASS_AQ && prev_move==PASS_AQ) break;
		prev_move = next_move;
	}

	// Return the result.
	return Win(b, pl, komi);

}
/*
int  UCTNode::playout(FullBoard& b, LGR& lgr, double komi){
///*
	int next_move;
	int prev_move = VNULL;
	int pl = b.my;
	std::array<int, 4> lgr_seed;
	std::vector<std::array<int, 3>> lgr_rollout_add[2];
	std::array<int, 3> lgr_rollout_seed;
	int update_v[2] = {VNULL, VNULL};
	double update_p[2] = {100, 25};

	while (b.move_cnt <= 720) {
		lgr_seed[0] = b.prev_ptn[0].bf;
		lgr_seed[1] = b.prev_move[b.her];
		lgr_seed[2] = b.prev_ptn[1].bf;
		lgr_seed[3] = b.prev_move[b.my];

		// �i�J�f�Ő΂���ꂽ�Ƃ��A�}���ɑł�
		// Forced move if removed stones is Nakade.
		if (b.response_move[0] != VNULL) {
			next_move = b.response_move[0];
			b.PlayLegal(next_move);
		}
		else{

			// lgr.policy�Ɋ܂܂��
			// Check whether lgr_seed is included in lgr.policy.
			auto itr = lgr.policy[b.my].find(lgr_seed);
			int v = VNULL;
			if (itr != lgr.policy[b.my].end()){
				v = itr->second;
				if(v < PASS_AQ && b.IsLegal(b.my, v) && !b.IsEyeShape(b.my, v) && !b.IsSeki(v)){
					if(b.prob[b.my][v] != 0){
						b.ReplaceProb(b.my, v, b.prob[b.my][v] * update_p[0]);
						update_v[0] = v;
					}
				}
			}

			v = VNULL;
			if(lgr_seed[1] < PASS_AQ && lgr_seed[3] < PASS_AQ){
				v = lgr.rollout[b.my][lgr_seed[1]][lgr_seed[3]];

				if(v < PASS_AQ){
					if(b.prob[b.my][v] != 0){
						b.ReplaceProb(b.my, v, b.prob[b.my][v] * update_p[1]);
						update_v[1] = v;
					}
				}
			}

			next_move = b.SelectMove();
            //printf(" selectmove ");

			// update_v�̎�̊m����ɖ߂�
			// Restore probability.
			for(int i=0;i<2;++i){
				if(update_v[i] != VNULL){
					if(b.prob[b.her][update_v[i]] != 0){
						b.ReplaceProb(b.her, update_v[i], b.prob[b.her][update_v[i]] / update_p[i]);
					}
					update_v[i] = VNULL;
				}
			}

		}

		if(lgr_seed[1] < PASS_AQ && lgr_seed[3] < PASS_AQ && next_move < PASS_AQ){
			lgr_rollout_seed[0] = lgr_seed[1];
			lgr_rollout_seed[1] = lgr_seed[3];
			lgr_rollout_seed[2] = next_move;
			lgr_rollout_add[b.her].push_back(lgr_rollout_seed);
		}

		// 2��A���Ńp�X�̏ꍇ�A�I��
		// Break in case of 2 consecutive pass.
		if (next_move==PASS_AQ && prev_move==PASS_AQ) break;
		prev_move = next_move;
	}

	prev_move = VNULL;
	while (b.move_cnt <= 720) {
        //printf(" randommove ");
		next_move = b.SelectRandomMove();
		if (next_move==PASS_AQ && prev_move==PASS_AQ) break;
		prev_move = next_move;
	}

	int win = Win(b, pl, komi);
	int win_pl = int(win == 1);
	int lose_pl = int(win != 1);

	for(auto& i:lgr_rollout_add[win_pl]){
		lgr.rollout[win_pl][i[0]][i[1]] = i[2];
	}

	for(auto& i:lgr_rollout_add[lose_pl]){
		if(lgr.rollout[lose_pl][i[0]][i[1]] == i[2]){
			lgr.rollout[lose_pl][i[0]][i[1]] = VNULL;
		}
	}

	// �I�ǐ}�̏��s��Ԃ�
	// Return the result.
    //std::cout<<win<<" ";
	return win;
}*/

int UCTNode::Win(FullBoard& b, int pl, double komi) {

	double score[2] = {0.0, 0.0};
	std::array<bool, EBVCNT> visited;
	std::fill(visited.begin(), visited.end(), false);

	// �Z�L�����邩�m�F. Check Seki.
	for(int i=0,i_max=b.empty_cnt;i<i_max;++i){
		int v = b.empty[i];
		if(b.IsSeki(v) && !visited[v]){
			// ���̋Ȃ���l�ڂ��m�F.
			// Check whether it is corner bent fours.
			int ren_idx[2] = {0,0};
			forEach4Nbr(v, v_nbr2, {
				if(b.color[v_nbr2] > 1){
					ren_idx[b.color[v_nbr2] - 2] = b.ren_idx[v_nbr2];
				}
			});
			bool is_bent4 = false;
			for(int j=0;j<2;++j){
				if(b.ren[ren_idx[j]].size == 3){
					int v_tmp = ren_idx[j];
					bool is_edge = true;
					bool is_conner = false;

					do{
						is_edge &= (DistEdge(v_tmp) == 1);
						if(!is_edge) break;
						if (	v_tmp == rtoe[0] 					||
								v_tmp == rtoe[BSIZE - 1] 			||
								v_tmp == rtoe[BSIZE * (BSIZE - 1)] 	||
								v_tmp == rtoe[BVCNT - 1]	){
							bool is_not_bnt = false;
							forEach4Nbr(v_tmp, v_nbr1, {
								// �G�΂̂Ƃ��A�Ȃ���4�ڂł͂Ȃ�
								// If the neighboring stone is an opponnent's one.
								is_not_bnt |= (b.color[v_nbr1] == int(j==0) + 2);
							});

							is_conner = !is_not_bnt;
						}

						v_tmp = b.next_ren_v[v_tmp];
					}while(v_tmp != ren_idx[j]);

					if(is_edge && is_conner){
						// �Ȃ���l�ڂ̂Ƃ��A4�ڑ��̒n�Ƃ���
						// Count all stones as that of the player of the bent fours.
						score[j] += b.ren[ren_idx[0]].size + b.ren[ren_idx[1]].size + 2.0;
						is_bent4 = true;
					}
				}
			}
			// visited��X�V. Update visited.
			int64 lib_bit;
			for(int i=0;i<6;++i){
				lib_bit = b.ren[ren_idx[0]].lib_bits[i];
				while(lib_bit != 0){
					int ntz = NTZ(lib_bit);
					int lib = rtoe[ntz + i * 64];
					visited[lib] = true;

					lib_bit ^= (0x1ULL << ntz);
				}
			}
			// �Ȃ���l�ڂ̂Ƃ�. If it bent fours exist.
			if(is_bent4){
				int v_tmp = ren_idx[0];
				do{
					visited[v_tmp] = true;
					v_tmp = b.next_ren_v[v_tmp];
				}while(v_tmp != ren_idx[0]);
				v_tmp = ren_idx[1];
				do{
					visited[v_tmp] = true;
					v_tmp = b.next_ren_v[v_tmp];
				}while(v_tmp != ren_idx[1]);
			}
		}
	}

	for (auto i: rtoe) {
		int stone_color = b.color[i] - 2;
		if (!visited[i] && (stone_color >= 0)) {
			visited[i] = true;
			++score[stone_color];
			forEach4Nbr(i, v_nbr, {
				if (!visited[v_nbr] && b.color[v_nbr] == 0) {
					visited[v_nbr] = true;
					++score[stone_color];
				}
			});
		}
	}

	// �����p�X�񐔂̍��A�����Ō�ɒ����+1
	// Correction factor of PASS. Add one if the last move is black.
	int pass_corr = b.pass_cnt[0] - b.pass_cnt[1] + int((b.move_cnt%2)!=0);
	double abs_score = score[1] - score[0] - komi - pass_corr * int(japanese_rule);

	// ������->0, ���ԍ�����->1�A���ԍ�����-> -1��Ԃ�
	// Return 0 if white wins, 1 if black wins and it's black's turn and else -1.
    //if(cfg_rollout_mode)
	    //return int(abs_score > 0)*(int(pl == 1) - int(pl == 0));
    //else
        //return int(abs_score > 0);
    if(abs_score > 0)
        return 1;
    else
        return 0;

}
