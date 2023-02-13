

#ifndef UCTNODE_H_INCLUDED
#define UCTNODE_H_INCLUDED

#include "config.h"

#include <atomic>
#include <memory>
#include <vector>
#include <cassert>
#include <cstring>

#include "GameState.h"
#include "Network.h"
#include "SMP.h"
#include "UCTNodePointer.h"


///////////////////////////////////////////////////////////////
#pragma once

#include <array>
#include <unordered_map>
#include <string.h>

// �n�b�V���֐��̓��ꉻ
// Specialization of hash function.
namespace std {
	template<typename T>
	struct hash<array<T, 4>> {
		size_t operator()(const array<T, 4>& p) const {
			size_t seed = 0;
			hash<T> h;
			seed ^= h(p[0]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= h(p[1]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= h(p[2]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			seed ^= h(p[3]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};
}


struct Statistics{
	int game[3];
	int stone[3][EBVCNT];
	int owner[2][EBVCNT];

	Statistics(){ Clear(); }
	Statistics(const Statistics& other){ *this = other; }

	void Clear(){
		for(auto& g:game) g = 0;
		for(auto& si:stone) for(auto& s:si) s = 0;
		for(auto& oi:owner) for(auto& o:oi) o = 0;
	}

	Statistics& operator=(const Statistics& rhs){
		std::memcpy(game, rhs.game, sizeof(game));
		std::memcpy(stone, rhs.stone, sizeof(stone));
		std::memcpy(owner, rhs.owner, sizeof(owner));
		return *this;
	}

	Statistics& operator+=(const Statistics& rhs){
		for(int i=0;i<3;++i) game[i] += rhs.game[i];
		for(int i=0;i<3;++i){
			for(int j=0;j<EBVCNT;++j) stone[i][j] += rhs.stone[i][j];
		}
		for(int i=0;i<2;++i){
			for(int j=0;j<EBVCNT;++j) owner[i][j] += rhs.owner[i][j];
		}
		return *this;
	}

	Statistics& operator-=(const Statistics& rhs){
		for(int i=0;i<3;++i) game[i] -= rhs.game[i];
		for(int i=0;i<3;++i){
			for(int j=0;j<EBVCNT;++j) stone[i][j] -= rhs.stone[i][j];
		}
		for(int i=0;i<2;++i){
			for(int j=0;j<EBVCNT;++j) owner[i][j] -= rhs.owner[i][j];
		}
		return *this;
	}
};

/*
struct LGR{

	// PolicyNet�ɂ���ē���ꂽ�őP���ێ�����
	// Container that has LGR moves obtained by PolicyNet.
	//     key: { previous 12-point pattern, previous move, ... }
	//     value: best move
	std::array<std::unordered_map<std::array<int,4>, int>, 2> policy;

	// �v���C�A�E�g�ɂ���ē���ꂽ���ێ�����z��
	// Array that has LGR moves obtained by rollout.
	std::array<std::array<std::array<int, EBVCNT>, EBVCNT>, 2> rollout;

	LGR(){ Clear(); }
	LGR(const LGR& other){ *this = other; }

	void Clear(){
		for(auto& p1:policy) p1.clear();
		for(auto& r1:rollout) for(auto& r2:r1) for(auto& r3:r2){ r3 = VNULL; }
	}

	LGR& operator=(const LGR& rhs){
		policy[0] = rhs.policy[0];
		policy[1] = rhs.policy[1];
		for(int i=0;i<2;++i){
			for(int j=0;j<EBVCNT;++j){
				for(int k=0;k<EBVCNT;++k){
					rollout[i][j][k] = rhs.rollout[i][j][k];
				}
			}
		}
		return *this;
	}

	LGR& operator+=(const LGR& rhs){
		policy[0].insert(rhs.policy[0].begin(),rhs.policy[0].end());
		policy[1].insert(rhs.policy[1].begin(),rhs.policy[1].end());
		for(int i=0;i<2;++i){
			for(int j=0;j<EBVCNT;++j){
				for(int k=0;k<EBVCNT;++k){
					if(rhs.rollout[i][j][k] != VNULL)
						rollout[i][j][k] = rhs.rollout[i][j][k];
				}
			}
		}
		return *this;
	}
};
*/

extern bool japanese_rule;
/////////////////////////////////////////////////////////////////////////////////////


class UCTNode {
public:
    // When we visit a node, add this amount of virtual losses
    // to it to encourage other CPUs to explore other parts of the
    // search tree.
    static constexpr auto VIRTUAL_LOSS_COUNT = 3;
    // Defined in UCTNode.cpp
    explicit UCTNode(int vertex, float score);
    UCTNode() = delete;
    ~UCTNode() = default;
    bool create_children(std::atomic<int>& nodecount,
                         GameState& state, float& eval,
                         float min_psa_ratio = 0.0f);

    const std::vector<UCTNodePointer>& get_children() const;
    void sort_children(int color);
    UCTNode& get_best_root_child(int color);
    UCTNode* uct_select_child(int color, bool is_root);
	UCTNode* max_p_child();

    size_t count_nodes() const;
    SMP::Mutex& get_mutex();
    bool first_visit() const;
    bool has_children() const;
    bool expandable(const float min_psa_ratio = 0.0f) const;
    void invalidate();
    void set_active(const bool active);
    bool valid() const;
    bool active() const;
    int get_move() const;
    int get_visits() const;
    float get_score() const;
    void set_score(float score);
    float get_nn_eval(int tomove) const;
	float get_rollouts(int tomove) const;
	float get_rollout_winrate(int tomove) const;
    float get_net_eval(int tomove) const;
    void virtual_loss(void);
    void virtual_loss_undo(void);
    void update(float eval,float rollout);

    // Defined in UCTNodeRoot.cpp, only to be called on m_root in UCTSearch
    void randomize_first_proportionally();
    void prepare_root_node(int color,
                           std::atomic<int>& nodecount,
                           GameState& state);

    UCTNode* get_first_child() const;
    UCTNode* get_nopass_child(FastState& state) const;
    std::unique_ptr<UCTNode> find_child(const int move);
    void inflate_all_children();
    //////////////////////////////////////////////////////
    float calculateRollout(GameState& state);
	//int playout(FullBoard& board, LGR& lgr, double komi);
	int Playout(FullBoard& b, double komi);
	int Win(FullBoard& board, int pl, double komi);
    /////////////////////////////////////////////////////


private:
    enum Status : char {
        INVALID, // superko
        PRUNED,
        ACTIVE
    };
    void link_nodelist(std::atomic<int>& nodecount,
                       std::vector<Network::ScoreVertexPair>& nodelist,
                       float min_psa_ratio);
    double get_blackevals() const;
	double get_blackrollouts() const;
	int get_rolloutwin() const;
    void accumulate_eval(float eval);
	void accumulate_rollouts(float rollout);
	void accumulate_rolloutwin();
    void kill_superkos(const KoState& state);
    void dirichlet_noise(float epsilon, float alpha);

    // Note : This class is very size-sensitive as we are going to create
    // tens of millions of instances of these.  Please put extra caution
    // if you want to add/remove/reorder any variables here.

    // Move
    std::int16_t m_move;
    // UCT
    std::atomic<std::int16_t> m_virtual_loss{0};
    std::atomic<int> m_visits{0};
    // UCT eval
    float m_score;
    // Original net eval for this node (not children).
    float m_net_eval{0.0f};
	std::atomic<int> m_rolloutwin{0};
    std::atomic<double> m_blackevals{0.0};
    std::atomic<double> m_blackrollouts{0.0};
    std::atomic<Status> m_status{ACTIVE};
    // Is someone adding scores to this node?
    bool m_is_expanding{false};
    SMP::Mutex m_nodemutex;

    // Tree data
    std::atomic<float> m_min_psa_ratio_children{2.0f};
    std::vector<UCTNodePointer> m_children;
};

#endif
