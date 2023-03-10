
#include "config.h"
#include "UCTSearch.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <algorithm>

#include "FastBoard.h"
#include "FastState.h"
#include "FullBoard.h"
#include "GTP.h"
#include "GameState.h"
#include "TimeControl.h"
#include "Timing.h"
#include "Training.h"
#include "Utils.h"

#include <iostream>
using namespace Utils;

constexpr int UCTSearch::UNLIMITED_PLAYOUTS;

UCTSearch::UCTSearch(GameState& g)
    : m_rootstate(g) {
    set_playout_limit(cfg_max_playouts);
    set_visit_limit(cfg_max_visits);
    m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    ////////////////////////////////////////////////
    //lgr.Clear();
    /////////////////////////////////////////////////

}

bool UCTSearch::advance_to_new_rootstate() {
    if (!m_root || !m_last_rootstate) {
        // No current state
        return false;
    }

    if (m_rootstate.get_komi() != m_last_rootstate->get_komi()) {
        return false;
    }

    auto depth =
        int(m_rootstate.get_movenum() - m_last_rootstate->get_movenum());

    if (depth < 0) {
        return false;
    }


    auto test = std::make_unique<GameState>(m_rootstate);
    for (auto i = 0; i < depth; i++) {
        test->undo_move();
    }

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // m_rootstate and m_last_rootstate don't match
        return false;
    }

    // Make sure that the nodes we destroyed the previous move are
    // in fact destroyed.
    while (!m_delete_futures.empty()) {
        m_delete_futures.front().wait_all();
        m_delete_futures.pop_front();
    }

    // Try to replay moves advancing m_root
    for (auto i = 0; i < depth; i++) {
        ThreadGroup tg(thread_pool);

        test->forward_move();
        const auto move = test->get_last_move();

        auto oldroot = std::move(m_root);
        m_root = oldroot->find_child(move);

		
		
        
        // Lazy tree destruction.  Instead of calling the destructor of the
        // old root node on the main thread, send the old root to a separate
        // thread and destroy it from the child thread.  This will save a
        // bit of time when dealing with large trees.
        auto p = oldroot.release();
        tg.add_task([p]() { delete p; });
        m_delete_futures.push_back(std::move(tg));

        if (!m_root) {
            // Tree hasn't been expanded this far
            return false;
        }
        m_last_rootstate->play_move(move);
    }

    assert(m_rootstate.get_movenum() == m_last_rootstate->get_movenum());

    if (m_last_rootstate->board.get_hash() != test->board.get_hash()) {
        // Can happen if user plays multiple moves in a row by same player
        return false;
    }

    return true;
}

void UCTSearch::update_root() {
    // Definition of m_playouts is playouts per search call.
    // So reset this count now.
    m_playouts = 0;

#ifndef NDEBUG
    auto start_nodes = m_root->count_nodes();
#endif

    if (!advance_to_new_rootstate() || !m_root) {
        m_root = std::make_unique<UCTNode>(FastBoard::PASS, 0.0f);
    }
    // Clear last_rootstate to prevent accidental use.
    m_last_rootstate.reset(nullptr);

    // Check how big our search tree (reused or new) is.
    m_nodes = m_root->count_nodes();
    
#ifndef NDEBUG
    if (m_nodes > 0) {
        myprintf("update_root, %d -> %d nodes (%.1f%% reused)\n",
            start_nodes, m_nodes.load(), 100.0 * m_nodes.load() / start_nodes);
    }
#endif
}

float UCTSearch::get_min_psa_ratio() const {
    const auto mem_full = m_nodes / static_cast<float>(MAX_TREE_SIZE);
    // If we are halfway through our memory budget, start trimming
    // moves with very low policy priors.
    if (mem_full > 0.5f) {
        // Memory is almost exhausted, trim more aggressively.
        if (mem_full > 0.95f) {
            return 0.01f;
        }
        return 0.001f;
    }
    return 0.0f;
}

SearchResult UCTSearch::play_simulation(GameState & currstate,
                                        UCTNode* const node) {
    
    const auto color = currstate.get_to_move();
    auto result = SearchResult{};
    auto rollout_value=0;
    node->virtual_loss();

    if (node->expandable()) {
        if (currstate.get_passes() >= 2) {
            auto score = currstate.final_score();
            result = SearchResult::from_score(score);
        } else if (m_nodes < MAX_TREE_SIZE) {
            float eval;
            const auto had_children = node->has_children();
            
            const auto success =
                node->create_children(m_nodes, currstate, eval,
                                      get_min_psa_ratio());

            if (!had_children && success) {
                if(cfg_use_rollout){
                    
                    rollout_value = node->calculateRollout(currstate);
                    
                    
                    //if(cfg_rollout_mode){
                        //double win_bias = (currstate.board.my == 0)? 0.5 : -0.5;
                        //rollout_value = 2.0 * (rollout_value + win_bias);
                        //std::cout<<rollout_value<<" ";
                    //}
                    //result = SearchResult::from_eval(eval,rollout_value);
                }
                //else
                result = SearchResult::from_eval(eval,rollout_value);
            }
        }
    }

    if (node->has_children() && !result.valid()) {
        auto next = node->uct_select_child(color, node == m_root.get());
        auto move = next->get_move();

        currstate.play_move(move);
        if (move != FastBoard::PASS && currstate.superko()) {
            next->invalidate();
        } else {
            result = play_simulation(currstate, next);
        }
    }

    if (result.valid()) {
        node->update(result.eval(),result.rollout());
    }
    node->virtual_loss_undo();

    return result;
}

void UCTSearch::dump_stats(FastState & state, UCTNode & parent) {
    if (cfg_quiet || !parent.has_children()) {
        return;
    }

    const int color = state.get_to_move();

    // sort children, put best move on top
    parent.sort_children(color);

    if (parent.get_first_child()->first_visit()) {
        return;
    }

    int movecount = 0;
    for (const auto& node : parent.get_children()) {
        // Always display at least two moves. In the case there is
        // only one move searched the user could get an idea why.
        if (++movecount > 2 && !node->get_visits()) break;

        std::string move = state.move_to_text(node->get_move());
        FastState tmpstate = state;
        tmpstate.play_move(node->get_move());
        std::string pv = move + " " + get_pv(tmpstate, *node);
        if(cfg_use_rollout){
        myprintf("%4s -> %5d (W:%5.2f%%) (V:%5.2f%%) (R:%5.2f%%) (N:%5.2f%%) PV:%s\n",
                move.c_str(),
                node->get_visits(),
                node->get_visits() ? (node->get_rollouts(color)*(1-lambda)+node->get_nn_eval(color)*lambda)*100.0f: 0.0f,
                node->get_visits() ? node->get_nn_eval(color)*100.0f : 0.0f,
                node->get_visits() ? node->get_rollouts(color)*100.0f : 0.0f,
                node->get_score() * 100.0f,
                pv.c_str());

        }
        else{
            myprintf("%4s -> %5d (V:%5.2f%%) (N:%5.2f%%) PV:%s\n",
                move.c_str(),
                node->get_visits(),
                node->get_visits() ? node->get_nn_eval(color)*100.0f : 0.0f,
                node->get_score() * 100.0f,
                pv.c_str());
        }
        
    }
    tree_stats(parent);
}

void tree_stats_helper(const UCTNode& node, size_t depth,
                       size_t& nodes, size_t& non_leaf_nodes,
                       size_t& depth_sum, size_t& max_depth,
                       size_t& children_count) {
    nodes += 1;
    non_leaf_nodes += node.get_visits() > 1;
    depth_sum += depth;
    if (depth > max_depth) max_depth = depth;

    for (const auto& child : node.get_children()) {
        if (child.get_visits() > 0) {
            children_count += 1;
            tree_stats_helper(*(child.get()), depth+1,
                              nodes, non_leaf_nodes, depth_sum,
                              max_depth, children_count);
        } else {
            nodes += 1;
            depth_sum += depth+1;
            if (depth+1 > max_depth) max_depth = depth+1;
        }
    }
}

void UCTSearch::tree_stats(const UCTNode& node) {
    size_t nodes = 0;
    size_t non_leaf_nodes = 0;
    size_t depth_sum = 0;
    size_t max_depth = 0;
    size_t children_count = 0;
    tree_stats_helper(node, 0,
                      nodes, non_leaf_nodes, depth_sum,
                      max_depth, children_count);

    if (nodes > 0) {
        myprintf("%.1f average depth, %d max depth\n",
                 (1.0f*depth_sum) / nodes, max_depth);
        myprintf("%d non leaf nodes, %.2f average children\n",
                 non_leaf_nodes, (1.0f*children_count) / non_leaf_nodes);
    }
}

bool UCTSearch::should_resign(passflag_t passflag, float bestscore) {
    if (passflag & UCTSearch::NORESIGN) {
        // resign not allowed
        return false;
    }

    if (cfg_resignpct == 0) {
        // resign not allowed
        return false;
    }

    const size_t board_squares = m_rootstate.board.get_boardsize()
                               * m_rootstate.board.get_boardsize();
    const auto move_threshold = board_squares / 4;
    const auto movenum = m_rootstate.get_movenum();
    if (movenum <= move_threshold) {
        // too early in game to resign
        return false;
    }

    const auto color = m_rootstate.board.get_to_move();

    const auto is_default_cfg_resign = cfg_resignpct < 0;
    const auto resign_threshold =
        0.01f * (is_default_cfg_resign ? 10 : cfg_resignpct);
    if (bestscore > resign_threshold) {
        // eval > cfg_resign
        return false;
    }

    if ((m_rootstate.get_handicap() > 0)
            && (color == FastBoard::WHITE)
            && is_default_cfg_resign) {
        const auto handicap_resign_threshold =
            resign_threshold / (1 + m_rootstate.get_handicap());

        // Blend the thresholds for the first ~215 moves.
        auto blend_ratio = std::min(1.0f, movenum / (0.6f * board_squares));
        auto blended_resign_threshold = blend_ratio * resign_threshold
            + (1 - blend_ratio) * handicap_resign_threshold;
        if (bestscore > blended_resign_threshold) {
            // Allow lower eval for white in handicap games
            // where opp may fumble.
            return false;
        }
    }

    return true;
}

int UCTSearch::get_best_move(passflag_t passflag) {
    int color = m_rootstate.board.get_to_move();

    // Make sure best is first
    m_root->sort_children(color);

    // Check whether to randomize the best move proportional
    // to the playout counts, early game only.
    auto movenum = int(m_rootstate.get_movenum());
    if (movenum < cfg_random_cnt) {
        m_root->randomize_first_proportionally();
    }

    auto first_child = m_root->get_first_child();
    assert(first_child != nullptr);

    auto bestmove = first_child->get_move();
    //float bestscore;
    //if(cfg_use_rollout)
    //bestscore = first_child->get_nn_eval(color)*lambda+first_child->get_rollouts(color)*(1-lambda);
    //else
    //bestscore = first_child->get_nn_eval(color);
    auto bestscore = first_child->get_nn_eval(color);

    // do we want to fiddle with the best move because of the rule set?
    if (passflag & UCTSearch::NOPASS) {
        // were we going to pass?
        if (bestmove == FastBoard::PASS) {
            UCTNode * nopass = m_root->get_nopass_child(m_rootstate);

            if (nopass != nullptr) {
                myprintf("Preferring not to pass.\n");
                bestmove = nopass->get_move();
                if (nopass->first_visit()) {
                    bestscore = 1.0f;
                } else {
                    //if(cfg_use_rollout)
                        //bestscore = nopass->get_nn_eval(color)*lambda+nopass->get_rollouts(color)*(1-lambda);
                    //else
                        bestscore = nopass->get_nn_eval(color);
                }
            } else {
                myprintf("Pass is the only acceptable move.\n");
            }
        }
    } else {
        if (!cfg_dumbpass && bestmove == FastBoard::PASS) {
            // Either by forcing or coincidence passing is
            // on top...check whether passing loses instantly
            // do full count including dead stones.
            // In a reinforcement learning setup, it is possible for the
            // network to learn that, after passing in the tree, the two last
            // positions are identical, and this means the position is only won
            // if there are no dead stones in our own territory (because we use
            // Trump-Taylor scoring there). So strictly speaking, the next
            // heuristic isn't required for a pure RL network, and we have
            // a commandline option to disable the behavior during learning.
            // On the other hand, with a supervised learning setup, we fully
            // expect that the engine will pass out anything that looks like
            // a finished game even with dead stones on the board (because the
            // training games were using scoring with dead stone removal).
            // So in order to play games with a SL network, we need this
            // heuristic so the engine can "clean up" the board. It will still
            // only clean up the bare necessity to win. For full dead stone
            // removal, kgs-genmove_cleanup and the NOPASS mode must be used.
            float score = m_rootstate.final_score();
            // Do we lose by passing?
            if ((score > 0.0f && color == FastBoard::WHITE)
                ||
                (score < 0.0f && color == FastBoard::BLACK)) {
                myprintf("Passing loses :-(\n");
                // Find a valid non-pass move.
                UCTNode * nopass = m_root->get_nopass_child(m_rootstate);
                if (nopass != nullptr) {
                    myprintf("Avoiding pass because it loses.\n");
                    bestmove = nopass->get_move();
                    if (nopass->first_visit()) {
                        bestscore = 1.0f;
                    } else {
                        //if(cfg_use_rollout)
                            //bestscore = nopass->get_nn_eval(color)*lambda+nopass->get_rollouts(color)*(1-lambda);
                        //else
                        bestscore = nopass->get_nn_eval(color);

                    }
                } else {
                    myprintf("No alternative to passing.\n");
                }
            } else {
                myprintf("Passing wins :-)\n");
            }
        } else if (!cfg_dumbpass
                   && m_rootstate.get_last_move() == FastBoard::PASS) {
            // Opponents last move was passing.
            // We didn't consider passing. Should we have and
            // end the game immediately?
            float score = m_rootstate.final_score();
            // do we lose by passing?
            if ((score > 0.0f && color == FastBoard::WHITE)
                ||
                (score < 0.0f && color == FastBoard::BLACK)) {
                myprintf("Passing loses, I'll play on.\n");
            } else {
                myprintf("Passing wins, I'll pass out.\n");
                bestmove = FastBoard::PASS;
            }
        }
    }

    // if we aren't passing, should we consider resigning?
    if (bestmove != FastBoard::PASS) {
        if (should_resign(passflag, bestscore)) {
            myprintf("Eval (%.2f%%) looks bad. Resigning.\n",
                     100.0f * bestscore);
            bestmove = FastBoard::RESIGN;
        }
    }

    return bestmove;
}

std::string UCTSearch::get_pv(FastState & state, UCTNode& parent) {
    if (!parent.has_children()) {
        return std::string();
    }

    auto& best_child = parent.get_best_root_child(state.get_to_move());
    if (best_child.first_visit()) {
        return std::string();
    }
    auto best_move = best_child.get_move();
    auto res = state.move_to_text(best_move);

    state.play_move(best_move);

    auto next = get_pv(state, best_child);
    if (!next.empty()) {
        res.append(" ").append(next);
    }
    return res;
}

void UCTSearch::dump_analysis(int playouts) {
    if (cfg_quiet) {
        return;
    }

    FastState tempstate = m_rootstate;
    int color = tempstate.board.get_to_move();

    std::string pvstring = get_pv(tempstate, *m_root);
    float winrate = 100.0f * m_root->get_nn_eval(color);
    //float rolloutrate;
    //if(cfg_rollout_mode)
    float rollout = 100.0f * m_root->get_rollouts(color);
    //float rollout_winrate = 100.0f * m_root->get_rollouts(color);
    //else
    //    rolloutrate = 100.0f * m_root->get_rollouts(color);
    if(cfg_use_rollout){
        myprintf("Playouts:%5d (W:%5.2f%%) (V:%5.2f%%) (R:%5.2f%%) PV:%s\n",
                playouts,winrate*lambda+rollout*(1-lambda), winrate, rollout,  pvstring.c_str());
    }
    else{
        myprintf("Playouts:%5d (V:%5.2f%%) PV:%s\n",
                playouts, winrate, pvstring.c_str());
    }
}

bool UCTSearch::is_running() const {
    return m_run && m_nodes < MAX_TREE_SIZE;
}

int UCTSearch::est_playouts_left(int elapsed_centis, int time_for_move) const {
    auto playouts = m_playouts.load();
    const auto playouts_left =
        std::max(0, std::min(m_maxplayouts - playouts,
                             m_maxvisits - m_root->get_visits()));

    // Wait for at least 1 second and 100 playouts
    // so we get a reliable playout_rate.
    if (elapsed_centis < 100 || playouts < 100) {
        return playouts_left;
    }
    const auto playout_rate = 1.0f * playouts / elapsed_centis;
    const auto time_left = std::max(0, time_for_move - elapsed_centis);
    return std::min(playouts_left,
                    static_cast<int>(std::ceil(playout_rate * time_left)));
}

size_t UCTSearch::prune_noncontenders(int elapsed_centis, int time_for_move) {
    auto Nfirst = 0;
    // There are no cases where the root's children vector gets modified
    // during a multithreaded search, so it is safe to walk it here without
    // taking the (root) node lock.
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            Nfirst = std::max(Nfirst, node->get_visits());
        }
    }
    const auto min_required_visits =
        Nfirst - est_playouts_left(elapsed_centis, time_for_move);
    auto pruned_nodes = size_t{0};
    for (const auto& node : m_root->get_children()) {
        if (node->valid()) {
            const auto has_enough_visits =
                node->get_visits() >= min_required_visits;

            node->set_active(has_enough_visits);
            if (!has_enough_visits) {
                ++pruned_nodes;
            }
        }
    }

    assert(pruned_nodes < m_root->get_children().size());
    return pruned_nodes;
}

bool UCTSearch::have_alternate_moves(int elapsed_centis, int time_for_move) {
    if (cfg_timemanage == TimeManagement::OFF) {
        return true;
    }
    auto pruned = prune_noncontenders(elapsed_centis, time_for_move);
    if (pruned < m_root->get_children().size() - 1) {
        return true;
    }
    // If we cannot save up time anyway, use all of it. This
    // behavior can be overruled by setting "fast" time management,
    // which will cause DeepStone to quickly respond to obvious/forced moves.
    // That comes at the cost of some playing strength as she now cannot
    // think ahead about her next moves in the remaining time.
    auto my_color = m_rootstate.get_to_move();
    auto tc = m_rootstate.get_timecontrol();
    if (!tc.can_accumulate_time(my_color)
        || m_maxplayouts < UCTSearch::UNLIMITED_PLAYOUTS) {
        if (cfg_timemanage != TimeManagement::FAST) {
            return true;
        }
    }
    // In a timed search we will essentially always exit because
    // the remaining time is too short to let another move win, so
    // avoid spamming this message every move. We'll print it if we
    // save at least half a second.
    if (time_for_move - elapsed_centis > 50) {
        myprintf("%.1fs left, stopping early.\n",
                    (time_for_move - elapsed_centis) / 100.0f);
    }
    return false;
}

bool UCTSearch::stop_thinking(int elapsed_centis, int time_for_move) const {
    return m_playouts >= m_maxplayouts
           || m_root->get_visits() >= m_maxvisits
           || elapsed_centis >= time_for_move;
}

void UCTWorker::operator()() {
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = m_search->play_simulation(*currstate, m_root);
        if (result.valid()) {
            m_search->increment_playouts();
        }
    } while (m_search->is_running());
}

void UCTSearch::increment_playouts() {
    m_playouts++;
}

int UCTSearch::think(int color, passflag_t passflag) {
    // Start counting time for us
    m_rootstate.start_clock(color);

    // set up timing info
    Time start;

    update_root();
    // set side to move
    m_rootstate.board.set_to_move(color);

    m_rootstate.get_timecontrol().set_boardsize(
        m_rootstate.board.get_boardsize());
    auto time_for_move = m_rootstate.get_timecontrol().max_time_for_move(color, m_rootstate.get_movenum());

    myprintf("Thinking at most %.1f seconds...\n", time_for_move/100.0f);
    auto currstate = std::make_unique<GameState>(m_rootstate);
	if (cfg_use_rollout) {
		//if ((double)currstate->board.move_cnt <= 20){
			//lambda = 0.8;
		//}
		//else if((double)currstate->board.move_cnt > 20 && (double)currstate->board.move_cnt <= 320){
			lambda = 0.8 - 0.4 * std::min(1.0, std::max(0.0, ((double)currstate->board.move_cnt - 160) / (360 - 160)));
		//}
       // else{
            //lambda = 1.0;
        //}
        
	}
    myprintf("lambda:%.4f\n",lambda);
    // create a sorted list of legal moves (make sure we
    // play something legal and decent even in time trouble)
    m_root->prepare_root_node(color, m_nodes, m_rootstate);

    m_run = true;
    int cpus = cfg_num_threads;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cpus; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }

    bool keeprunning = true;
    int last_update = 0;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);

        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }

        Time elapsed;
        int elapsed_centis = Time::timediff_centis(start, elapsed);

        // output some stats every few seconds
        // check if we should still search
        if (elapsed_centis - last_update > 250) {
            last_update = elapsed_centis;
            dump_analysis(static_cast<int>(m_playouts));
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(elapsed_centis, time_for_move);
        keeprunning &= have_alternate_moves(elapsed_centis, time_for_move);
    } while (keeprunning);

    // stop the search
    m_run = false;
    tg.wait_all();

    // reactivate all pruned root children
    for (const auto& node : m_root->get_children()) {
        node->set_active(true);
    }

    m_rootstate.stop_clock(color);
    if (!m_root->has_children()) {
        return FastBoard::PASS;
    }

    // display search info
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);
    Training::record(m_rootstate, *m_root);

    Time elapsed;
    int elapsed_centis = Time::timediff_centis(start, elapsed);
    if (elapsed_centis+1 > 0) {
        myprintf("%d visits, %d nodes, %d playouts, %.0f n/s\n\n",
                 m_root->get_visits(),
                 static_cast<int>(m_nodes),
                 static_cast<int>(m_playouts),
                 (m_playouts * 100.0) / (elapsed_centis+1));
    }
    int bestmove = get_best_move(passflag);

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
    return bestmove;
}

void UCTSearch::ponder() {
    update_root();

    m_root->prepare_root_node(m_rootstate.board.get_to_move(),
                              m_nodes, m_rootstate);

    m_run = true;
    ThreadGroup tg(thread_pool);
    for (int i = 1; i < cfg_num_threads; i++) {
        tg.add_task(UCTWorker(m_rootstate, this, m_root.get()));
    }
    auto keeprunning = true;
    do {
        auto currstate = std::make_unique<GameState>(m_rootstate);
        auto result = play_simulation(*currstate, m_root.get());
        if (result.valid()) {
            increment_playouts();
        }
        keeprunning  = is_running();
        keeprunning &= !stop_thinking(0, 1);
    } while (!Utils::input_pending() && keeprunning);

    // stop the search
    m_run = false;
    tg.wait_all();

    // display search info
    myprintf("\n");
    dump_stats(m_rootstate, *m_root);

    myprintf("\n%d visits, %d nodes\n\n", m_root->get_visits(), m_nodes.load());

    // Copy the root state. Use to check for tree re-use in future calls.
    m_last_rootstate = std::make_unique<GameState>(m_rootstate);
}

void UCTSearch::set_playout_limit(int playouts) {
    static_assert(std::is_convertible<decltype(playouts),
                                      decltype(m_maxplayouts)>::value,
                  "Inconsistent types for playout amount.");
    m_maxplayouts = std::min(playouts, UNLIMITED_PLAYOUTS);
}

void UCTSearch::set_visit_limit(int visits) {
    static_assert(std::is_convertible<decltype(visits),
                                      decltype(m_maxvisits)>::value,
                  "Inconsistent types for visits amount.");
    // Limit to type max / 2 to prevent overflow when multithreading.
    m_maxvisits = std::min(visits, UNLIMITED_PLAYOUTS);
}
