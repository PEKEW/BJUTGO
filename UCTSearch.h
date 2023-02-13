

#ifndef UCTSEARCH_H_INCLUDED
#define UCTSEARCH_H_INCLUDED

#include <list>
#include <atomic>
#include <memory>
#include <string>
#include <tuple>
#include <future>

#include "ThreadPool.h"
#include "FastBoard.h"
#include "FastState.h"
#include "GameState.h"
#include "UCTNode.h"


class SearchResult {
public:
    SearchResult() = default;
    bool valid() const { return m_valid;  }
    float eval() const { return m_eval;  }
    float rollout() const { return m_rollout;  }
    static SearchResult from_eval(float eval,float rollout) {
        return SearchResult(eval,rollout);
    }
    static SearchResult from_score(float board_score) {
        if (board_score > 0.0f) {
            return SearchResult(1.0f,1.0f);
        } else if (board_score < 0.0f) {
            return SearchResult(0.0f,0.0f);
        } else {
            return SearchResult(0.5f,0.5f);
        }
    }
private:
    explicit SearchResult(float eval,float rollout)
        : m_valid(true), m_eval(eval),m_rollout(rollout) {}
    bool m_valid{false};
    float m_eval{0.0f};
    float m_rollout{0.0f};
};

namespace TimeManagement {
    enum enabled_t {
        AUTO = -1, OFF = 0, ON = 1, FAST = 2
    };
};

class UCTSearch {
public:
    /*
        Depending on rule set and state of the game, we might
        prefer to pass, or we might prefer not to pass unless
        it's the last resort. Same for resigning.
    */
    using passflag_t = int;
    static constexpr passflag_t NORMAL   = 0;
    static constexpr passflag_t NOPASS   = 1 << 0;
    static constexpr passflag_t NORESIGN = 1 << 1;
    std::mutex  mtx_lgr;
    /////////////////////////////////////////////////////////////////////////
   // LGR lgr;
    ////////////////////////////////////////////////////////////////////////

    /*
        Maximum size of the tree in memory. Nodes are about
        48 bytes, so limit to ~1.2G on 32-bits and about 5.5G
        on 64-bits.
    */
    static constexpr auto MAX_TREE_SIZE =
        (sizeof(void*) == 4 ? 25'000'000 : 100'000'000);

    /*
        Value representing unlimited visits or playouts. Due to
        concurrent updates while multithreading, we need some
        headroom within the native type.
    */
    static constexpr auto UNLIMITED_PLAYOUTS =
        std::numeric_limits<int>::max() / 2;

    UCTSearch(GameState& g);
    int think(int color, passflag_t passflag = NORMAL);
    void set_playout_limit(int playouts);
    void set_visit_limit(int visits);
    void ponder();
    bool is_running() const;
    void increment_playouts();
    SearchResult play_simulation(GameState& currstate, UCTNode* const node);

private:
    float get_min_psa_ratio() const;
    void dump_stats(FastState& state, UCTNode& parent);
    void tree_stats(const UCTNode& node);
    std::string get_pv(FastState& state, UCTNode& parent);
    void dump_analysis(int playouts);
    bool should_resign(passflag_t passflag, float bestscore);
    bool have_alternate_moves(int elapsed_centis, int time_for_move);
    int est_playouts_left(int elapsed_centis, int time_for_move) const;
    size_t prune_noncontenders(int elapsed_centis = 0, int time_for_move = 0);
    bool stop_thinking(int elapsed_centis = 0, int time_for_move = 0) const;
    int get_best_move(passflag_t passflag);
    void update_root();
    bool advance_to_new_rootstate();

    GameState & m_rootstate;
    std::unique_ptr<GameState> m_last_rootstate;
    std::unique_ptr<UCTNode> m_root;
    std::atomic<int> m_nodes{0};
    std::atomic<int> m_playouts{0};
    std::atomic<bool> m_run{false};
    int m_maxplayouts;
    int m_maxvisits;

    std::list<Utils::ThreadGroup> m_delete_futures;
};

class UCTWorker {
public:
    UCTWorker(GameState & state, UCTSearch * search, UCTNode * root)
      : m_rootstate(state), m_search(search), m_root(root) {}
    void operator()();
private:
    GameState & m_rootstate;
    UCTSearch * m_search;
    UCTNode * m_root;
};

#endif
