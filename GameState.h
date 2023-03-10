

#ifndef GAMESTATE_H_INCLUDED
#define GAMESTATE_H_INCLUDED

#include <memory>
#include <string>
#include <vector>

#include "FastState.h"
#include "FullBoard.h"
#include "KoState.h"
#include "TimeControl.h"

class GameState : public KoState {
public:
    explicit GameState() = default;
    explicit GameState(const KoState* rhs) {
        // Copy in fields from base class.
        *(static_cast<KoState*>(this)) = *rhs;
        anchor_game_history();
    }
    void init_game(int size, float komi);
    void reset_game();
    bool set_fixed_handicap(int stones);
    int set_fixed_handicap_2(int stones);
    void place_free_handicap(int stones);
    void anchor_game_history(void);

    void rewind(void); /* undo infinite */
    bool undo_move(void);
    bool forward_move(void);
    const FullBoard& get_past_board(int moves_ago) const;

    void play_move(int color, int vertex);
    void play_move(int vertex);
    bool play_textmove(const std::string& color,
                       const std::string& vertex);

    void start_clock(int color);
    void stop_clock(int color);
    TimeControl& get_timecontrol();
    void set_timecontrol(int maintime, int byotime, int byostones,
                         int byoperiods);
    void set_timecontrol(TimeControl tmc);
    void adjust_time(int color, int time, int stones);

    void display_state();
    bool has_resigned() const;
    int who_resigned() const;
    
private:
    bool valid_handicap(int stones);

    std::vector<std::shared_ptr<const KoState>> game_history;
    TimeControl m_timecontrol;
    int m_resigned{FastBoard::EMPTY};
};

#endif
