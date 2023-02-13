

#ifndef FASTSTATE_H_INCLUDED
#define FASTSTATE_H_INCLUDED

#include <cstddef>
#include <array>
#include <string>
#include <vector>

#include "FullBoard.h"

class FastState {
public:
    void init_game(int size, float komi);
    void reset_game();
    void reset_board();

    void play_move(int vertex);

    bool is_move_legal(int color, int vertex);

    void set_komi(float komi);
    float get_komi() const;
    void set_handicap(int hcap);
    int get_handicap() const;
    int get_passes() const;
    int get_to_move() const;
    void set_to_move(int tomove);
    void set_passes(int val);
    void increment_passes();

    float final_score() const;

    size_t get_movenum() const;
    int get_last_move() const;
    void display_state();
    std::string move_to_text(int move);

    FullBoard board;

    float m_komi;
    int m_handicap;
    int m_passes;
    int m_komove;
    size_t m_movenum;
    int m_lastmove;

protected:
    void play_move(int color, int vertex);
};

#endif
