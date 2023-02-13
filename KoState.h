
#ifndef KOSTATE_H_INCLUDED
#define KOSTATE_H_INCLUDED

#include "config.h"

#include <vector>

#include "FastState.h"
#include "FullBoard.h"

class KoState : public FastState {
public:
    void init_game(int size, float komi);
    bool superko(void) const;
    void reset_game();

    void play_move(int color, int vertex);
    void play_move(int vertex);

private:
    std::vector<std::uint64_t> m_ko_hash_history;
};

#endif
