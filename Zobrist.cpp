

#include "config.h"
#include "Zobrist.h"
#include "Random.h"

std::random_device rd;
std::mt19937 mt_32(rd());
std::mt19937_64 mt_64(rd());

std::uniform_real_distribution<double> mt_double(0.0, 1.0);
std::uniform_int_distribution<int> mt_int8(0, 7);


std::array<std::array<std::uint64_t, FastBoard::MAXSQ>,     4> Zobrist::zobrist;
std::array<std::uint64_t, FastBoard::MAXSQ>                    Zobrist::zobrist_ko;
std::array<std::array<std::uint64_t, FastBoard::MAXSQ * 2>, 2> Zobrist::zobrist_pris;
std::array<std::uint64_t, 5>                                   Zobrist::zobrist_pass;

void Zobrist::init_zobrist(Random& rng) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < FastBoard::MAXSQ; j++) {
            Zobrist::zobrist[i][j] = rng.randuint64();
        }
    }

    for (int j = 0; j < FastBoard::MAXSQ; j++) {
        Zobrist::zobrist_ko[j] = rng.randuint64();
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < FastBoard::MAXSQ * 2; j++) {
            Zobrist::zobrist_pris[i][j] = rng.randuint64();
        }
    }

    for (int i = 0; i < 5; i++) {
        Zobrist::zobrist_pass[i]  = rng.randuint64();
    }
}
