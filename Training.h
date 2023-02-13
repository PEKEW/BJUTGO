

#ifndef TRAINING_H_INCLUDED
#define TRAINING_H_INCLUDED

#include "config.h"

#include <bitset>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "GameState.h"
#include "Network.h"
#include "UCTNode.h"

class TimeStep {
public:
    using BoardPlane = std::bitset<BOARD_SQUARES>;
    using NNPlanes = std::vector<BoardPlane>;
    NNPlanes planes;
    std::vector<float> probabilities;
    int to_move;
    float net_winrate;
    float root_uct_winrate;
    float child_uct_winrate;
    int bestmove_visits;
};

std::ostream& operator<< (std::ostream& stream, const TimeStep& timestep);
std::istream& operator>> (std::istream& stream, TimeStep& timestep);

class OutputChunker {
public:
    OutputChunker(const std::string& basename, bool compress = false);
    ~OutputChunker();
    void append(const std::string& str);

    // Group this many games in a batch.
    static constexpr size_t CHUNK_SIZE = 32;
private:
    std::string gen_chunk_name() const;
    void flush_chunks();
    size_t m_game_count{0};
    size_t m_chunk_count{0};
    std::string m_buffer;
    std::string m_basename;
    bool m_compress{false};
};

class Training {
public:
    static void clear_training();
    static void dump_training(int winner_color,
                              const std::string& out_filename);
    static void dump_debug(const std::string& out_filename);
    static void record(GameState& state, UCTNode& node);

    static void dump_supervised(const std::string& sgf_file,
                                const std::string& out_filename);
    static void save_training(const std::string& filename);
    static void load_training(const std::string& filename);

private:
    static TimeStep::NNPlanes get_planes(const GameState* const state);
    static void process_game(GameState& state, size_t& train_pos, int who_won,
                             const std::vector<int>& tree_moves,
                             OutputChunker& outchunker);
    static void dump_training(int winner_color,
                              OutputChunker& outchunker);
    static void dump_debug(OutputChunker& outchunker);
    static void save_training(std::ofstream& out);
    static void load_training(std::ifstream& in);
    static std::vector<TimeStep> m_data;
};

#endif
