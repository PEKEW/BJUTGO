

#include "config.h"

#include <atomic>
#include <memory>
#include <cassert>
#include <cstring>

#include "UCTNode.h"

UCTNodePointer::~UCTNodePointer() {
    if (is_inflated()) {
        delete read_ptr();
    }
}

UCTNodePointer::UCTNodePointer(UCTNodePointer&& n) {
    if (is_inflated()) {
        delete read_ptr();
    }
    m_data = n.m_data;
    n.m_data = 1; // non-inflated garbage
}

UCTNodePointer::UCTNodePointer(std::int16_t vertex, float score) {
    std::uint32_t i_score;
    auto i_vertex = static_cast<std::uint16_t>(vertex);
    std::memcpy(&i_score, &score, sizeof(i_score));

    m_data =  (static_cast<std::uint64_t>(i_score)  << 32)
            | (static_cast<std::uint64_t>(i_vertex) << 16) | 1ULL;
}

UCTNodePointer& UCTNodePointer::operator=(UCTNodePointer&& n) {
    if (is_inflated()) {
        delete read_ptr();
    }
    m_data = n.m_data;
    n.m_data = 1;

    return *this;
}

void UCTNodePointer::inflate() const {
    if (is_inflated()) return;
    m_data = reinterpret_cast<std::uint64_t>(
        new UCTNode(read_vertex(), read_score()));
}

bool UCTNodePointer::valid() const {
    if (is_inflated()) return read_ptr()->valid();
    return true;
}

int UCTNodePointer::get_visits() const {
    if (is_inflated()) return read_ptr()->get_visits();
    return 0;
}

float UCTNodePointer::get_score() const {
    if (is_inflated()) return read_ptr()->get_score();
    return read_score();
}

bool UCTNodePointer::active() const {
    if (is_inflated()) return read_ptr()->active();
    return true;
}

float UCTNodePointer::get_nn_eval(int tomove) const {
    // this can only be called if it is an inflated pointer
    assert(is_inflated());
    return read_ptr()->get_nn_eval(tomove);
}

float UCTNodePointer::get_rollouts(int tomove) const {
    // this can only be called if it is an inflated pointer
    assert(is_inflated());
    return read_ptr()->get_rollouts(tomove);
}

int UCTNodePointer::get_move() const {
    if (is_inflated()) return read_ptr()->get_move();
    return read_vertex();
}
