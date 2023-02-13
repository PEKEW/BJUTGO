

#ifndef UCTNODEPOINTER_H_INCLUDED
#define UCTNODEPOINTER_H_INCLUDED

#include "config.h"

#include <atomic>
#include <memory>
#include <cassert>
#include <cstring>

#include "SMP.h"

class UCTNode;

// 'lazy-initializable' version of std::unique_ptr<UCTNode>.
// When a UCTNodePointer is constructed, the constructor arguments
// are stored instead of constructing the actual UCTNode instance.
// Later when the UCTNode is needed, the external code calls inflate()
// which actually constructs the UCTNode. Basically, this is a 'tagged union'
// of:
//  - std::unique_ptr<UCTNode> pointer;
//  - std::pair<float, std::int16_t> args;

// WARNING : inflate() is not thread-safe and hence has to be protected
// by an external lock.

class UCTNodePointer {
private:
    // the raw storage used here.
    // if bit 0 is 0, m_data is the actual pointer.
    // if bit 0 is 1, bit [31:16] is the vertex value, bit [63:32] is the score.
    // (C-style bit fields and unions are not portable)
    mutable uint64_t m_data = 1;

    UCTNode * read_ptr() const {
        assert(is_inflated());
        return reinterpret_cast<UCTNode*>(m_data);
    }

    std::int16_t read_vertex() const {
        assert(!is_inflated());
        return static_cast<std::int16_t>(m_data >> 16);
    }

    float read_score() const {
        static_assert(sizeof(float) == 4,
            "This code assumes floats are 32-bit");
        assert(!is_inflated());

        auto x = static_cast<std::uint32_t>(m_data >> 32);
        float ret;
        std::memcpy(&ret, &x, sizeof(ret));
        return ret;
    }

public:
    ~UCTNodePointer();
    UCTNodePointer(UCTNodePointer&& n);
    UCTNodePointer(std::int16_t vertex, float score);
    UCTNodePointer(const UCTNodePointer&) = delete;

    bool is_inflated() const {
        return (m_data & 1ULL) == 0;
    }

    // methods from std::unique_ptr<UCTNode>
    typename std::add_lvalue_reference<UCTNode>::type operator*() const{
        return *read_ptr();
    }
    UCTNode* operator->() const {
        return read_ptr();
    }
    UCTNode* get() const {
        return read_ptr();
    }
    UCTNodePointer& operator=(UCTNodePointer&& n);
    UCTNode * release() {
        auto ret = read_ptr();
        m_data = 1;
        return ret;
    }

    // construct UCTNode instance from the vertex/score pair
    void inflate() const;

    // proxy of UCTNode methods which can be called without
    // constructing UCTNode
    bool valid() const;
    int get_visits() const;
    float get_score() const;
    bool active() const;
    int get_move() const;
    // this can only be called if it is an inflated pointer
    float get_nn_eval(int tomove) const;
    float get_rollouts(int tomove) const;
};

#endif
