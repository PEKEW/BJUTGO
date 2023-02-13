

#ifndef RANDOM_H_INCLUDED
#define RANDOM_H_INCLUDED

#include "config.h"
#include <cstdint>
#include <limits>

/*
    Random number generator xoroshiro128+
*/
class Random {
public:
    Random() = delete;
    Random(std::uint64_t seed = 0);
    void seedrandom(std::uint64_t s);

    // Random numbers from [0, max - 1]
    template<int MAX>
    std::uint32_t randfix() {
        static_assert(0 < MAX &&
                     MAX < std::numeric_limits<std::uint32_t>::max(),
                     "randfix out of range");
        // Last bit isn't random, so don't use it in isolation. We specialize
        // this case.
        static_assert(MAX != 2, "don't isolate the LSB with xoroshiro128+");
        return gen() % MAX;
    }

    std::uint64_t randuint64();

    // Random number from [0, max - 1]
    std::uint64_t randuint64(const std::uint64_t max);

    // return the thread local RNG
    static Random& get_Rng(void);

    // UniformRandomBitGenerator interface
    using result_type = std::uint64_t;
    constexpr static result_type min() {
        return std::numeric_limits<result_type>::min();
    }
    constexpr static result_type max() {
        return std::numeric_limits<result_type>::max();
    }
    result_type operator()() {
        return gen();
    }

private:
    std::uint64_t gen(void);
    std::uint64_t m_s[2];
};

// Specialization for last bit: use sign test
template<>
inline std::uint32_t Random::randfix<2>() {
    return (gen() > (std::numeric_limits<std::uint64_t>::max() / 2));
}

#endif
