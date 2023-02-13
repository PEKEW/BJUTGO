

#ifndef UTILS_H_DEFINED
#define UTILS_H_DEFINED

#include "config.h"

#include <atomic>
#include <limits>
#include <string>

#include "ThreadPool.h"

extern Utils::ThreadPool thread_pool;

namespace Utils {
    void myprintf(const char *fmt, ...);
    void gtp_printf(int id, const char *fmt, ...);
    void gtp_fail_printf(int id, const char *fmt, ...);
    void log_input(const std::string& input);
    bool input_pending();

    template<class T>
    void atomic_add(std::atomic<T> &f, T d) {
        T old = f.load();
        while (!f.compare_exchange_weak(old, old + d));
    }

    template<typename T>
    T rotl(const T x, const int k) {
        return (x << k) | (x >> (std::numeric_limits<T>::digits - k));
    }

    inline bool is7bit(int c) {
        return c >= 0 && c <= 127;
    }

    size_t ceilMultiple(size_t a, size_t b);
}

#endif
