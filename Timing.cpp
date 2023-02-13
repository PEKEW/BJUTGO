

#include "Timing.h"

#include <chrono>


int Time::timediff_centis(Time start, Time end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>
        (end.m_time - start.m_time).count() / 10;
}

double Time::timediff_seconds(Time start, Time end) {
    return std::chrono::duration<double>(end.m_time - start.m_time).count();
}

Time::Time(void) {
    m_time = std::chrono::steady_clock::now();
}
