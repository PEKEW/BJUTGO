

#ifndef TIMING_H_INCLUDED
#define TIMING_H_INCLUDED

#include <chrono>

class Time {
public:
    /* sets to current time */
    Time(void);

    /* time difference in centiseconds */
    static int timediff_centis(Time start, Time end);

    /* time difference in seconds */
    static double timediff_seconds(Time start, Time end);

private:
    std::chrono::steady_clock::time_point m_time;
};

#endif
