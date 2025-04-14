#ifndef BENCHMARKER_H_
#define BENCHMARKER_H_

#include <array>
#include <chrono>

class Benchmarker {
    using Clock = std::chrono::system_clock;
    static constexpr size_t SIZE = 1000;

public:
    Benchmarker();
    void start();
    void end();

    float getSec() const;
    float getFPS() const;
    float getRunningTime() const;
    size_t getCount() const;
    float getTimeSinceCreated() const;

private:
    std::array<float, SIZE> mTimes;
    float mSum = 0;
    size_t mCount = 0;
    Clock::time_point mPrev;
    Clock::time_point mCreated;
    float mRunningTime = 0;
};

#endif
