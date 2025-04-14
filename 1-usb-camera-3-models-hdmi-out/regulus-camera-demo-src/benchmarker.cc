#include <array>
#include <chrono>

#include "benchmarker.h"

Benchmarker::Benchmarker() {
    mCreated = Clock::now();
    start();
}

void Benchmarker::start() { mPrev = Clock::now(); }

void Benchmarker::end() {
    std::chrono::duration<float> dt = Clock::now() - mPrev;

    if (mCount >= SIZE) {
	mSum -= mTimes[mCount % SIZE];
    }

    float t = dt.count();
    mTimes[mCount++ % SIZE] = t;
    mSum += t;
    mRunningTime += t;
}

float Benchmarker::getSec() const {
    if (mCount == 0) {
	return 0;
    }

    return mSum / (SIZE < mCount ? SIZE : mCount);
}

float Benchmarker::getFPS() const {
    float avg_time = getSec();

    if (avg_time == 0) {
	return 0;
    }
    return 1 / avg_time;
}

float Benchmarker::getRunningTime() const { return mRunningTime; }

size_t Benchmarker::getCount() const { return mCount; }

float Benchmarker::getTimeSinceCreated() const {
    std::chrono::duration<float> dt = Clock::now() - mCreated;
    return dt.count();
}
