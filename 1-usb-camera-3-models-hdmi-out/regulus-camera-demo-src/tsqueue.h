#ifndef TSQUEUE_H_
#define TSQUEUE_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(int max_size = 1) : mMaxSize(max_size) {}

    enum StatusCode {
        OK = 0,
        CLOSED = 1,
    };

    StatusCode push(const T& value);
    StatusCode push(T&& value);
    StatusCode pop(T& value);

    void close();
    int size();

private:
    // mMutex protects mQueue and mOn.
    std::mutex mMutex;
    std::condition_variable mCV;
    std::queue<T> mQueue;
    bool mOn = true;

    const int mMaxSize;
};

template <typename T>
class ThreadSafeBuffer {
public:
    enum StatusCode { OK = 0, CLOSED = 1 };

    StatusCode put(const T& value);
    StatusCode get(T& value, int64_t& index);
    StatusCode get_nonblock(T& value, int64_t& index);

    void open();
    void close();

private:
    mutable std::mutex mMutex;
    std::condition_variable mCV;
    T mBuffer;
    int64_t mBufferIndex = 0;
    bool mOn = true;
};
#endif
