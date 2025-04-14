#ifndef TSQUEUE_H_
#define TSQUEUE_H_

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue(int max_size = -1) : mMaxSize(max_size) {}
    enum StatusCode {
        OK = 0,
        CLOSED = 1,
    };

    StatusCode push(const T& value) {
        {
            std::unique_lock<std::mutex> lk(mMutex);
            mCV.wait(lk, [this] {
                return !(mMaxSize > 0 && mMaxSize <= mQueue.size() && mOn);
            });
            if (!mOn) {
                return CLOSED;
            }
            mQueue.push(value);
        }
        mCV.notify_one();
        return OK;
    }

    StatusCode push(T&& value) {
        {
            std::unique_lock<std::mutex> lk(mMutex);
            mCV.wait(lk, [this] {
                return !(mMaxSize > 0 && mMaxSize <= mQueue.size() && mOn);
            });
            if (!mOn) {
                return CLOSED;
            }
            mQueue.push(std::move(value));
        }
        mCV.notify_one();
        return OK;
    }

    StatusCode pop(T& value) {
        {
            std::unique_lock<std::mutex> lk(mMutex);
            mCV.wait(lk, [this] { return !(mQueue.empty() && mOn); });
            if (!mOn) {
                return CLOSED;
            }
            value = std::move(mQueue.front());
            mQueue.pop();
        }
        mCV.notify_one();
        return OK;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mOn = false;
        }
        mCV.notify_all();
    }

    int size() { return mQueue.size(); }

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

    StatusCode put(const T& value) {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mBuffer = value;
            mBufferIndex++;
        }
        mCV.notify_all();
        return OK;
    }

    StatusCode get(T& value, int64_t& index) {
        std::unique_lock<std::mutex> lk(mMutex);
        mCV.wait(lk, [this, index] { return mBufferIndex > index || !mOn; });
        if (!mOn) {
            return CLOSED;
        }
        value = mBuffer;
        index = mBufferIndex;

        return OK;
    }

    StatusCode get_nonblock(T& value, int64_t& index) {
        std::unique_lock<std::mutex> lk(mMutex);
        if (mBufferIndex <= index) {
            return OK;
        }
        if (!mOn) {
            return CLOSED;
        }
        value = mBuffer;
        index = mBufferIndex;

        return OK;
    }

    void open() {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mOn = true;
        }
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mMutex);
            mOn = false;
        }
        mCV.notify_all();
    }

private:
    mutable std::mutex mMutex;
    std::condition_variable mCV;
    T mBuffer;
    int64_t mBufferIndex = 0;
    bool mOn = true;
};
#endif
