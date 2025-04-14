#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

#include "app_type.h"
#include "tsqueue.h"

template <class T>
ThreadSafeQueue<T>::StatusCode ThreadSafeQueue<T>::push(const T& value) {
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

template <class T>
ThreadSafeQueue<T>::StatusCode ThreadSafeQueue<T>::push(T&& value) {
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

template <class T>
ThreadSafeQueue<T>::StatusCode ThreadSafeQueue<T>::pop(T& value) {
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

template <class T>
void ThreadSafeQueue<T>::close() {
    {
	std::lock_guard<std::mutex> lk(mMutex);
	mOn = false;
    }
    mCV.notify_all();
}

template <class T>
int ThreadSafeQueue<T>::size() { return mQueue.size(); }

template <class T>
ThreadSafeBuffer<T>::StatusCode ThreadSafeBuffer<T>::put(const T& value) {
    {
	std::lock_guard<std::mutex> lk(mMutex);
	mBuffer = value;
	mBufferIndex++;
    }
    mCV.notify_all();
    return OK;
}

template <class T>
ThreadSafeBuffer<T>::StatusCode ThreadSafeBuffer<T>::get(T& value, int64_t& index) {
    std::unique_lock<std::mutex> lk(mMutex);
    mCV.wait(lk, [this, index] { return mBufferIndex > index || !mOn; });
    if (!mOn) {
	return CLOSED;
    }
    value = mBuffer;
    index = mBufferIndex;

    return OK;
}

template <class T>
ThreadSafeBuffer<T>::StatusCode ThreadSafeBuffer<T>::get_nonblock(T& value, int64_t& index) {
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

template <class T>
void ThreadSafeBuffer<T>::open() {
    {
	std::lock_guard<std::mutex> lk(mMutex);
	mOn = true;
    }
}

template <class T>
void ThreadSafeBuffer<T>::close() {
    {
	std::lock_guard<std::mutex> lk(mMutex);
	mOn = false;
    }
    mCV.notify_all();
}

template class ThreadSafeQueue<app::InterThreadData>;
template class ThreadSafeBuffer<app::InterThreadData>;
