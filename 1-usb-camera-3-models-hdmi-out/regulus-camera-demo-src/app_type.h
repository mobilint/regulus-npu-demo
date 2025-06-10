#ifndef TYPE_H_
#define TYPE_H_

#include <maccel/type.h>

#include <opencv2/opencv.hpp>

#include "tsqueue.h"

namespace app {
struct InterThreadData {
    cv::Mat frame;
    cv::Mat pre;
    std::vector<std::vector<float>> npu_result;
};

using Buffer = ThreadSafeBuffer<InterThreadData>;
using Queue = ThreadSafeQueue<InterThreadData>;
}  // namespace app

#endif
