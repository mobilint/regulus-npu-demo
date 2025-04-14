#ifndef APP_CAPTURE_PREPROCESS_H_
#define APP_CAPTURE_PREPROCESS_H_

#include <cstdint>
#include <opencv2/videoio.hpp>
#include <string>
#include <atomic>
#include <cassert>
#include <thread>

#include <maccel/model.h>
#include "app_type.h"

namespace app {
    std::thread module_opencv_capture_and_preprocess(std::string dev_path, uint32_t camera_width, uint32_t camera_height, uint32_t camera_fps, int npu_w, int npu_h, mobilint::Model* model, Buffer &buffer_out, std::atomic<bool> &push_on, std::vector<mobilint::Buffer> &repositioned_buffer);
}

#endif
