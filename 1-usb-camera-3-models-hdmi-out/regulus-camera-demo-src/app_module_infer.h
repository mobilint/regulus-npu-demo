#ifndef APP_INFER_H_
#define APP_INFER_H_

#include <atomic>
#include <thread>

#include "app_type.h"
#include "tsqueue.h"

#include "maccel/model.h"

namespace app {
    std::thread module_infer(mobilint::Model* model, Buffer &buffer_in, Queue &queue_out, std::atomic<bool> &push_on, std::vector<mobilint::Buffer> &repositioned_buffer);
}

#endif
