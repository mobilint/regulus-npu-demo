#ifndef APP_POST_PROCESS_H_
#define APP_POST_PROCESS_H_

#include <atomic>
#include <thread>

#include "app_type.h"

namespace app {
std::thread module_post_process_face(int npu_h, int npu_w, Queue& queue_in,
                                     Queue& queue_out, std::atomic<bool>& push_on);

std::thread module_post_process_od(int npu_h, int npu_w, Queue& queue_in,
                                   Queue& queue_out, std::atomic<bool>& push_on);

std::thread module_post_process_pose(int npu_h, int npu_w, Queue& queue_in,
                                     Queue& queue_out, std::atomic<bool>& push_on);
}  // namespace app

#endif
