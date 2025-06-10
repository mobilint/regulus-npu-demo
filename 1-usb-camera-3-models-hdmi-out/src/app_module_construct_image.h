#ifndef APP_CONSTRUCT_IMAGE_H
#define APP_CONSTRUCT_IMAGE_H

#include <atomic>
#include <thread>

#include "app_type.h"

namespace app {
std::thread module_construct_image(Buffer& buffer_in_org, Queue& queue_in_face,
                                   Queue& queue_in_od, Queue& queue_in_pose,
                                   Queue& queue_out, std::atomic<bool>& push_on);
}

#endif
