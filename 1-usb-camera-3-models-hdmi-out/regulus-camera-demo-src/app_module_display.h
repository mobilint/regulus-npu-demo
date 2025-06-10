#ifndef APP_DISPLAY_H_
#define APP_DISPLAY_H_

#include <atomic>
#include <opencv2/opencv.hpp>
#include <thread>

#include "app_type.h"

namespace app {
std::thread module_display_to_hdmi(Queue& queue_in, std::atomic<bool>& push_on);
}

#endif
