#include "app_module_display.h"

#include <atomic>
#include <opencv2/opencv.hpp>
#include <thread>

#include "app_type.h"
#include "benchmarker.h"
#include "gst_appsrc_kmssink.h"
#include "gst_base_pipeline.h"

namespace app {
std::thread module_display_to_hdmi(Queue& queue_in, std::atomic<bool>& push_on) {
    auto func([&]() {
        gst_wrapper::AppsrcKmssinkPipeline gst_pipeline_display("BGR", 1920, 1080, 30);
        gst_pipeline_display.start();

        Benchmarker bc;
        int prev_sec = 0;
        while (push_on) {
            bc.end();
            bc.start();

            int curr_sec = bc.getRunningTime();
            if (curr_sec - prev_sec > 1) {
                std::cout << "Average FPS: " << bc.getFPS() << std::endl;
                prev_sec = curr_sec;
            }

            InterThreadData data_in;
            auto qsc = queue_in.pop(data_in);
            if (qsc == Queue::StatusCode::CLOSED) {
                std::cout << "queue closed.\n";
                exit(1);
            }

            gst_wrapper::push_data_to_appsrc(gst_pipeline_display.appsrc,
                                             data_in.frame.data, 1920 * 1080 * 3);
        }

        gst_pipeline_display.stop();
    });

    return std::thread(func);
}
}  // namespace app
