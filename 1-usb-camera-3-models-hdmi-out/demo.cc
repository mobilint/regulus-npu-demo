#include <csignal>
#include <iostream>
#include <atomic>
#include <opencv2/imgcodecs.hpp>
#include <thread>

#include <opencv2/opencv.hpp>
#include <gst/gst.h>

#include "app_parse_argv.h"
#include "app_init_models.h"
#include "app_type.h"
#include "app_module_capture_preprocess.h"
#include "app_module_infer.h"
#include "app_module_post_process.h"
#include "app_module_display.h"
#include "app_module_construct_image.h"

std::atomic<bool> push_on(true);

void sigintHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    push_on.store(false);
    exit(-1);
}

int main(int argc, char* argv[]) {
    signal(SIGINT, sigintHandler);
    gst_init(NULL, NULL);

    auto [dev_paths, camera_width, camera_height, camera_fps] = parse_dvec1_w_h_f(argc, argv);

    std::string mxq_dir = "./mxq/";
    auto [acc, mc, models] =
	init_3_models(mxq_dir + "yolov8n-face_h384_w576_reg.mxq",
		      mxq_dir + "yolov8s-det_h384_w576_reg.mxq",
		      mxq_dir + "yolov8s-pose_h384_w576_reg.mxq");

    int npu_w = 576, npu_h = 384;
    std::vector<mobilint::Buffer> repositioned_buffer = models[0]->acquireInputBuffer();

    std::array<app::Buffer, 1> buffers;
    std::array<app::Queue, 10> queues;
    std::vector<std::thread> threads;;

    threads.push_back(app::module_opencv_capture_and_preprocess(dev_paths[0], camera_width, camera_height, camera_fps, npu_w, npu_h, models[0].get(), buffers[0], push_on, repositioned_buffer));

    // face
    threads.push_back(app::module_infer(models[0].get(), buffers[0], queues[0], push_on, repositioned_buffer));
    
    threads.push_back(app::module_post_process_face(npu_h, npu_w, queues[0], queues[1], push_on));

    // od
    threads.push_back(app::module_infer(models[1].get(), buffers[0], queues[2], push_on, repositioned_buffer));
    
    threads.push_back(app::module_post_process_od(npu_h, npu_w, queues[2], queues[3], push_on));

    // pose
    threads.push_back(app::module_infer(models[2].get(), buffers[0], queues[4], push_on, repositioned_buffer));
    
    threads.push_back(app::module_post_process_pose(npu_h, npu_w, queues[4], queues[5], push_on));

    // construct image
    threads.push_back(app::module_construct_image(buffers[0], queues[1], queues[3], queues[5], queues[9], push_on));

    // display
    threads.push_back(app::module_display_to_hdmi(queues[9], push_on));
    
    for (auto& thread : threads) { thread.join(); }
    for (auto& queue : queues) { queue.close(); }
    for (auto& buffer : buffers) { buffer.close(); }
    
    gst_deinit();
    
    return 0;
}
