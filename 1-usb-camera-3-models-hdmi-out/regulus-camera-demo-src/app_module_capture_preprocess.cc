#include "app_module_capture_preprocess.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <opencv2/videoio.hpp>
#include <string>
#include <thread>

#include "app_type.h"
#include "benchmarker.h"
#include "maccel/model.h"

static cv::Mat resize_frame(cv::Mat frame, int npu_w, int npu_h) {
    int w = frame.cols;
    int h = frame.rows;

    // assert(npu_w > 0 && npu_h > 0 && w > 0 && h > 0);

    cv::Mat resized_frame;

    int w_resize;
    int h_resize;
    if ((float)npu_w / npu_h > (float)w / h) {
        float w_ratio = (float)npu_w / w;

        w_resize = w;
        h_resize = std::ceil(h * w_ratio);

        cv::resize(frame, resized_frame, cv::Size(w_resize, h_resize));
    } else {
        float h_ratio = (float)npu_h / h;

        w_resize = std::ceil(w * h_ratio);
        h_resize = h;

        cv::resize(frame, resized_frame, cv::Size(w_resize, h_resize));
    }

    cv::Mat frame_out = resized_frame({(int)((w_resize - npu_w) / 2),
                                       (int)((h_resize - npu_h) / 2), npu_w, npu_h})
                            .clone();

    return frame_out;
}

namespace app {
std::thread module_opencv_capture_and_preprocess(
    std::string dev_path, uint32_t camera_width, uint32_t camera_height,
    uint32_t camera_fps, int npu_w, int npu_h, mobilint::Model* model, Buffer& buffer_out,
    std::atomic<bool>& push_on, std::vector<mobilint::Buffer>& repositioned_buffer) {
    auto func = [=, &buffer_out, &repositioned_buffer, &push_on]() {
        cv::VideoCapture cap;
        cap.open(dev_path, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height);
        cap.set(cv::CAP_PROP_FPS, camera_fps);

        bool cap_open_failed = false;
        if (!cap.isOpened()) {
            std::cout << "Src Open Error: " << dev_path << std::endl;
            cap_open_failed = true;
        }

        Benchmarker bc_g;
        while (push_on) {
            cv::Mat frame;
            if (cap_open_failed) {
                frame = cv::Mat::zeros(cv::Size(npu_w, npu_h), CV_8UC3);
                usleep(10000);
            } else {
                cap >> frame;
                if (frame.empty()) {
                    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                    continue;
                }
            }

            cv::Mat frame_pre = resize_frame(frame, npu_w, npu_h);
            cv::Mat pre(npu_h, npu_w, CV_32FC3);

            for (int i = 0; i < npu_w * npu_h; i++) {
                // convert BGR to RGB
                ((float*)pre.data)[3 * i + 0] =
                    (float)(frame_pre.data[3 * i + 2]) / 255.0f;
                ((float*)pre.data)[3 * i + 1] =
                    (float)(frame_pre.data[3 * i + 1]) / 255.0f;
                ((float*)pre.data)[3 * i + 2] =
                    (float)(frame_pre.data[3 * i + 0]) / 255.0f;
            }

            model->repositionInputs({(float*)pre.data}, repositioned_buffer);
            buffer_out.put({frame_pre, pre});
        }
    };

    return std::thread(func);
}
}  // namespace app
