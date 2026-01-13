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
    CV_Assert(!frame.empty());
    CV_Assert(npu_w > 0 && npu_h > 0);

    const int w = frame.cols;
    const int h = frame.rows;

    const float scale = std::max((float)npu_w / (float)w, (float)npu_h / (float)h);

    const int new_w = (int)std::ceil(w * scale);
    const int new_h = (int)std::ceil(h * scale);

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int x = (new_w - npu_w) / 2;
    int y = (new_h - npu_h) / 2;

    x = std::clamp(x, 0, new_w - npu_w);
    y = std::clamp(y, 0, new_h - npu_h);

    return resized_frame(cv::Rect(x, y, npu_w, npu_h));
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
