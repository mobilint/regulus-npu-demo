#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <csignal>
#include <functional>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include "appsrc_kmssink.h"
#include "gst/gst.h"
#include "gst/gstelement.h"
#include "maccel/maccel.h"
#include "maccel/type.h"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "post_yolov8_face.h"
#include "tsqueue.h"
#include "utils.h"

#define REGULUS 1

using namespace std;

using mobilint::Accelerator;
using mobilint::BufferInfo;
using mobilint::Cluster;
using mobilint::Core;
using mobilint::Model;
using mobilint::ModelConfig;
using mobilint::StatusCode;

struct InferItem {
    cv::Mat frame;
    std::vector<mobilint::Buffer> buffer;
};

struct PostItem {
    cv::Mat frame;
    std::vector<mobilint::NDArray<float>> result;
};

std::atomic<bool> push_on(true);

void sigintHandler(int signum) {
    cout << "Interrupt signal (" << signum << ") received.\n";
    push_on.store(false);
    exit(-1);
}

using PreQueue = ThreadSafeQueue<cv::Mat>;
using InferQueue = ThreadSafeQueue<InferItem>;
using PostQueue = ThreadSafeQueue<PostItem>;
using DisplayQueue = ThreadSafeQueue<cv::Mat>;

namespace {

void work_feed(const YamlConfig& cfg, PreQueue* pre_queue) {
    cv::VideoCapture cap;
    bool opened = false;

    auto is_index = [](const std::string& device) {
        return !device.empty() &&
               std::all_of(device.begin(), device.end(),
                           [](unsigned char c) { return std::isdigit(c); });
    };

    if (cfg.camera_device.empty()) {
        opened = cap.open(0, cv::CAP_V4L2);
    } else if (is_index(cfg.camera_device)) {
        opened = cap.open(std::stoi(cfg.camera_device), cv::CAP_V4L2);
    } else {
        opened = cap.open(cfg.camera_device, cv::CAP_V4L2);
    }

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    if (cfg.camera_width > 0 && cfg.camera_height > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, cfg.camera_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.camera_height);
    }
    if (cfg.camera_fps > 0) {
        cap.set(cv::CAP_PROP_FPS, cfg.camera_fps);
    }
    if (!opened || !cap.isOpened()) {
        std::cout << "Source Open Error! Checkout camera device: " << cfg.camera_device
                  << std::endl;
        push_on = false;
        pre_queue->close();
        return;
    }

    while (push_on) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            usleep(1000);
            continue;
        }

        pre_queue->push(frame);
    }

    pre_queue->close();
}

void work_pre(Model* model, PreQueue* pre_queue, InferQueue* infer_queue) {
    const int target_w = 512;
    const int target_h = 640;

    std::vector<mobilint::Buffer> repositioned_buffer = model->acquireInputBuffer();

    while (push_on) {
        cv::Mat frame;
        auto qsc = pre_queue->pop(frame);
        if (qsc == PreQueue::StatusCode::CLOSED) {
            break;
        }
        if (frame.empty()) {
            continue;
        }

        cv::Mat resized_frame;

        cv::resize(frame, resized_frame, cv::Size(target_w, target_h), 0, 0,
                   cv::INTER_LINEAR);

        cv::Mat pre;
        resized_frame.convertTo(pre, CV_32FC3, 1.f / 255.f);

        model->repositionInputs({pre.ptr<float>()}, repositioned_buffer);

        infer_queue->push({resized_frame, repositioned_buffer});
    }

    pre_queue->close();
    infer_queue->close();
    model->releaseBuffer(repositioned_buffer);
}

void work_infer(Model* model, InferQueue* infer_queue, PostQueue* post_queue) {
    StatusCode sc;
    std::vector<mobilint::NDArray<float>> result;

    while (push_on) {
        InferItem item;
        auto qsc = infer_queue->pop(item);
        if (qsc == InferQueue::StatusCode::CLOSED) {
            break;
        }
        if (item.frame.empty()) {
            continue;
        }

        sc = model->inferBufferToFloat(item.buffer, result);
        if (!sc) {
            push_on = false;
            break;
        }

        post_queue->push({item.frame, result});
    }

    infer_queue->close();
    post_queue->close();
}

void work_post(PostQueue* post_queue, DisplayQueue* display_queue) {
    float face_conf_thres = 0.15;
    float face_iou_thres = 0.5;
    int face_nc = 1;  // number of classes
    int face_imw = 512;
    int face_imh = 640;

    mobilint::postface::YOLOv8FacePostProcessor post(
        face_nc, face_imh, face_imw, face_conf_thres, face_iou_thres, 1, false);

    while (push_on) {
        PostItem item;
        auto qsc = post_queue->pop(item);
        if (qsc == PostQueue::StatusCode::CLOSED) {
            break;
        }

        std::vector<std::array<float, 4>> boxes;
        std::vector<float> scores;
        std::vector<int> labels;

        uint64_t ticket =
            post.enqueue(item.frame, item.result, boxes, scores, labels, item.result);
        post.receive(ticket);

        display_queue->push(item.frame);
    }

    post_queue->close();
    display_queue->close();
}

void printUsage() { cout << "Usage: demo [config.yaml]\n"; }

}  // namespace

int main(int argc, char* argv[]) {
    printUsage();

    signal(SIGINT, sigintHandler);

    YamlConfig cfg;
    std::string config_path = "./config.yaml";
    if (argc > 1) config_path = argv[1];
    if (!loadYamlConfig(config_path, cfg)) {
        std::cerr << "[WARN] Failed to load config file: " << config_path
                  << ", using defaults.\n";
    }

    if (cfg.camera_device == "") {
        cfg.camera_device = findVideoDevice();
    }

    StatusCode sc;
    std::unique_ptr<Accelerator> acc = Accelerator::create(sc);
    if (!sc) exit(1);

    ModelConfig mc;
    mc.excludeAllCores();
    mc.include(Core::Core0);
    std::unique_ptr<Model> model = Model::create("./face_yolov8n_640_512.mxq", mc, sc);

    if (!sc) exit(1);

    sc = model->launch(*acc);

    PreQueue pre_queue(1);
    InferQueue infer_queue(1);
    PostQueue post_queue(1);
    DisplayQueue display_queue(1);

    gst_init(NULL, NULL);

    gst_wrapper::AppsrcKmssinkPipeline gst_pipeline_display(
        "BGR", cfg.display_width, cfg.display_height,
        cfg.camera_fps > 0 ? cfg.camera_fps : 30);

    gst_pipeline_display.start();

    std::thread thread_feed(work_feed, std::cref(cfg), &pre_queue);
    std::thread thread_pre(work_pre, model.get(), &pre_queue, &infer_queue);
    std::thread thread_infer(work_infer, model.get(), &infer_queue, &post_queue);
    std::thread thread_post(work_post, &post_queue, &display_queue);

    cv::Mat display =
        cv::Mat::zeros(cv::Size(cfg.display_width, cfg.display_height), CV_8UC3);

    while (push_on) {
        cv::Mat frame;
        auto qsc = display_queue.pop(frame);
        if (qsc == DisplayQueue::StatusCode::CLOSED) {
            break;
        }

        if (frame.empty()) {
            continue;
        }

        cv::Mat frame_resized;
        cv::resize(frame, frame_resized, display.size(), 0, 0, cv::INTER_LINEAR);
        frame_resized.copyTo(display);

        gst_wrapper::push_data_to_appsrc(gst_pipeline_display.appsrc, display.data,
                                         cfg.display_width * cfg.display_height * 3);
    }

    pre_queue.close();
    infer_queue.close();
    post_queue.close();
    display_queue.close();

    thread_feed.join();
    thread_pre.join();
    thread_infer.join();
    thread_post.join();

    gst_pipeline_display.stop();

    gst_deinit();

    std::cout << "end\n";
    return 0;
}
