#include <unistd.h>

#include <atomic>
#include <csignal>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <cmath>

#include "appsrc_kmssink.h"
#include "gst/gst.h"
#include "gst/gstelement.h"
#include "maccel/maccel.h"
#include "maccel/type.h"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "post_yolov8_face.h"
#include "tsqueue.h"

#define REGULUS 1

#define DISPLAY_WIDTH 1200
#define DISPLAY_HEIGHT 1920

using namespace std;

using mobilint::Accelerator;
using mobilint::BufferInfo;
using mobilint::Cluster;
using mobilint::Core;
using mobilint::Model;
using mobilint::ModelConfig;
using mobilint::StatusCode;

struct InferItem
{
    cv::Mat frame;
    std::vector<mobilint::Buffer> buffer;
};

struct PostItem
{
    cv::Mat frame;
    std::vector<mobilint::NDArray<float>> result;
};

std::atomic<bool> push_on(true);

void sigintHandler(int signum)
{
    cout << "Interrupt signal (" << signum << ") received.\n";
    push_on.store(false);
    exit(-1);
}

using PreQueue = ThreadSafeQueue<cv::Mat>;
using InferQueue = ThreadSafeQueue<InferItem>;
using PostQueue = ThreadSafeQueue<PostItem>;
using DisplayQueue = ThreadSafeQueue<cv::Mat>;

namespace
{
    void work_feed(std::string src_path, PreQueue *pre_queue)
    {
        cv::VideoCapture cap;
        
        cap.open(src_path);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 960);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
        if (!cap.isOpened())
        {
            std::cout << "Source Open Error! Checkout argv[1]: " << src_path << std::endl;
            push_on = false;
        }

        while (push_on)
        {
            cv::Mat frame;
            cap >> frame;

            pre_queue->push(frame);
        }

        pre_queue->close();
    }

    void work_pre(Model *model, PreQueue *pre_queue, InferQueue *infer_queue)
    {
        int crop_w = 512;
        int crop_h = 640;
        int i = 0;

        std::vector<mobilint::Buffer> repositioned_buffer = model->acquireInputBuffer();

        while (push_on)
        {
            cv::Mat frame;
            auto qsc = pre_queue->pop(frame);
            if (qsc == PreQueue::StatusCode::CLOSED)
                break;

            cv::Mat resized_frame;
            int resize_w = frame.cols * 640 / frame.rows;
            int resize_h = 640;
            cv::resize(frame, resized_frame, cv::Size(resize_w, resize_h));

            cv::Mat cropped_frame;
            cropped_frame = resized_frame({(int)((resize_w - crop_w) / 2),
                                           (int)((resize_h - crop_h) / 2), crop_w, crop_h})
                                .clone();

            cv::Mat pre(crop_h, crop_w, CV_32FC3);
            for (int i = 0; i < crop_w * crop_h * 3; i++)
            {
                ((float *)pre.data)[i] = (float)(cropped_frame.data[i]) / 255.0f;
            }

            model->repositionInputs({(float *)pre.data}, repositioned_buffer);

            infer_queue->push({cropped_frame, repositioned_buffer});
        }

        pre_queue->close();
        infer_queue->close();
        model->releaseBuffer(repositioned_buffer);
    }

    void work_infer(Model *model, InferQueue *infer_queue, PostQueue *post_queue)
    {
        StatusCode sc;
        std::vector<mobilint::NDArray<float>> result;
        
        while (push_on)
        {
            InferItem item;
            auto qsc = infer_queue->pop(item);
            if (qsc == InferQueue::StatusCode::CLOSED)
                break;

            sc = model->inferBufferToFloat(item.buffer, result);
            if (!sc)
                exit(1);

            post_queue->push({item.frame, result});
        }

        infer_queue->close();
        post_queue->close();
    }

    void work_post(PostQueue *post_queue, DisplayQueue *display_queue)
    {
        float face_conf_thres = 0.15;
        float face_iou_thres = 0.5;
        int face_nc = 1; // number of classes
        int face_imh = 640;
        int face_imw = 512;

        mobilint::postface::YOLOv8FacePostProcessor post(face_nc, face_imh, face_imw, face_conf_thres, face_iou_thres, 1, false);

        while (push_on)
        {
            PostItem item;
            auto qsc = post_queue->pop(item);
            if (qsc == PostQueue::StatusCode::CLOSED)
                break;

            std::vector<std::array<float, 4>> boxes;
            std::vector<float> scores;
            std::vector<int> labels;

            uint64_t ticket = post.enqueue(item.frame, item.result, boxes, scores, labels, item.result);
            post.receive(ticket);

            display_queue->push(item.frame);
        }

        post_queue->close();
        display_queue->close();
    }

    void printUsage()
    {
        cout << "Usage: demo {Dir path to camera device}, default is \"/dev/video3\n";
    }

} // namespace

int main(int argc, char *argv[])
{
    printUsage();

    signal(SIGINT, sigintHandler);

    std::string src_path = argc > 1 ? argv[1] : "/dev/video3";

    StatusCode sc;
    std::unique_ptr<Accelerator> acc = Accelerator::create(sc);
    if (!sc)
        exit(1);

    ModelConfig mc;
    mc.excludeAllCores();
    mc.include(Core::Core0);
    std::unique_ptr<Model> model = Model::create("./face_yolov8n_640_512.mxq", mc, sc);

    if (!sc)
        exit(1);

    sc = model->launch(*acc);

    PreQueue pre_queue(1);
    InferQueue infer_queue(1);
    PostQueue post_queue(1);
    DisplayQueue display_queue(1);

    gst_init(NULL, NULL);

    gst_wrapper::AppsrcKmssinkPipeline gst_pipeline_display("BGR", DISPLAY_WIDTH, DISPLAY_HEIGHT, 30);

    gst_pipeline_display.start();

    std::thread thread_feed(work_feed, src_path, &pre_queue);
    std::thread thread_pre(work_pre, model.get(), &pre_queue, &infer_queue);
    std::thread thread_infer(work_infer, model.get(), &infer_queue, &post_queue);
    std::thread thread_post(work_post, &post_queue, &display_queue);

    cv::Mat display = cv::Mat::zeros(cv::Size(DISPLAY_WIDTH, DISPLAY_HEIGHT), CV_8UC3);

    while (push_on)
    {
        cv::Mat frame;
        display_queue.pop(frame);
        frame.copyTo(display({0, 0, 512, 640}));

        gst_wrapper::push_data_to_appsrc(gst_pipeline_display.appsrc, display.data, DISPLAY_WIDTH * DISPLAY_HEIGHT * 3);
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