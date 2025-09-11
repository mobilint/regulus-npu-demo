#pragma once
#include <math.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "types.h"

namespace mobilint::post {

enum class PostType { BASE, FACE, POSE, SEG };

struct PostItem {
    uint64_t id;
    std::vector<std::vector<float>>& npu_outs;
    std::vector<std::array<float, 4>>& boxes;
    std::vector<float>& scores;
    std::vector<int>& labels;
    std::vector<std::vector<float>>& extras;
};
class PostProcessor {
public:
    PostProcessor() {}
    PostProcessor(int imh, int imw, const ModelInfo& cfg);
    PostProcessor(int nc, int imh, int imw);
    PostProcessor(int nc, int imh, int imw, float conf_thres, float iou_thres, int nl,
                  int max_num_threads = 4);
    virtual ~PostProcessor() noexcept;

public:
    virtual void set_params(int nc, int imh, int imw, float conf_thres = 0.3,
                            float iou_thres = 0.6, int nl = 3, int max_num_threads = 4);
    void set_nl(int nl) { m_nl = nl; };
    virtual void run_postprocess(const std::vector<std::vector<float>>& npu_outs){};

    int get_nl() const;
    int get_nc() const;
    int get_imh() const;
    int get_imw() const;
    PostType getType() const;
    virtual void nms(const std::vector<std::array<float, 4>>& pred_boxes,
                     std::vector<std::pair<float, int>>& pred_scores,
                     const std::vector<int>& pred_labels,
                     const std::vector<std::vector<float>>& pred_extra,
                     std::vector<std::array<float, 4>>& final_boxes,
                     std::vector<float>& final_scores, std::vector<int>& final_labels,
                     std::vector<std::vector<float>>& final_extra);
    virtual void nms(const std::vector<std::array<float, 4>>& pred_boxes,
                     std::vector<std::pair<float, int>>& pred_scores,
                     const std::vector<int>& pred_labels,
                     std::vector<std::array<float, 4>>& final_boxes,
                     std::vector<float>& final_scores, std::vector<int>& final_labels);
    double set_timer();

    std::vector<std::array<float, 4>>& get_result_box();
    std::vector<float>& get_result_score();
    std::vector<int>& get_result_label();
    std::vector<std::vector<float>>& get_result_extra();

    virtual void rescale_boxes(int org_h, int org_w,
                               std::vector<std::array<float, 4>>& boxes,
                               std::vector<int>& labels);
    virtual void plot_results(cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
                              const std::vector<float>& scores,
                              const std::vector<int>& labels,
                              const std::vector<std::vector<float>>& extras = {});
    virtual void plot_results(cv::Mat& im, const std::vector<float>& scores,
                              const std::vector<int>& labels){};
    virtual void plot_boxes(cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
                            const std::vector<float>& scores,
                            const std::vector<int>& labels);
    virtual void plot_extras(cv::Mat& im, const std::vector<std::vector<float>>& extras);

    void print(std::string msg);
    virtual void worker() = 0;
    virtual uint64_t enqueue(std::vector<std::vector<float>>& npu_outs,
                             std::vector<std::array<float, 4>>& boxes,
                             std::vector<float>& scores, std::vector<int>& labels,
                             std::vector<std::vector<float>>& extras);
    void receive(uint64_t receipt_no);

protected:
    int m_max_num_threads;
    int m_nextra;                // number of keypoints/landmarks/masks
    int m_nl;                    // number of detection layers
    int m_nc;                    // number of classes
    uint32_t m_imh;              // model input image height
    uint32_t m_imw;              // model input image width
    float m_conf_thres;          // confidence threshold, used in decoding
    float m_inverse_conf_thres;  // inverse confidence threshold, used in decoding
    float m_iou_thres;           // iou threshold, used in nms
    int m_max_det_num;
    PostType mType;

    std::vector<std::array<float, 4>> final_boxes;
    std::vector<float> final_scores;
    std::vector<int> final_labels;
    std::vector<std::vector<float>> final_extra;  // keypoints/landmarks/masks

    std::thread mThread;
    std::queue<PostItem> mQueueIn;
    std::vector<uint64_t> mOut;
    uint64_t ticket;

    std::mutex mPrintMutex;
    std::mutex mMutexIn;
    std::mutex mMutexOut;
    std::condition_variable mCondIn;
    std::condition_variable mCondOut;
    bool destroyed;

    const std::vector<std::string> DET_LABELS = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "baseball bat", "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",
        "bottle",        "wine glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush",
    };

    const std::vector<int> YOLO_TO_COCO = {
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

    const std::vector<std::array<int, 3>> COLORS = {
        {56, 56, 255},  {151, 157, 255}, {31, 112, 255}, {29, 178, 255},  {49, 210, 207},
        {10, 249, 72},  {23, 204, 146},  {134, 219, 61}, {52, 147, 26},   {187, 212, 0},
        {168, 153, 44}, {255, 194, 0},   {147, 69, 52},  {255, 115, 100}, {236, 24, 0},
        {255, 56, 132}, {133, 0, 82},    {255, 56, 203}, {200, 149, 255}, {199, 55, 255}};
};
};  // namespace mobilint::post
