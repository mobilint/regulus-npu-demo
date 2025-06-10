#ifndef YOLOV8_POSTPROCESSOR
#define YOLOV8_POSTPROCESSOR

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

namespace mobilint::post {
enum class PostType { BASE, FACE, POSE, SEG };

struct YOLOv8PostItem {
    uint64_t id;
    cv::Mat& im;
    std::vector<std::vector<float>>& npu_outs;
    std::vector<std::array<float, 4>>& boxes;
    std::vector<float>& scores;
    std::vector<int>& labels;
    std::vector<std::vector<float>>& extras;
};

class YOLOv8PostProcessor {
public:
    YOLOv8PostProcessor();
    YOLOv8PostProcessor(int nc, int imh, int imw, float conf_thres, float iou_thres,
                        bool verbose);
    ~YOLOv8PostProcessor();

public:
    std::vector<int> generate_strides(int nl);
    std::vector<std::vector<int>> generate_grids(int imh, int imw,
                                                 const std::vector<int>& strides);
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs);

    int get_nl() const;
    int get_nc() const;
    PostType getType() const;
    float sigmoid(float num);
    float inverse_sigmoid(float num);
    std::vector<float> softmax(const std::vector<float>& vec);
    float area(float xmin, float ymin, float xmax, float ymax);
    float get_iou(const std::array<float, 4>& box1, const std::array<float, 4>& box2);
    void xywh2xyxy(std::vector<std::array<float, 4>>& pred_boxes);
    virtual void decode_extra(const std::vector<float>& extra,
                              const std::vector<int>& grid, int stride, int idx,
                              std::vector<float>& pred_extra);
    void decode_boxes(const std::vector<float>& npu_out, const std::vector<int>& grid,
                      int stride, int idx, std::array<float, 4>& pred_box);
    void decode_conf_thres(const std::vector<float>& npu_out_box,
                           const std::vector<float>& npu_out_cls,
                           const std::vector<int>& grid, int stride,
                           std::vector<std::array<float, 4>>& pred_boxes,
                           std::vector<float>& pred_conf, std::vector<int>& pred_label,
                           std::vector<std::pair<float, int>>& pred_scores,
                           const std::vector<float>& extra,
                           std::vector<std::vector<float>>& pred_extra);

    void compute_ratio(int srcHeight, int srcWidth, float& h_ratio, float& w_ratio);
    void compute_ratio_pads(int srcHeight, int srcWidth, float& ratio, int& xpad,
                            int& ypad);
    void rescaleImg(std::vector<std::array<int, 4>>& rescaled_boxes, int srcHeight,
                    int srcWidth);
    void rescaleImgWithPadding(std::vector<std::array<float, 4>>& rescaled_boxes,
                               int srcHeight, int srcWidth);
    void rescaleImgWithPaddingV2(std::vector<std::array<float, 4>>& rescaled_boxes,
                                 int srcHeight, int srcWidth);
    void nms(const std::vector<std::array<float, 4>>& pred_boxes,
             const std::vector<float>& pred_conf, const std::vector<int>& pred_label,
             std::vector<std::pair<float, int>>& scores,
             const std::vector<std::vector<float>>& pred_extra,
             std::vector<std::array<float, 4>>& final_boxes,
             std::vector<float>& final_scores, std::vector<int>& final_labels,
             std::vector<std::vector<float>>& final_extra);
    double set_timer();

    std::vector<std::array<float, 4>>& get_result_box();
    std::vector<float>& get_result_score();
    std::vector<int>& get_result_label();
    std::vector<std::vector<float>>& get_result_extra();
    void compute_ratio_pads(cv::Mat im, float& ratio, float& xpad, float& ypad);
    void plot_boxes(cv::Mat& im, std::vector<std::array<float, 4>>& boxes,
                    std::vector<float>& scores, std::vector<int>& labels);
    virtual void plot_extras(cv::Mat& im, std::vector<std::vector<float>>& extras);

    void print(std::string msg);
    virtual void worker();
    uint64_t enqueue(cv::Mat& im, std::vector<std::vector<float>>& npu_outs,
                     std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
                     std::vector<int>& labels, std::vector<std::vector<float>>& extras);
    void receive(uint64_t receipt_no);

protected:
    const int mOpenmpThreadCount = 2;
    int m_nextra;        // number of keypoints/landmarks/masks
    int m_nl;            // number of detection layers
    int m_nc;            // number of classes
    uint32_t m_imh;      // model input image height
    uint32_t m_imw;      // model input image width
    float m_conf_thres;  // confidence threshold, used in decoding
    float m_inverse_conf_thres;
    float m_iou_thres;  // iou threshold, used in nms
    int m_max_det_num;
    bool m_verbose;
    PostType mType;

    std::vector<std::array<float, 4>> final_boxes;
    std::vector<float> final_scores;
    std::vector<int> final_labels;
    std::vector<std::vector<float>> final_extra;  // keypoints/landmarks/masks

    std::vector<int> m_strides;
    std::vector<std::vector<int>> m_grids;

    std::thread mThread;
    std::queue<YOLOv8PostItem> mQueueIn;
    std::vector<uint64_t> mOut;
    uint64_t ticket;

    std::mutex mPrintMutex;
    std::mutex mMutexIn;
    std::mutex mMutexOut;
    std::condition_variable mCondIn;
    std::condition_variable mCondOut;
    bool destroyed;

    // not the best practice, should find better way
    const std::vector<std::string> COCO_LABELS = {
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

    const std::vector<std::array<int, 3>> COLORS = {
        {56, 56, 255},  {151, 157, 255}, {31, 112, 255}, {29, 178, 255},  {49, 210, 207},
        {10, 249, 72},  {23, 204, 146},  {134, 219, 61}, {52, 147, 26},   {187, 212, 0},
        {168, 153, 44}, {255, 194, 0},   {147, 69, 52},  {255, 115, 100}, {236, 24, 0},
        {255, 56, 132}, {133, 0, 82},    {255, 56, 203}, {200, 149, 255}, {199, 55, 255}};
};
}  // namespace mobilint::post

#endif
