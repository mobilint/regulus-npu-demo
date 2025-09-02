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

#include "./maccel/maccel.h"
#include "postprocess.h"

namespace mobilint::post {

class SSDPostProcessor : public PostProcessor {
public:
    SSDPostProcessor();
    SSDPostProcessor(int nc, int imh, int imw);
    SSDPostProcessor(int nc, int imh, int imw, float conf_thres, float iou_thres,
                     int max_num_threads = 4);
    ~SSDPostProcessor();

public:
    void set_params(int nc, int imh, int imw, float conf_thres = 0.3,
                    float iou_thres = 0.6, int max_num_threads = 4) override;
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs);

    void prior_generation(const std::vector<std::pair<int, int>>& feat_sizes,
                          const std::vector<std::vector<float>>& aspect_ratios,
                          int num_boxes, float min_scale = 0.2, float max_scale = 0.95);
    void decode_conf_thres(const std::vector<float>& boxes_out,
                           const std::vector<float>& cls_out, int idx, int prior_offset,
                           std::vector<std::array<float, 4>>& pred_boxes,
                           std::vector<std::pair<float, int>>& pred_scores,
                           std::vector<int>& pred_labels);
    void rescale_boxes(int org_h, int org_w, std::vector<std::array<float, 4>>& boxes,
                       std::vector<int>& labels) override;
    void plot_boxes(cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
                    const std::vector<float>& scores,
                    const std::vector<int>& labels) override;
    void worker() override;

protected:
    std::vector<float> m_coder_weights;
    std::vector<float> m_priors;
    std::vector<std::pair<int, int>> m_feat_sizes;
    int m_reg_max = 16;

    const std::vector<std::string> COCO_LABELS = {
        "background",    "person",      "bicycle",      "car",          "motorbike",
        "aeroplane",     "bus",         "train",        "truck",        "boat",
        "trafficlight",  "firehydrant", "streetsign",   "stopsign",     "parkingmeter",
        "bench",         "bird",        "cat",          "dog",          "horse",
        "sheep",         "cow",         "elephant",     "bear",         "zebra",
        "giraffe",       "hat",         "backpack",     "umbrella",     "shoe",
        "eyeglasses",    "handbag",     "tie",          "suitcase",     "frisbee",
        "skis",          "snowboard",   "sportsball",   "kite",         "baseballbat",
        "baseballglove", "skateboard",  "surfboard",    "tennisracket", "bottle",
        "plate",         "wineglass",   "cup",          "fork",         "knife",
        "spoon",         "bowl",        "banana",       "apple",        "sandwich",
        "orange",        "broccoli",    "carrot",       "hotdog",       "pizza",
        "donut",         "cake",        "chair",        "sofa",         "pottedplant",
        "bed",           "mirror",      "diningtable",  "window",       "desk",
        "toilet",        "door",        "tvmonitor",    "laptop",       "mouse",
        "remote",        "keyboard",    "cellphone",    "microwave",    "oven",
        "toaster",       "sink",        "refrigerator", "blender",      "book",
        "clock",         "vase",        "scissors",     "teddybear",    "hairdrier",
        "toothbrush",    "hairbrush"};
};
};  // namespace mobilint::post
