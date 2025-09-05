#include <maccel/maccel.h>

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "src/model/model.h"
#include "src/postprocess/ssd_post.h"
#include "src/postprocess/yolo_anchor_post.h"
#include "src/postprocess/yolo_anchorless_post.h"
#include "src/preprocess/preprocess.h"

using namespace mobilint;
using namespace mobilint::post;

bool contains(const std::string& str, const std::string& target) {
    std::string lower_str = str;
    std::string lower_target = target;

    std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::transform(lower_target.begin(), lower_target.end(), lower_target.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    return lower_str.find(lower_target) != std::string::npos;
}

std::string get_filename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

void inference_det_anchor(std::string mxq_path, std::string img_path,
                          std::string save_path) {
    NPUModel model(mxq_path);

    int imh = 640;
    int imw = 640;
    int num_class = 80;
    float conf_thres = 0.3;
    float iou_thres = 0.6;

    // YOLOv5 P5 anchors
    std::vector<std::vector<std::vector<double>>> anchors = {
        {{10, 13}, {16, 30}, {33, 23}},      // P3/8
        {{30, 61}, {62, 45}, {59, 119}},     // P4/16
        {{116, 90}, {156, 198}, {373, 326}}  // P5/32
    };

    int nl = anchors.size();

    YOLOAnchorPostProcessor post_processor(num_class, imh, imw, conf_thres, iou_thres,
                                           nl);
    post_processor.set_anchors(anchors);

    PreProcessor pre_processor;
    cv::Mat org_img = cv::imread(img_path);
    auto preprocessed = pre_processor(org_img, imh, imw, "yolo");

    auto output = model(std::move(preprocessed));

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;

    uint64_t ticket = post_processor.enqueue(output, boxes, scores, labels, extras);
    post_processor.receive(ticket);

    post_processor.plot_results(org_img, boxes, scores, labels);
    cv::imwrite(save_path, org_img);
}

void inference_det_anchorless(std::string mxq_path, std::string img_path,
                              std::string save_path) {
    NPUModel model(mxq_path);

    int imh = 640;
    int imw = 640;
    int num_class = 80;
    float conf_thres = 0.3;
    float iou_thres = 0.6;

    YOLOAnchorlessPostProcessor post_processor(num_class, imh, imw, conf_thres,
                                               iou_thres);
    PreProcessor pre_processor;
    cv::Mat org_img = cv::imread(img_path);

    auto preprocessed = pre_processor(org_img, imh, imw, "yolo");
    auto output = model(std::move(preprocessed));

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;

    uint64_t ticket = post_processor.enqueue(output, boxes, scores, labels, extras);
    post_processor.receive(ticket);

    post_processor.plot_results(org_img, boxes, scores, labels);
    cv::imwrite(save_path, org_img);
}

void inference_ssd(std::string mxq_path, std::string img_path, std::string save_path) {
    NPUModel model(mxq_path);

    int imh = 300;
    int imw = 300;
    int num_class = 91;
    float conf_thres = 0.3;
    float iou_thres = 0.6;

    SSDPostProcessor post_processor(num_class, imh, imw, conf_thres, iou_thres);
    PreProcessor pre_processor;
    cv::Mat org_img = cv::imread(img_path);

    auto preprocessed = pre_processor(org_img, imh, imw, "ssd");
    auto output = model(std::move(preprocessed));

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;

    uint64_t ticket = post_processor.enqueue(output, boxes, scores, labels, extras);
    post_processor.receive(ticket);

    post_processor.plot_results(org_img, boxes, scores, labels);
    cv::imwrite(save_path, org_img);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr,
                "User must provide two arguments: 1) mxq model path 2) image path\n");
        exit(1);
    }
    std::string mxq_path = argv[1];
    std::string img_path = argv[2];
    std::string save_path =
        img_path.substr(0, img_path.find_last_of('.')) + "_result.jpg";

    std::string model_name = get_filename(mxq_path);

    if (contains(model_name, "yolov5") || contains(model_name, "yolov7")) {
        inference_det_anchor(mxq_path, img_path, save_path);
    } else if (contains(model_name, "yolo")) {
        inference_det_anchorless(mxq_path, img_path, save_path);
    } else if (contains(model_name, "ssd")) {
        inference_ssd(mxq_path, img_path, save_path);
    } else {
        throw std::invalid_argument("Invalid model name");
    }

    return 0;
}