#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace mobilint::post {

void make_anchors(std::vector<std::array<float, 2>>& anchors_out,
                  std::vector<float>& strides_out, int imh, int imw,
                  const std::vector<int>& strides, float grid_cell_offset = 0.5);
float softmax_inplace_idx(const std::vector<float>& npu_out, int start_idx, int end_idx);
float sigmoid(const float& num);
float inverse_sigmoid(const float& num);
float area(const float& xmin, const float& ymin, const float& xmax, const float& ymax);
float get_iou(const std::array<float, 4>& box1, const std::array<float, 4>& box2);
void xywh2xyxy(std::vector<std::array<float, 4>>& pred_boxes);
void compute_ratio_pads(const cv::Mat& im, const int& input_w, const int& input_h,
                        float& ratio, float& xpad, float& ypad);
void compute_ratio_pads(const int& org_w, const int& org_h, const int& input_w,
                        const int& input_h, float& ratio, float& xpad, float& ypad);

cv::Mat interpolate(const cv::Mat& input, const cv::Size& size, int mode);
cv::Mat unpad_yolov8_seg(const cv::Mat& image, int xpad, int ypad);

template <typename FloatContainer>
void softmax_inplace(FloatContainer& con) {
    float sum = 0;
    for (auto v : con) {
        sum += std::exp(v);
    }
    for (auto& v : con) {
        v = std::exp(v) / sum;
    }
}

}  // namespace mobilint::post
