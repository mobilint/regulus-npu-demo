#include "commons.h"

#include <cmath>

void mobilint::post::make_anchors(std::vector<std::array<float, 2>>& anchors_out,
                                  std::vector<float>& strides_out, int imh, int imw,
                                  const std::vector<int>& strides,
                                  float grid_cell_offset) {
    anchors_out.clear();
    strides_out.clear();

    size_t total_anchor_count = 0;
    for (int stride : strides) {
        int h = imh / stride;
        int w = imw / stride;
        total_anchor_count += static_cast<size_t>(h) * w;
    }

    anchors_out.reserve(total_anchor_count);
    strides_out.reserve(total_anchor_count);

    for (int stride : strides) {
        int h = imh / stride;
        int w = imw / stride;

        float stride_f = static_cast<float>(stride);

        for (int y = 0; y < h; ++y) {
            float cy = static_cast<float>(y) + grid_cell_offset;
            for (int x = 0; x < w; ++x) {
                float cx = static_cast<float>(x) + grid_cell_offset;
                anchors_out.push_back({cx, cy});
                strides_out.emplace_back(stride_f);
            }
        }
    }
}

float mobilint::post::softmax_inplace_idx(const std::vector<float>& npu_out,
                                          int start_idx, int end_idx) {
    if (end_idx <= start_idx) return 0.f;
    float m = npu_out[start_idx];
    for (int i = start_idx + 1; i < end_idx; ++i) m = std::max(m, npu_out[i]);

    double denom = 0.0, num = 0.0;
    for (int i = start_idx; i < end_idx; ++i) {
        const double t = std::exp(double(npu_out[i] - m));
        denom += t;
        num += t * double(i - start_idx);
    }
    return float(num / denom);
}

float mobilint::post::sigmoid(const float& num) { return 1.f / (1.f + std::exp(-num)); }

float mobilint::post::inverse_sigmoid(const float& num) { return -log(1 / num - 1); }

float mobilint::post::area(const float& xmin, const float& ymin, const float& xmax,
                           const float& ymax) {
    float width = xmax - xmin;
    float height = ymax - ymin;
    return std::max(0.f, width) * std::max(0.f, height);
}

float mobilint::post::get_iou(const std::array<float, 4>& box1,
                              const std::array<float, 4>& box2) {
    const float iw =
        std::max(0.f, std::min(box1[2], box2[2]) - std::max(box1[0], box2[0]));
    const float ih =
        std::max(0.f, std::min(box1[3], box2[3]) - std::max(box1[1], box2[1]));
    const float inter = iw * ih;

    const float aw = std::max(0.f, box1[2] - box1[0]);
    const float ah = std::max(0.f, box1[3] - box1[1]);
    const float bw = std::max(0.f, box2[2] - box2[0]);
    const float bh = std::max(0.f, box2[3] - box2[1]);

    const float ua = aw * ah + bw * bh - inter;
    return inter / (ua + 1e-6f);
}

void mobilint::post::xywh2xyxy(std::vector<std::array<float, 4>>& pred_boxes) {
    for (auto& b : pred_boxes) {
        const float cx = b[0], cy = b[1];
        const float w = b[2], h = b[3];
        b[0] = cx - 0.5f * w;
        b[1] = cy - 0.5f * h;
        b[2] = cx + 0.5f * w;
        b[3] = cy + 0.5f * h;
    }
}

void mobilint::post::compute_ratio_pads(const cv::Mat& im, const int& input_w,
                                        const int& input_h, float& ratio, float& xpad,
                                        float& ypad) {
    cv::Size size = im.size();
    compute_ratio_pads(size.width, size.height, input_w, input_h, ratio, xpad, ypad);
}

/*
    Compute the ratio and pads needed to switch between
    original image size and model input image size
*/
void mobilint::post::compute_ratio_pads(const int& org_w, const int& org_h,
                                        const int& input_w, const int& input_h,
                                        float& ratio, float& xpad, float& ypad) {
    ratio = std::min(input_w / float(org_w), input_h / float(org_h));
    const float new_w = ratio * org_w;
    const float new_h = ratio * org_h;
    xpad = (input_w - new_w) * 0.5f;
    ypad = (input_h - new_h) * 0.5f;
}

cv::Mat mobilint::post::interpolate(const cv::Mat& input, const cv::Size& size,
                                    int mode) {
    // Resize the input tensor using the specified interpolation mode
    cv::Mat output;
    cv::resize(input, output, size, 0, 0, mode);

    return output;
}

cv::Mat mobilint::post::unpad_yolov8_seg(const cv::Mat& image, int xpad, int ypad) {
    const int cols = image.cols, rows = image.rows;
    xpad = std::clamp(xpad, 0, cols / 2);
    ypad = std::clamp(ypad, 0, rows / 2);
    const int w = cols - 2 * xpad;
    const int h = rows - 2 * ypad;
    return image(cv::Rect(xpad, ypad, w, h)).clone();
}