#include "commons.h"

#include <cmath>

void mobilint::post::make_anchors(std::vector<std::vector<float>>& anchors_out,
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
                anchors_out.emplace_back(std::vector<float>{cx, cy});
                strides_out.emplace_back(stride_f);
            }
        }
    }
}

float mobilint::post::softmax_inplace_idx(const std::vector<float>& npu_out,
                                          int start_idx, int end_idx) {
    float sum = 0, result = 0;
    for (int i = start_idx; i < end_idx; i++) {
        sum += exp(npu_out[i]);
    }
    for (int i = start_idx; i < end_idx; i++) {
        result += exp(npu_out[i]) / sum * (i - start_idx);
    }
    return result;
}

float mobilint::post::sigmoid(const float& num) { return 1 / (1 + exp(-(float)num)); }

float mobilint::post::inverse_sigmoid(const float& num) { return -log(1 / num - 1); }

float mobilint::post::area(const float& xmin, const float& ymin, const float& xmax,
                           const float& ymax) {
    float width = xmax - xmin;
    float height = ymax - ymin;

    if (width < 0) return 0;

    if (height < 0) return 0;

    return width * height;
}

float mobilint::post::get_iou(const std::array<float, 4>& box1,
                              const std::array<float, 4>& box2) {
    float epsilon = 1e-6;

    // Coordinated of the overlapped region(intersection of two boxes)
    float overlap_xmin = std::max(box1[0], box2[0]);
    float overlap_ymin = std::max(box1[1], box2[1]);
    float overlap_xmax = std::min(box1[2], box2[2]);
    float overlap_ymax = std::min(box1[3], box2[3]);

    // Calculate areas
    float overlap_area = area(overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax);
    float area1 = area(box1[0], box1[1], box1[2], box1[3]);
    float area2 = area(box2[0], box2[1], box2[2], box2[3]);
    float iou = overlap_area / (area1 + area2 - overlap_area + epsilon);

    return iou;
}

void mobilint::post::xywh2xyxy(std::vector<std::array<float, 4>>& pred_boxes) {
    for (uint32_t i = 0; i < pred_boxes.size(); i++) {
        float cx = pred_boxes[i][0];
        float cy = pred_boxes[i][1];

        pred_boxes[i][0] = cx - pred_boxes[i][2] * 0.5;
        pred_boxes[i][1] = cy - pred_boxes[i][3] * 0.5;
        pred_boxes[i][2] = cx + pred_boxes[i][2] * 0.5;
        pred_boxes[i][3] = cy + pred_boxes[i][3] * 0.5;
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
    if (org_w > org_h) {
        ratio = (float)input_w / org_w;
        xpad = 0;
        ypad = (input_h - ratio * org_h) / 2;
    } else {
        ratio = (float)input_h / org_h;
        xpad = (input_w - ratio * org_w) / 2;
        ypad = 0;
    }
}

cv::Mat mobilint::post::interpolate(const cv::Mat& input, const cv::Size& size,
                                    int mode) {
    // Resize the input tensor using the specified interpolation mode
    cv::Mat output;
    cv::resize(input, output, size, 0, 0, mode);

    return output;
}

cv::Mat mobilint::post::unpad_yolov8_seg(const cv::Mat& image, int xpad, int ypad) {
    int rows = image.rows;
    int cols = image.cols;

    int width = cols - 2 * xpad;
    int height = rows - 2 * ypad;

    cv::Rect rect(xpad, ypad, width, height);

    cv::Mat roi = image(rect);

    cv::Mat cropped;
    roi.copyTo(cropped);

    return cropped;
}