#include "preprocess.h"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace chrono;

static bool get_hw_from_imagesize(const ImageSize& s, int& h, int& w) {
    if (std::holds_alternative<std::pair<int, int>>(s)) {
        auto [hh, ww] = std::get<std::pair<int, int>>(s);
        h = hh;
        w = ww;
        return true;
    }
    return false;
}

static bool get_short_edge_from_imagesize(const ImageSize& s, int& short_edge) {
    if (std::holds_alternative<int>(s)) {
        short_edge = std::get<int>(s);
        return true;
    }
    return false;
}

PreProcessor::PreProcessor() {}
PreProcessor::PreProcessor(bool auto_padding, int stride) { set(auto_padding, stride); }

void PreProcessor::set(bool auto_padding, int stride) {
    m_auto_padding = auto_padding;
    m_stride = stride;
}

std::unique_ptr<float[]> PreProcessor::operator()(const cv::Mat& input,
                                                  const ModelInfo& cfg) {
    cv::Mat img = input.clone();
    for (const auto& p : cfg.m_preprocess_list) {
        switch (p.op) {
        case PreProcessOps::YOLO: {
            int h = 0, w = 0;
            if (!get_hw_from_imagesize(p.img_size, h, w)) {
                throw runtime_error("Error: YOLO preprocess requires img_size as [h,w]");
            }
            letter_box(img, Size(w, h), m_auto_padding, m_stride);
            normalize(img, "div255");
        } break;
        case PreProcessOps::RESIZE: {
            int h = 0, w = 0;
            int short_edge = 0;
            if (get_hw_from_imagesize(p.img_size, h, w)) {
                resize(img, cv::Size(w, h), p.style);
            } else if (get_short_edge_from_imagesize(p.img_size, short_edge)) {
                resize_short_edge(img, short_edge, p.style);
            } else {
                std::cerr << "[WARN] Resize has no size; skipped\n";
            }
            cv::imwrite("preprocessed.jpg", img);
        } break;
        case PreProcessOps::CENTERCROP: {
            int h = 0, w = 0;
            int s = 0;
            if (get_hw_from_imagesize(p.img_size, h, w)) {
                center_crop(img, cv::Size(w, h));
            } else if (get_short_edge_from_imagesize(p.img_size, s)) {
                center_crop(img, cv::Size(s, s));
            } else {
                std::cerr << "[WARN] CenterCrop has no size; skipped\n";
            }
        } break;
        case PreProcessOps::NORMALIZE: {
            normalize(img, p.style);
        } break;
        }
    }

    if (img.depth() != CV_32F) {
        img.convertTo(img, CV_32F);
    }

    int imh = img.rows;
    int imw = img.cols;
    int c = img.channels();
    CV_Assert(img.isContinuous());

    const float* src = img.ptr<float>(0);
    auto input_img = std::make_unique<float[]>(imw * imh * c);
    float* ptr = input_img.get();

#pragma omp parallel for
    for (int i = 0; i < imw * imh; i++) {
        for (int j = 0; j < c; j++) {
            ptr[c * i + j] = src[c * i + (2 - j)];
        }
    }
    return input_img;
}

std::unique_ptr<float[]> PreProcessor::operator()(const cv::Mat& input, int h, int w,
                                                  const std::string& type) {
    cv::Mat img = input.clone();
    if (type == "yolo") {
        letter_box(img, Size(w, h), m_auto_padding, m_stride);
        normalize(img, "div255");
    } else if (type == "ssd") {
        resize(img, Size(w, h), "bilinear");
        normalize(img, "tf");
    } else {
        throw runtime_error("Error: Unknown preprocess type: " + type);
    }

    if (img.depth() != CV_32F) {
        img.convertTo(img, CV_32F);
    }

    int imh = img.rows;
    int imw = img.cols;
    int c = img.channels();
    CV_Assert(img.isContinuous());

    const float* src = img.ptr<float>(0);
    auto input_img = std::make_unique<float[]>(imw * imh * c);
    float* ptr = input_img.get();

#pragma omp parallel for
    for (int i = 0; i < imw * imh; i++) {
        for (int j = 0; j < c; j++) {
            ptr[c * i + j] = src[c * i + (2 - j)];
        }
    }
    return input_img;
}

int PreProcessor::parse_interpolation(const std::string& s) {
    if (s == "nearest") return cv::INTER_NEAREST;
    if (s == "bicubic") return cv::INTER_CUBIC;
    if (s == "area") return cv::INTER_AREA;
    return cv::INTER_LINEAR;  // default
}

void PreProcessor::letter_box(cv::Mat& img, cv::Size im_shape, bool auto_padding,
                              int stride) {
    cv::Size current_shape = img.size();
    float ratio = min((float)im_shape.height / (float)current_shape.height,
                      (float)im_shape.width / (float)current_shape.width);

    int new_unpadw = (int)(floor(current_shape.width * ratio + 0.5));
    int new_unpadh = (int)(floor(current_shape.height * ratio + 0.5));

    int dw = im_shape.width - new_unpadw;
    int dh = im_shape.height - new_unpadh;

    if (auto_padding) {
        dw = dw % stride;
        dh = dh % stride;
    }

    float ddw = (float)dw / 2;
    float ddh = (float)dh / 2;

    if (current_shape.height != new_unpadh || current_shape.width != new_unpadw) {
        cv::resize(img, img, Size(new_unpadw, new_unpadh), 0, 0, cv::INTER_LINEAR);
    }

    int top = (int)(floor(ddh - 0.1 + 0.5));
    int bottom = (int)(floor(ddh + 0.1 + 0.5));
    int left = (int)(floor(ddw - 0.1 + 0.5));
    int right = (int)(floor(ddw + 0.1 + 0.5));

    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
}

void PreProcessor::resize(cv::Mat& img, cv::Size im_shape,
                          const std::string& interpolation) {
    cv::resize(img, img, im_shape, 0, 0, parse_interpolation(interpolation));
}

void PreProcessor::resize_short_edge(cv::Mat& img, int short_edge,
                                     const std::string& interpolation) {
    if (short_edge <= 0) return;

    int H = img.rows;
    int W = img.cols;
    if (H <= 0 || W <= 0) return;

    int min_hw = std::min(H, W);
    if (min_hw == short_edge) return;

    float scale = static_cast<float>(short_edge) / static_cast<float>(min_hw);

    int newH = static_cast<int>(std::round(H * scale));
    int newW = static_cast<int>(std::round(W * scale));

    newH = std::max(1, newH);
    newW = std::max(1, newW);

    cv::resize(img, img, cv::Size(newW, newH), 0, 0, parse_interpolation(interpolation));
}

void PreProcessor::center_crop(cv::Mat& img, cv::Size im_shape) {
    int imw = im_shape.width;
    int imh = im_shape.height;
    const int w = std::min(imw, img.cols);
    const int h = std::min(imh, img.rows);
    const int x = std::max(0, (img.cols - w) / 2);
    const int y = std::max(0, (img.rows - h) / 2);
    img = img(cv::Rect(x, y, w, h)).clone();
}

void PreProcessor::normalize(cv::Mat& img, const std::string& style) {
    if (img.depth() != CV_32F) {
        img.convertTo(img, CV_32F);
    }
    if (style == "torch") {
        const cv::Scalar mean_bgr(0.406, 0.456, 0.485);
        const cv::Scalar std_bgr(0.225, 0.224, 0.229);
        img *= (1.0f / 255.0f);
        cv::subtract(img, mean_bgr, img);
        cv::divide(img, std_bgr, img);
    } else if (style == "tf") {
        img.convertTo(img, CV_32F, 1.0 / 127.5, -1.0);
    } else if (style == "div255") {
        img *= (1.0f / 255.0f);
    } else {
        throw runtime_error("Error: Unknown normalize style: " + style);
    }
}
