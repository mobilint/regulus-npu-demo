#pragma once
#include <opencv2/opencv.hpp>
#include <string>

#include "types.h"

class PreProcessor {
public:
    PreProcessor();
    PreProcessor(bool auto_padding, int stride);
    void set(bool auto_padding = false, int stride = 32);
    std::unique_ptr<float[]> operator()(const cv::Mat& input, const ModelInfo& cfg);
    std::unique_ptr<float[]> operator()(const cv::Mat& input, int h, int w,
                                        const std::string& type = "yolo");

private:
    int parse_interpolation(const std::string& s);
    void letter_box(cv::Mat& img, cv::Size im_shape, bool auto_padding = false,
                    int stride = 32);
    void resize(cv::Mat& img, cv::Size im_shape, const std::string& interpolation);
    void resize_short_edge(cv::Mat& img, int short_edge,
                           const std::string& interpolation);
    void center_crop(cv::Mat& img, cv::Size im_shape);
    void normalize(cv::Mat& img, const std::string& style);

    bool m_auto_padding = false;
    int m_stride = 32;
};