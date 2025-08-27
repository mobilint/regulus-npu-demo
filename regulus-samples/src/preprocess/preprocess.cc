#include "preprocess.h"

#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace chrono;

PreProcessor::PreProcessor() {}
PreProcessor::PreProcessor(int imh, int imw, bool is_ssd, bool auto_padding, int stride) {
    set(imh, imw, is_ssd, auto_padding, stride);
}

void PreProcessor::set(int imh, int imw, bool is_ssd, bool auto_padding, int stride) {
    m_imh = imh;
    m_imw = imw;
    m_is_ssd = is_ssd;
    m_auto_padding = auto_padding;
    m_stride = stride;
}

std::unique_ptr<float[]> PreProcessor::operator()(cv::Mat image) {
    if (m_is_ssd) {
        cv::resize(image, image, Size(m_imw, m_imh), cv::INTER_LINEAR);
        int c = image.channels();
        auto input_img = std::make_unique<float[]>(m_imw * m_imh * c);
        float* ptr = input_img.get();
#pragma omp parallel for
        for (int i = 0; i < m_imw * m_imh; i++) {
            for (int j = 0; j < c; j++) {
                ptr[3 * i + j] = ((float)image.data[3 * i + (2 - j)] - 127.5f) / 127.5f;
            }
        }
        return input_img;
    } else {
        image = letter_box(image, Size(m_imw, m_imh), m_auto_padding, m_stride);
        int c = image.channels();
        auto input_img = std::make_unique<float[]>(m_imw * m_imh * c);
        float* ptr = input_img.get();
#pragma omp parallel for
        for (int i = 0; i < m_imw * m_imh; i++) {
            for (int j = 0; j < c; j++) {
                ptr[3 * i + j] = (float)image.data[3 * i + (2 - j)] / 255.0f;
            }
        }
        return input_img;
    }
}

cv::Mat PreProcessor::letter_box(cv::Mat image, cv::Size im_shape, bool auto_padding,
                                 int stride) {
    cv::Size current_shape = image.size();
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
        cv::resize(image, image, Size(new_unpadw, new_unpadh), cv::INTER_LINEAR);
    }

    int top = (int)(floor(ddh - 0.1 + 0.5));
    int bottom = (int)(floor(ddh + 0.1 + 0.5));
    int left = (int)(floor(ddw - 0.1 + 0.5));
    int right = (int)(floor(ddw + 0.1 + 0.5));

    cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    return image;
}
