#ifndef APP_IMAGE_H_
#define APP_IMAGE_H_

#include <opencv2/opencv.hpp>

namespace img {
cv::Mat get_mask(cv::Mat frame);
cv::Mat get_alpha(cv::Mat frame);
}  // namespace img

#endif
