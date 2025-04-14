#include "app_image.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace img {
    cv::Mat get_mask(cv::Mat frame) {
	cv::Mat gray;
	cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
	cv::Mat mask;
	cv::inRange(gray, cv::Scalar(0), cv::Scalar(1), mask);
	cv::bitwise_not(mask, mask);

	return mask;
    }

    cv::Mat get_alpha(cv::Mat frame) {
	std::vector<cv::Mat> channels(4);
	cv::split(frame, channels);
	cv::Mat alpha = channels[3];

        return alpha;
    }
}
