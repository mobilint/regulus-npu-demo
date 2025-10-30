#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "utils.h"

bool loadYamlConfig(const std::string& yaml_path, YamlConfig& cfg) {
    cv::FileStorage fs(yaml_path, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    if (!fs.isOpened()) return false;

    cv::FileNode camera = fs["camera"];
    if (!camera.empty()) {
        auto node = camera["device"];
        if (!node.empty()) node >> cfg.camera_device;

        node = camera["width"];
        if (!node.empty()) node >> cfg.camera_width;

        node = camera["height"];
        if (!node.empty()) node >> cfg.camera_height;

        node = camera["fps"];
        if (!node.empty()) node >> cfg.camera_fps;
    }

    cv::FileNode display = fs["display"];
    if (!display.empty()) {
        auto node = display["width"];
        if (!node.empty()) node >> cfg.display_width;

        node = display["height"];
        if (!node.empty()) node >> cfg.display_height;
    }

    return true;
}

std::string findVideoDevice() {
    for (int i = 0; i < 10; i++) {
        cv::VideoCapture cap(i, cv::CAP_V4L2);

        if (cap.isOpened()) {
            std::string path = "/dev/video" + std::to_string(i);
            printf("Device found at index %d (%s)\n", i, path.c_str());
            cap.release();
            return path;
        }
    }
    return "";
}