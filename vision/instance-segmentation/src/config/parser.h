#pragma once
#include <opencv2/core.hpp>

#include "types.h"

bool open_yaml_relaxed(const std::string& path, cv::FileStorage& fs);
bool parse_yaml(const std::string& yaml_path, ModelInfo& config);
std::string lower_copy(std::string s);
void parse_one_pre_block(const std::string& key, const cv::FileNode& node,
                         ModelInfo& config);
ImageSize read_img_size(const cv::FileNode& node);
std::string read_string(const cv::FileNode& node);
Task parse_task(const std::string& s);
std::vector<std::vector<std::vector<double>>> normalize_anchors(
    const cv::FileNode& anchors);
static bool is_empty_img_size(const ImageSize& s);
