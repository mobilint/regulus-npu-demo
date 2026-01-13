#pragma once
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

enum class PreProcessOps {
    YOLO,
    CENTERCROP,
    NORMALIZE,
    RESIZE,
};

enum class Task {
    CLS,
    DET,
    SEG,
    POSE,
    FACE,
};

using ImageSize = std::variant<std::monostate, int, std::pair<int, int>>;

struct PreProcessInfo {
    PreProcessOps op;
    std::string style;
    ImageSize img_size{};
};

struct PostProcessInfo {
    Task task = Task::DET;
    std::string type = "yolo";
    int num_classes = 0;
    int num_layers = 0;
    float conf_thres = 0.f;
    float iou_thres = 0.f;
    std::vector<std::vector<std::vector<double>>> anchors;
};

struct ModelInfo {
    std::vector<PreProcessInfo> m_preprocess_list;
    PostProcessInfo m_postprocess;
};