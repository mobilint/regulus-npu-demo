#pragma once
#include <iostream>
#include <string>
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

struct PreProcessInfo {
    PreProcessOps op;
    std::string style;
    std::pair<int, int> img_size{0, 0};
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