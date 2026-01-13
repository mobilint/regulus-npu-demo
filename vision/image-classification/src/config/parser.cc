#include "parser.h"

#include <algorithm>
#include <cctype>
#include <fstream>

bool open_yaml_relaxed(const std::string& path, cv::FileStorage& fs) {
    try {
        fs.open(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    } catch (const cv::Exception&) {
    }
    if (fs.isOpened()) return true;

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    std::string text((std::istreambuf_iterator<char>(ifs)), {});
    if (text.size() >= 3 && (unsigned char)text[0] == 0xEF &&
        (unsigned char)text[1] == 0xBB && (unsigned char)text[2] == 0xBF) {
        text.erase(0, 3);
    }

    auto starts_with = [&](const char* p) { return text.rfind(p, 0) == 0; };
    if (!starts_with("%YAML:")) {
        text = std::string("%YAML:1.0\n") + text;
    }

    try {
        fs.open(text, cv::FileStorage::READ | cv::FileStorage::MEMORY |
                          cv::FileStorage::FORMAT_YAML);
    } catch (const cv::Exception&) {
    }
    return fs.isOpened();
}

bool parse_yaml(const std::string& yaml_path, ModelInfo& config) {
    cv::FileStorage fs;
    if (!open_yaml_relaxed(yaml_path, fs)) {
        std::cerr << "Failed to open YAML: " << yaml_path << std::endl;
        return false;
    }

    config.m_preprocess_list.clear();
    cv::FileNode pre = fs["Pre-processing"];
    if (pre.empty()) {
        std::cerr << "Missing 'Pre-processing'\n";
        return false;
    }

    cv::FileNode preOrder = fs["Pre-order"];
    if (!preOrder.empty() && preOrder.isSeq()) {
        for (auto it = preOrder.begin(); it != preOrder.end(); ++it) {
            std::string name;
            (*it) >> name;
            if (name.empty()) continue;
            auto node = pre[name];
            if (node.empty()) {
                std::cerr << "[WARN] Pre-Order item '" << name
                          << "' not found in 'Pre-processing'\n";
                continue;
            }
            parse_one_pre_block(name, node, config);
        }
    }

    cv::FileNode post = fs["Post-processing"];
    if (post.empty()) {
        std::cerr << "Missing 'Post-processing'\n";
        return false;
    }

    PostProcessInfo post_info;
    post_info.task = parse_task(read_string(post["task"]));
    {
        auto n = post["type"];
        if (!n.empty()) n >> post_info.type;
    }
    {
        auto n = post["nc"];
        if (!n.empty()) n >> post_info.num_classes;
    }
    {
        auto n = post["nl"];
        if (!n.empty()) n >> post_info.num_layers;
    }
    {
        auto n = post["conf_thres"];
        if (!n.empty()) n >> post_info.conf_thres;
    }
    {
        auto n = post["iou_thres"];
        if (!n.empty()) n >> post_info.iou_thres;
    }
    {
        cv::FileNode an = post["anchors"];
        if (!an.empty()) {
            if (an.type() == cv::FileNode::SEQ) {
                post_info.anchors = normalize_anchors(an);
                post_info.num_layers = static_cast<int>(post_info.anchors.size());
            } else {
                std::cerr << "[WARN] 'anchors' is not a sequence; ignored.\n";
            }
        }
    }
    config.m_postprocess = post_info;
    return true;
}

std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

void parse_one_pre_block(const std::string& key, const cv::FileNode& node,
                         ModelInfo& config) {
    std::string lower = lower_copy(key);

    if (lower == "yolopre" || lower == "yolo") {
        PreProcessInfo p{};
        p.op = PreProcessOps::YOLO;
        p.style.clear();
        p.img_size = read_img_size(node);
        if (is_empty_img_size(p.img_size))
            std::cerr << "[WARN] YoloPre.img_size missing/invalid\n";
        config.m_preprocess_list.push_back(std::move(p));

    } else if (lower == "centercrop" || lower == "center_crop") {
        PreProcessInfo p{};
        p.op = PreProcessOps::CENTERCROP;
        p.style.clear();
        p.img_size = read_img_size(node);
        if (is_empty_img_size(p.img_size))
            std::cerr << "[WARN] CenterCrop.img_size missing/invalid\n";
        config.m_preprocess_list.push_back(std::move(p));

    } else if (lower == "normalize") {
        PreProcessInfo p{};
        p.op = PreProcessOps::NORMALIZE;
        p.style = read_string(node["style"]);
        p.img_size = std::monostate{};
        config.m_preprocess_list.push_back(std::move(p));

    } else if (lower == "resize") {
        PreProcessInfo p{};
        p.op = PreProcessOps::RESIZE;
        p.img_size = read_img_size(node);
        p.style = read_string(node["interpolation"]);
        if (is_empty_img_size(p.img_size))
            std::cerr << "[WARN] Resize.size/img_size missing/invalid\n";
        config.m_preprocess_list.push_back(std::move(p));

    } else {
        std::cerr << "[WARN] Unknown pre op: '" << key << "' (skipped)\n";
    }
}

ImageSize read_img_size(const cv::FileNode& node) {
    cv::FileNode n = node["size"];
    if (n.empty()) n = node["img_size"];
    if (n.empty()) return std::monostate{};

    if (n.isSeq()) {
        std::vector<int> v;
        n >> v;
        if (v.size() >= 2) return std::pair<int, int>{v[0], v[1]};
        if (v.size() == 1) return v[0];
        return std::monostate{};
    } else {
        // scalar
        int s = 0;
        if (n.isInt()) {
            n >> s;
        } else if (n.isReal()) {
            double d = 0.0;
            n >> d;
            s = static_cast<int>(d);
        } else {
            n >> s;
        }
        return s;
    }
}

std::string read_string(const cv::FileNode& node) {
    if (node.empty()) return std::string();
    std::string s;
    node >> s;
    return s;
}

Task parse_task(const std::string& s) {
    std::string t = s;
    std::transform(t.begin(), t.end(), t.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (t == "classification" || t == "image_classification") return Task::CLS;
    if (t == "object_detection" || t == "detection") return Task::DET;
    if (t == "segmentation" || t == "instance_segmentation") return Task::SEG;
    if (t == "pose_estimation") return Task::POSE;
    if (t == "face_detection") return Task::FACE;
    return Task::DET;
}

std::vector<std::vector<std::vector<double>>> normalize_anchors(
    const cv::FileNode& anchors) {
    std::vector<std::vector<std::vector<double>>> out;

    for (auto it = anchors.begin(); it != anchors.end(); ++it) {
        cv::FileNode layer_node = *it;
        std::vector<std::vector<double>> layer_anchors;

        if (layer_node.type() == cv::FileNode::SEQ) {
            if (layer_node.size() > 0 &&
                (*layer_node.begin()).type() == cv::FileNode::SEQ) {
                for (auto jt = layer_node.begin(); jt != layer_node.end(); ++jt) {
                    std::vector<double> paird;
                    (*jt) >> paird;
                    if (paird.size() == 2) {
                        layer_anchors.push_back({paird[0], paird[1]});
                    }
                }
            } else {
                std::vector<double> flatd;
                layer_node >> flatd;
                if (flatd.size() % 2 != 0) {
                    std::cerr << "[WARN] anchors row length is odd; last value dropped\n";
                    flatd.pop_back();
                }
                for (size_t i = 0; i + 1 < flatd.size(); i += 2) {
                    layer_anchors.push_back({flatd[i], flatd[i + 1]});
                }
            }
        }
        out.push_back(std::move(layer_anchors));
    }
    return out;
}

bool is_empty_img_size(const ImageSize& s) {
    return std::holds_alternative<std::monostate>(s);
}