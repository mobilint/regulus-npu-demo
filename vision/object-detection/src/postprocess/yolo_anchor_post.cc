#include "yolo_anchor_post.h"

#include <chrono>

#include "commons.h"
using namespace std::chrono;

using namespace mobilint::post;

mobilint::post::YOLOAnchorPostProcessor::YOLOAnchorPostProcessor(int imh, int imw,
                                                                 const ModelInfo& cfg) {
    const auto& post_info = cfg.m_postprocess;
    int nc = post_info.num_classes;
    int nl = post_info.num_layers;
    float conf_thres = post_info.conf_thres;
    float iou_thres = post_info.iou_thres;

    set_anchors(post_info.anchors);
    set_params(nc, imh, imw, conf_thres, iou_thres, nl);
}

mobilint::post::YOLOAnchorPostProcessor::YOLOAnchorPostProcessor(int nc, int imh,
                                                                 int imw) {
    set_params(nc, imh, imw);
    set_default_anchors();
}

mobilint::post::YOLOAnchorPostProcessor::YOLOAnchorPostProcessor(int nc, int imh, int imw,
                                                                 float conf_thres,
                                                                 float iou_thres, int nl,
                                                                 int max_num_threads) {
    set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
    set_default_anchors();
}

void mobilint::post::YOLOAnchorPostProcessor::set_params(int nc, int imh, int imw,
                                                         float conf_thres,
                                                         float iou_thres, int nl,
                                                         int max_num_threads) {
    PostProcessor::set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
    m_no = m_nc + 5 + m_nextra;
    mType = PostType::BASE;
    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);
}

std::vector<std::vector<int>> mobilint::post::YOLOAnchorPostProcessor::generate_grids(
    int imh, int imw, std::vector<int> strides) {
    std::vector<std::vector<int>> all_grids;
    for (int i = 0; i < strides.size(); i++) {
        int grid_h = imh / strides[i];
        int grid_w = imw / strides[i];
        int grid_size = grid_h * grid_w * 2;

        std::vector<int> grids;
        grids.reserve(grid_size);
        for (int p = 0; p < grid_h * grid_w; ++p) {
            int x = p % grid_w;
            int y = p / grid_w;
            grids.push_back(x);
            grids.push_back(y);
        }

        all_grids.push_back(grids);
    }
    return all_grids;
}

std::vector<int> mobilint::post::YOLOAnchorPostProcessor::generate_strides(int nl) {
    std::vector<int> strides;
    if (nl == 2) {
        return {16, 32};
    }

    for (int i = 0; i < nl; ++i) {
        strides.push_back(1 << (3 + i));  // 8,16,32,64...
    }
    return strides;
}

void mobilint::post::YOLOAnchorPostProcessor::set_default_anchors() {
    if (m_nl == 2) {
        // YOLOv3 Tiny anchors
        m_anchors = {{{10, 14}, {23, 27}, {37, 58}}, {{81, 82}, {135, 169}, {344, 319}}};
    } else if (m_nl == 3) {
        // YOLOv5 P5 anchors
        m_anchors = {
            {{10, 13}, {16, 30}, {33, 23}},      // P3/8
            {{30, 61}, {62, 45}, {59, 119}},     // P4/16
            {{116, 90}, {156, 198}, {373, 326}}  // P5/32
        };

        // YOLOv7 anchors
        // m_anchors = {
        // 	{{12,16}, {19,36}, {40,28}},       // P3/8
        // 	{{36,75}, {76,55}, {72,146}},      // P4/16
        // 	{{142,110}, {192,243}, {459,401}}   // P5/32
        // };

    } else if (m_nl == 4) {
        // YOLOv5 P6 anchors
        m_anchors = {
            {{19, 27}, {44, 40}, {38, 94}},        // P3/8
            {{96, 68}, {86, 152}, {180, 137}},     // P4/16
            {{140, 301}, {303, 264}, {238, 542}},  // P5/32
            {{436, 615}, {739, 380}, {925, 792}}   // P6/64
        };
    } else {
        throw std::invalid_argument("Number of detection layers should 3 or 4");
    }
    m_na = m_anchors[0].size();
}

void mobilint::post::YOLOAnchorPostProcessor::set_anchors(
    const std::vector<std::vector<std::vector<double>>>& anchors) {
    m_anchors = anchors;
    m_na = m_anchors[0].size();
}

/*
        Decoding and masking with conf threshold
*/
void mobilint::post::YOLOAnchorPostProcessor::decode_conf_thres(
    const std::vector<float>& npu_out, const std::vector<int>& grid, int stride,
    const std::vector<std::vector<double>>& anchor,
    std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<std::pair<float, int>>& pred_scores, std::vector<int>& pred_label) {
    int grid_h = m_imh / stride;
    int grid_w = m_imw / stride;

    for (int i = 0; i < m_na; i++) {
#pragma omp parallel for num_threads(m_max_num_threads) \
    shared(pred_boxes, pred_label, pred_scores)
        for (int j = 0; j < grid_h * grid_w; j++) {
            int idx = j * m_na * m_no + i * m_no;
            int grid_idx = j * 2;

            if (npu_out[idx + 4] > m_inverse_conf_thres) {
                float conf_value = sigmoid(npu_out[idx + 4]);
                float x = (sigmoid(npu_out[idx]) * 2 - 0.5 + grid[grid_idx]) * stride;
                float y =
                    (sigmoid(npu_out[idx + 1]) * 2 - 0.5 + grid[grid_idx + 1]) * stride;
                float sx = sigmoid(npu_out[idx + 2]) * 2.f;
                float sy = sigmoid(npu_out[idx + 3]) * 2.f;
                float w = sx * sx * anchor[i][0];
                float h = sy * sy * anchor[i][1];

                std::array<float, 4> pred_box = {x, y, w, h};
                for (int k = 0; k < m_nc; k++) {
                    float cls_score = sigmoid(npu_out[idx + 5 + k]);
                    if (conf_value * cls_score > m_conf_thres) {
#pragma omp critical
                        {
                            pred_label.push_back(k);
                            pred_boxes.push_back(pred_box);
                            pred_scores.push_back(std::make_pair(conf_value * cls_score,
                                                                 pred_scores.size()));
                        }
                    }
                }
            }
        }
    }
}

/*
    npu_outs shape:
    [20, 20, 255]
    [40, 40, 255]
    [80, 80, 255]
*/
void mobilint::post::YOLOAnchorPostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() != m_nl)
        throw std::invalid_argument(
            "Size of model outputs does not match "
            "number of detection layers, expected " +
            std::to_string(m_nl) + " but received " + std::to_string(npu_outs.size()));

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<int> pred_labels;

    for (int i = 0; i < m_nl; i++) {
        decode_conf_thres(npu_outs[m_nl - i - 1], m_grids[i], m_strides[i], m_anchors[i],
                          pred_boxes, pred_scores, pred_labels);
    }
    xywh2xyxy(pred_boxes);
    nms(pred_boxes, pred_scores, pred_labels, final_boxes, final_scores, final_labels);
    double end = set_timer();
}

void mobilint::post::YOLOAnchorPostProcessor::worker() {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Worker | ";

    print(title + "Start");
    while (!destroyed) {
        std::unique_lock<std::mutex> lk(mMutexIn);
        if (mQueueIn.empty()) {
            mCondIn.wait(lk, [this] { return !mQueueIn.empty() || destroyed; });
        }

        if (destroyed) {
            break;
        }

        auto k = mQueueIn.front();
        mQueueIn.pop();
        lk.unlock();

        auto start = set_timer();

        run_postprocess(k.npu_outs);
        k.boxes = get_result_box();
        k.scores = get_result_score();
        k.labels = get_result_label();
        // k.extras = get_result_extra();

        auto end = set_timer();
        auto elapsed = std::to_string(end - start);

        print(title + "Postprocessing time: " + elapsed);
        print(title + "Number of detections " + std::to_string(k.boxes.size()));

        std::unique_lock<std::mutex> lk2(mMutexOut);
        mOut.push_back(k.id);
        lk2.unlock();

        std::unique_lock<std::mutex> lk_(mMutexOut);  // JUST IN CASE
        mCondOut.notify_all();
        lk_.unlock();  // JUST IN CASE
    }
    print(title + "Finish");
}
