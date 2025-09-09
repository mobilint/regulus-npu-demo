#include "yolo_anchor_seg_post.h"

#include <chrono>

#include "commons.h"

using namespace std::chrono;

using namespace mobilint::post;

mobilint::post::YOLOAnchorSegPostProcessor::YOLOAnchorSegPostProcessor(
    int imh, int imw, const ModelInfo& cfg) {
    const auto& post_info = cfg.m_postprocess;
    int nc = post_info.num_classes;
    int nl = post_info.num_layers;
    float conf_thres = post_info.conf_thres;
    float iou_thres = post_info.iou_thres;

    set_anchors(post_info.anchors);
    set_params(nc, imh, imw, conf_thres, iou_thres, nl);
}

mobilint::post::YOLOAnchorSegPostProcessor::YOLOAnchorSegPostProcessor(int nc, int imh,
                                                                       int imw) {
    set_params(nc, imh, imw);
    set_default_anchors();
}

mobilint::post::YOLOAnchorSegPostProcessor::YOLOAnchorSegPostProcessor(
    int nc, int imh, int imw, float conf_thres, float iou_thres, int nl,
    int max_num_threads) {
    set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
    set_default_anchors();
}

void mobilint::post::YOLOAnchorSegPostProcessor::set_params(int nc, int imh, int imw,
                                                            float conf_thres,
                                                            float iou_thres, int nl,
                                                            int max_num_threads) {
    PostProcessor::set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
    mType = PostType::SEG;
    m_proto_stride = 4;
    m_proto_h = m_imh / m_proto_stride;
    m_proto_w = m_imw / m_proto_stride;
    m_nextra = 32;
    m_no = m_nc + 5 + m_nextra;

    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);
}

std::vector<std::vector<int>> mobilint::post::YOLOAnchorSegPostProcessor::generate_grids(
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

std::vector<int> mobilint::post::YOLOAnchorSegPostProcessor::generate_strides(int nl) {
    std::vector<int> strides;
    if (nl == 2) {
        return {16, 32};
    }

    for (int i = 0; i < nl; ++i) {
        strides.push_back(1 << (3 + i));  // 8,16,32,64...
    }
    return strides;
}

void mobilint::post::YOLOAnchorSegPostProcessor::set_default_anchors() {
    if (m_nl == 2) {
        // YOLOv3 Tiny anchors
        m_anchors = {{{10, 14}, {23, 27}, {37, 58}}, {{81, 82}, {135, 169}, {344, 319}}};
    }
    if (m_nl == 3) {
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

void mobilint::post::YOLOAnchorSegPostProcessor::set_anchors(
    const std::vector<std::vector<std::vector<double>>>& anchors) {
    m_anchors = anchors;
    m_na = m_anchors[0].size();
}

std::vector<std::array<float, 4>>
mobilint::post::YOLOAnchorSegPostProcessor::downsample_boxes(
    const std::vector<std::array<float, 4>>& boxes) {
    std::vector<std::array<float, 4>> out;
    out.reserve(boxes.size());
    for (auto b : boxes) {
        for (int j = 0; j < 4; ++j) b[j] /= m_proto_stride;
        out.push_back(b);
    }
    return out;
}

void mobilint::post::YOLOAnchorSegPostProcessor::process_mask(
    const std::vector<float>& proto, const std::vector<std::vector<float>>& masks,
    const std::vector<std::array<float, 4>>& boxes, const std::vector<int>& labels) {
    int num_boxes = boxes.size();
    m_instance_masks.clear();

    if (num_boxes == 0) {
        return;
    }

    auto boxes_down = downsample_boxes(boxes);
    int matmul_col = 32;

    m_instance_masks.resize(num_boxes);

#pragma omp parallel for num_threads(m_max_num_threads)
    for (int i = 0; i < num_boxes; ++i) {
        const auto& box = boxes_down[i];
        const auto& coeff = masks[i];

        int x_min = std::max(static_cast<int>(std::floor(box[0])), 0);
        int y_min = std::max(static_cast<int>(std::floor(box[1])), 0);
        int x_max = std::min(static_cast<int>(std::ceil(box[2])), m_proto_w - 1);
        int y_max = std::min(static_cast<int>(std::ceil(box[3])), m_proto_h - 1);

        cv::Mat inst_mask = cv::Mat::zeros(m_proto_h, m_proto_w, CV_32F);

        for (int h = y_min; h <= y_max; ++h) {
            float* p_mask = inst_mask.ptr<float>(h);
            for (int w = x_min; w <= x_max; ++w) {
                int idx_proto = h * m_proto_w * matmul_col + w * matmul_col;

                float acc = 0.f;
                for (int j = 0; j < matmul_col; ++j) {
                    acc += coeff[j] * proto[idx_proto + j];
                }

                float val = 1.f / (1.f + std::exp(-acc));  // sigmoid
                p_mask[w] = val;
            }
        }

        m_instance_masks[i] = inst_mask;
    }
}

/*
        Decoding and masking with conf threshold
*/
void mobilint::post::YOLOAnchorSegPostProcessor::decode_conf_thres(
    const std::vector<float>& npu_out, const std::vector<int>& grid, int stride,
    const std::vector<std::vector<double>>& anchor,
    std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<std::pair<float, int>>& pred_scores, std::vector<int>& pred_label,
    std::vector<std::vector<float>>& pred_extra) {
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
                std::vector<float> pred_extra_values;
                int extra_idx = idx + 5 + m_nc;
                for (int k = 0; k < m_nextra; k++) {
                    pred_extra_values.push_back(npu_out[extra_idx + k]);
                }
                for (int k = 0; k < m_nc; k++) {
                    float cls_score = sigmoid(npu_out[idx + 5 + k]);
                    if (conf_value * cls_score > m_conf_thres) {
#pragma omp critical
                        {
                            pred_label.push_back(k);
                            pred_boxes.push_back(pred_box);
                            pred_scores.push_back(std::make_pair(conf_value * cls_score,
                                                                 pred_scores.size()));
                            pred_extra.push_back(pred_extra_values);
                        }
                    }
                }
            }
        }
    }
}

void mobilint::post::YOLOAnchorSegPostProcessor::plot_results(
    cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores, const std::vector<int>& labels,
    const std::vector<std::vector<float>>& extras) {
    plot_boxes(im, boxes, scores, labels);
    plot_masks(im, boxes, labels);
}

/*
        Plot masks on image
*/
void mobilint::post::YOLOAnchorSegPostProcessor::plot_masks(
    cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
    const std::vector<int>& labels) {
    float ratio, xpad, ypad;
    compute_ratio_pads(im, m_imw, m_imh, ratio, xpad, ypad);

    cv::Mat colored_masks(m_imh, m_imw, CV_8UC3);
    colored_masks.setTo(cv::Scalar(0, 0, 0));  // refresh image

    for (int i = 0; i < boxes.size(); i++) {
        cv::Mat mask_up = interpolate(m_instance_masks[i], cv::Size(m_imw, m_imh), 1);
        int cls = labels[i];

        int x_min = std::max(int(boxes[i][0]), 0);
        int y_min = std::max(int(boxes[i][1]), 0);
        int x_max = std::min(int(boxes[i][2]), (int)m_imw - 1);
        int y_max = std::min(int(boxes[i][3]), (int)m_imh - 1);

        std::array<int, 3> bgr = COLORS[cls];

        for (int h = int(y_min); h <= y_max; h++) {
            const float* p_mask = mask_up.ptr<float>(h);
            cv::Vec3b* p_color = colored_masks.ptr<cv::Vec3b>(h);
            for (int w = int(x_min); w <= x_max; w++) {
                if (p_mask[w] > 0.5f) {
                    p_color[w][0] = (uint8_t)bgr[0];
                    p_color[w][1] = (uint8_t)bgr[1];
                    p_color[w][2] = (uint8_t)bgr[2];
                }
            }
        }
    }

    colored_masks = unpad_yolov8_seg(colored_masks, xpad, ypad);
    colored_masks = interpolate(colored_masks, im.size(), 1);
    cv::addWeighted(im, 0.9, colored_masks, 0.4, 0.0, im);
}

/*
    npu_outs shape:
    [20, 20, 351]
    [40, 40, 351]
    [80, 80, 351]
    [160, 160, 32]
*/
void mobilint::post::YOLOAnchorSegPostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() != m_nl + 1)
        throw std::invalid_argument(
            "Size of model outputs does not match "
            "number of detection layers, expected " +
            std::to_string(m_nl + 1) + " but received " +
            std::to_string(npu_outs.size()));

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<int> pred_labels;
    std::vector<std::vector<float>> pred_extra;

    for (int i = 0; i < m_nl; i++) {
        decode_conf_thres(npu_outs[m_nl - i - 1], m_grids[i], m_strides[i], m_anchors[i],
                          pred_boxes, pred_scores, pred_labels, pred_extra);
    }
    xywh2xyxy(pred_boxes);
    nms(pred_boxes, pred_scores, pred_labels, pred_extra, final_boxes, final_scores,
        final_labels, final_extra);
    process_mask(npu_outs[3], final_extra, final_boxes, final_labels);
    double end = set_timer();
}

void mobilint::post::YOLOAnchorSegPostProcessor::worker() {
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
