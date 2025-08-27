#include "yolo_anchorless_pose_post.h"

#include "commons.h"

using namespace mobilint::post;

mobilint::post::YOLOAnchorlessPosePostProcessor::YOLOAnchorlessPosePostProcessor(
    int nc, int imh, int imw) {
    setParams(nc, imh, imw);
}

mobilint::post::YOLOAnchorlessPosePostProcessor::YOLOAnchorlessPosePostProcessor(
    int nc, int imh, int imw, float conf_thres, float iou_thres, int max_num_threads) {
    setParams(nc, imh, imw, conf_thres, iou_thres, max_num_threads);
}

mobilint::post::YOLOAnchorlessPosePostProcessor::~YOLOAnchorlessPosePostProcessor() {
    destroyed = true;
    mCondIn.notify_all();
    mCondOut.notify_all();
    if (mThread.joinable()) {
        mThread.join();
    }
}

void mobilint::post::YOLOAnchorlessPosePostProcessor::setParams(int nc, int imh, int imw,
                                                                float conf_thres,
                                                                float iou_thres,
                                                                int max_num_threads) {
    PostProcessor::setParams(nc, imh, imw, conf_thres, iou_thres, max_num_threads);
    mType = PostType::POSE;
    m_nextra = 51;

    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);
}

std::vector<std::vector<int>>
mobilint::post::YOLOAnchorlessPosePostProcessor::generate_grids(
    int imh, int imw, std::vector<int> strides) {
    std::vector<std::vector<int>> all_grids;
    for (int i = 0; i < strides.size(); i++) {
        int grid_h = imh / strides[i];
        int grid_w = imw / strides[i];
        int grid_size = grid_h * grid_w * 2;

        std::vector<int> grids;
        for (int j = 0; j < grid_size; j++) {
            if (j % 2 == 0) {
                grids.push_back(((int)j / 2) % grid_w);
            } else {
                grids.push_back(((int)j / 2) / grid_w);
            }
        }

        all_grids.push_back(grids);
    }
    return all_grids;
}

std::vector<int> mobilint::post::YOLOAnchorlessPosePostProcessor::generate_strides(
    int nl) {
    std::vector<int> strides;
    for (int i = 0; i < nl; i++) {
        strides.push_back(pow(2, 3 + i));
    }
    return strides;
}

/*
        Access elements in output related to box coordinates and decode them
*/
void mobilint::post::YOLOAnchorlessPosePostProcessor::decode_boxes(
    const std::vector<float>& npu_out_box, const std::vector<int>& grid, int stride,
    int idx, std::array<float, 4>& pred_box) {
    std::array<float, 4> box;
    std::array<float, 16> tmp;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < m_reg_max; k++)
            tmp[k] = npu_out_box[idx * (4 * m_reg_max) + j * m_reg_max + k];
        softmax_inplace(tmp);

        float box_value = 0;
        for (int k = 0; k < m_reg_max; k++) box_value += tmp[k] * k;
        box[j] = box_value;
    }

    float xmin = grid[idx * 2 + 0] - box[0] + 0.5;
    float ymin = grid[idx * 2 + 1] - box[1] + 0.5;
    float xmax = grid[idx * 2 + 0] + box[2] + 0.5;
    float ymax = grid[idx * 2 + 1] + box[3] + 0.5;

    // float x = (xmin + xmax) / 2 * stride;
    // float y = (ymin + ymax) / 2 * stride;
    // float w = (xmax - xmin) * stride;
    // float h = (ymax - ymin) * stride;

    pred_box = {xmin * stride, ymin * stride, xmax * stride, ymax * stride};
}

void mobilint::post::YOLOAnchorlessPosePostProcessor::decode_extra(
    const std::vector<float>& npu_out_extra, const std::vector<int>& grid, int stride,
    int idx, std::vector<float>& pred_extra) {
    pred_extra.clear();
    int num_kpts = m_nextra / 3;  // 51 / 3
    for (int i = 0; i < num_kpts; i++) {
        auto first = npu_out_extra[idx * m_nextra + 3 * i + 0];
        auto second = npu_out_extra[idx * m_nextra + 3 * i + 1];
        auto third = npu_out_extra[idx * m_nextra + 3 * i + 2];

        first = (first * 2 + grid[idx * 2 + 0]) * stride;
        second = (second * 2 + grid[idx * 2 + 1]) * stride;
        third = sigmoid(third);

        pred_extra.push_back(first);
        pred_extra.push_back(second);
        pred_extra.push_back(third);
    }
}

/*
        Decoding with conf threshold
*/
void mobilint::post::YOLOAnchorlessPosePostProcessor::decode_conf_thres(
    const std::vector<float>& npu_out_box, const std::vector<float>& npu_out_cls,
    const std::vector<float>& npu_out_extra, const std::vector<int>& grid, int stride,
    std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<std::pair<float, int>>& pred_scores, std::vector<int>& pred_label,
    std::vector<std::vector<float>>& pred_extra) {
    int grid_h = m_imh / stride;
    int grid_w = m_imw / stride;

#pragma omp parallel for num_threads(m_max_num_threads) \
    shared(pred_boxes, pred_label, pred_scores, pred_extra)
    for (int i = 0; i < grid_h * grid_w; i++) {
        std::array<float, 4> pred_box = {-999, -999, -999, -999};
        std::vector<float> pred_extra_values;
        for (int j = 0; j < m_nc; j++) {
            float conf = npu_out_cls[i * m_nc + j];
            if (conf > m_conf_thres) {
                if (pred_box[0] == -999) {  // decode box only once
                    decode_boxes(npu_out_box, grid, stride, i, pred_box);
                    decode_extra(npu_out_extra, grid, stride, i, pred_extra_values);
                }

#pragma omp critical
                {
                    pred_label.push_back(j);
                    pred_boxes.push_back(pred_box);
                    pred_scores.push_back(std::make_pair(conf, pred_scores.size()));
                    pred_extra.push_back(pred_extra_values);
                }
            }
        }
    }
}

/*
    npu_outs shape:
    [20, 20, 1]
    [20, 20, 51]
    [20, 20, 64]
    [40, 40, 1]
    [40, 40, 51]
    [40, 40, 64]
    [80, 80, 1]
    [80, 80, 51]
    [80, 80, 64]
*/

void mobilint::post::YOLOAnchorlessPosePostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() != m_nl * 3)
        throw std::invalid_argument(
            "Size of model outputs does not match "
            "number of detection layers, expected " +
            std::to_string(m_nl * 3) + " but received " +
            std::to_string(npu_outs.size()));

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<int> pred_labels;
    std::vector<std::vector<float>> pred_extra;
    std::vector<std::array<int, 3>> indices = {{6, 8, 7}, {3, 5, 4}, {0, 2, 1}};

    for (int i = 0; i < m_nl; i++) {
        auto [cls_idx, box_idx, extra_idx] = indices[i];
        decode_conf_thres(npu_outs[box_idx], npu_outs[cls_idx], npu_outs[extra_idx],
                          m_grids[i], m_strides[i], pred_boxes, pred_scores, pred_labels,
                          pred_extra);
    }

    // xywh2xyxy(pred_boxes);
    nms(pred_boxes, pred_scores, pred_labels, pred_extra, final_boxes, final_scores,
        final_labels, final_extra);

    double end = set_timer();
}

/*
        Draw human keypoints
*/
void mobilint::post::YOLOAnchorlessPosePostProcessor::plot_keypoints(
    cv::Mat& im, std::vector<std::vector<float>>& kpts) {
    int radius = 5;               // circle size
    int steps = 3;                // (x, y, conf) * 17
    int num_kpts = m_nextra / 3;  // 51 / 3
    float kpts_conf_thres = 0.4;  // Do not draw low confidence skeleton

    float ratio, xpad, ypad;
    compute_ratio_pads(im, m_imw, m_imh, ratio, xpad, ypad);

    for (int i = 0; i < kpts.size(); i++) {
        for (int j = 0; j < num_kpts; j++) {
            kpts[i][3 * j + 0] = (kpts[i][3 * j + 0] - xpad) / ratio;
            kpts[i][3 * j + 1] = (kpts[i][3 * j + 1] - ypad) / ratio;
        }
    }

    for (const auto& kpt_t : kpts) {
        for (int j = 0; j < num_kpts; j++) {
            auto bgr = m_pose_kpt_color[j];
            cv::Scalar color(bgr[0], bgr[1], bgr[2]);
            int kpt_idx = steps * j;
            int x_coord = (int)kpt_t[kpt_idx];
            int y_coord = (int)kpt_t[kpt_idx + 1];
            float conf = kpt_t[kpt_idx + 2];

            if (conf < kpts_conf_thres) {
                continue;
            }

            if (x_coord % m_imw != 0 && y_coord % m_imh != 0) {
                cv::Point p(x_coord, y_coord);
                cv::circle(im, p, radius, color, -1);
            }
        }

        for (int j = 0; j < m_skeleton.size(); j++) {
            auto bgr = m_pose_limb_color[j];
            cv::Scalar color(bgr[0], bgr[1], bgr[2]);
            const auto& sk = m_skeleton[j];

            float conf1 = kpt_t[(sk[0] - 1) * steps + 2];
            float conf2 = kpt_t[(sk[1] - 1) * steps + 2];
            if (conf1 < 0.5 || conf2 < 0.5) {
                continue;
            }

            cv::Point p1((int)kpt_t[(sk[0] - 1) * steps],
                         (int)kpt_t[(sk[0] - 1) * steps + 1]);
            cv::Point p2((int)kpt_t[(sk[1] - 1) * steps],
                         (int)kpt_t[(sk[1] - 1) * steps + 1]);

            if (p1.x % m_imw == 0 || p1.y % m_imh == 0 || p1.x < 0 || p1.y < 0) {
                continue;
            }

            if (p2.x % m_imw == 0 || p2.y % m_imh == 0 || p2.x < 0 || p2.y < 0) {
                continue;
            }
            cv::line(im, p1, p2, color, 2);
        }
    }
}

void mobilint::post::YOLOAnchorlessPosePostProcessor::worker() {
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
        k.extras = get_result_extra();

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
