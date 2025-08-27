#include "yolo_anchorless_post.h"

#include <chrono>

#include "commons.h"
using namespace std::chrono;

using namespace mobilint::post;

mobilint::post::YOLOAnchorlessPostProcessor::YOLOAnchorlessPostProcessor(int nc, int imh,
                                                                         int imw) {
    setParams(nc, imh, imw);
}

mobilint::post::YOLOAnchorlessPostProcessor::YOLOAnchorlessPostProcessor(
    int nc, int imh, int imw, float conf_thres, float iou_thres, int max_num_threads) {
    setParams(nc, imh, imw, conf_thres, iou_thres, max_num_threads);
}

mobilint::post::YOLOAnchorlessPostProcessor::~YOLOAnchorlessPostProcessor() {
    destroyed = true;
    mCondIn.notify_all();
    mCondOut.notify_all();
    if (mThread.joinable()) {
        mThread.join();
    }
}

void mobilint::post::YOLOAnchorlessPostProcessor::setParams(int nc, int imh, int imw,
                                                            float conf_thres,
                                                            float iou_thres,
                                                            int max_num_threads) {
    PostProcessor::setParams(nc, imh, imw, conf_thres, iou_thres, max_num_threads);
    mType = PostType::BASE;
    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);
}

std::vector<std::vector<int>> mobilint::post::YOLOAnchorlessPostProcessor::generate_grids(
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

std::vector<int> mobilint::post::YOLOAnchorlessPostProcessor::generate_strides(int nl) {
    std::vector<int> strides;
    for (int i = 0; i < nl; i++) {
        strides.push_back(pow(2, 3 + i));
    }
    return strides;
}

/*
        Access elements in output related to box coordinates and decode them
*/
void mobilint::post::YOLOAnchorlessPostProcessor::decode_boxes(
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

/*
        Decoding and masking with conf threshold
*/
void mobilint::post::YOLOAnchorlessPostProcessor::decode_conf_thres(
    const std::vector<float>& npu_out_box, const std::vector<float>& npu_out_cls,
    const std::vector<int>& grid, int stride,
    std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<std::pair<float, int>>& pred_scores, std::vector<int>& pred_label) {
    int grid_h = m_imh / stride;
    int grid_w = m_imw / stride;

#pragma omp parallel for num_threads(m_max_num_threads) \
    shared(pred_boxes, pred_label, pred_scores)
    for (int i = 0; i < grid_h * grid_w; i++) {
        std::array<float, 4> pred_box = {-999, -999, -999, -999};
        std::vector<float> pred_extra_values;
        for (int j = 0; j < m_nc; j++) {
            float conf = npu_out_cls[i * m_nc + j];
            if (conf > m_conf_thres) {
                if (pred_box[0] == -999) {  // decode box only once
                    decode_boxes(npu_out_box, grid, stride, i, pred_box);
                }

#pragma omp critical
                {
                    pred_label.push_back(j);
                    pred_boxes.push_back(pred_box);
                    pred_scores.push_back(std::make_pair(conf, pred_scores.size()));
                }
            }
        }
    }
}

/*
    npu_outs shape:
    [20, 20, 64]
    [20, 20, 80]
    [40, 40, 64]
    [40, 40, 80]
    [80, 80, 64]
    [80, 80, 80]
*/
void mobilint::post::YOLOAnchorlessPostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() != m_nl * 2)
        throw std::invalid_argument(
            "Size of model outputs does not match "
            "number of detection layers, expected " +
            std::to_string(m_nl * 2) + " but received " +
            std::to_string(npu_outs.size()));

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<int> pred_labels;
    std::vector<std::pair<int, int>> pairs = {{5, 4}, {3, 2}, {1, 0}};

    for (int i = 0; i < m_nl; i++) {
        auto [cls_idx, box_idx] = pairs[i];
        decode_conf_thres(npu_outs[box_idx], npu_outs[cls_idx], m_grids[i], m_strides[i],
                          pred_boxes, pred_scores, pred_labels);
    }
    nms(pred_boxes, pred_scores, pred_labels, final_boxes, final_scores, final_labels);
    double end = set_timer();
}

void mobilint::post::YOLOAnchorlessPostProcessor::worker() {
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
