#include "yolo_anchorless_seg_post.h"

#include "commons.h"

using namespace mobilint::post;

mobilint::post::YOLOAnchorlessSegPostProcessor::YOLOAnchorlessSegPostProcessor(
    int imh, int imw, const ModelInfo& cfg) {
    const auto& post_info = cfg.m_postprocess;
    int nc = post_info.num_classes;
    int nl = post_info.num_layers;
    float conf_thres = post_info.conf_thres;
    float iou_thres = post_info.iou_thres;

    set_params(nc, imh, imw, conf_thres, iou_thres, nl);
}

mobilint::post::YOLOAnchorlessSegPostProcessor::YOLOAnchorlessSegPostProcessor(int nc,
                                                                               int imh,
                                                                               int imw) {
    set_params(nc, imh, imw);
}

mobilint::post::YOLOAnchorlessSegPostProcessor::YOLOAnchorlessSegPostProcessor(
    int nc, int imh, int imw, float conf_thres, float iou_thres, int nl,
    int max_num_threads) {
    set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
}

void mobilint::post::YOLOAnchorlessSegPostProcessor::set_params(int nc, int imh, int imw,
                                                                float conf_thres,
                                                                float iou_thres, int nl,
                                                                int max_num_threads) {
    PostProcessor::set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
    mType = PostType::SEG;
    m_proto_stride = 4;
    m_proto_h = m_imh / m_proto_stride;
    m_proto_w = m_imw / m_proto_stride;
    m_nextra = 32;

    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);
}

std::vector<std::vector<int>>
mobilint::post::YOLOAnchorlessSegPostProcessor::generate_grids(int imh, int imw,
                                                               std::vector<int> strides) {
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

std::vector<int> mobilint::post::YOLOAnchorlessSegPostProcessor::generate_strides(
    int nl) {
    std::vector<int> strides;
    if (nl == 2) {
        return {16, 32};
    }

    for (int i = 0; i < nl; ++i) {
        strides.push_back(1 << (3 + i));  // 8,16,32,64...
    }
    return strides;
}

std::vector<std::array<float, 4>>
mobilint::post::YOLOAnchorlessSegPostProcessor::downsample_boxes(
    const std::vector<std::array<float, 4>>& boxes) {
    std::vector<std::array<float, 4>> out;
    out.reserve(boxes.size());
    for (auto b : boxes) {
        for (int j = 0; j < 4; ++j) b[j] /= m_proto_stride;
        out.push_back(b);
    }
    return out;
}

/*
        Access elements in output related to box coordinates and decode them
*/
void mobilint::post::YOLOAnchorlessSegPostProcessor::decode_boxes(
    const std::vector<float>& npu_out_box, const std::vector<int>& grid, int stride,
    int idx, std::array<float, 4>& pred_box) {
    std::array<float, 4> box;
    for (int j = 0; j < 4; j++) {
        const float* base = &npu_out_box[idx * (4 * m_reg_max) + j * m_reg_max];
        std::array<float, 16> tmp;
        for (int k = 0; k < m_reg_max; k++) tmp[k] = base[k];
        softmax_inplace(tmp);

        float box_value = 0;
        for (int k = 0; k < m_reg_max; k++) box_value += tmp[k] * k;
        box[j] = box_value;
    }

    float xmin = grid[idx * 2 + 0] - box[0] + 0.5;
    float ymin = grid[idx * 2 + 1] - box[1] + 0.5;
    float xmax = grid[idx * 2 + 0] + box[2] + 0.5;
    float ymax = grid[idx * 2 + 1] + box[3] + 0.5;

    pred_box = {xmin * stride, ymin * stride, xmax * stride, ymax * stride};
}

void mobilint::post::YOLOAnchorlessSegPostProcessor::decode_extra(
    const std::vector<float>& npu_out_extra, const std::vector<int>& grid, int stride,
    int idx, std::vector<float>& pred_extra) {
    pred_extra.resize(32);
    const float* src = &npu_out_extra[idx * 32];
    std::memcpy(pred_extra.data(), src, 32 * sizeof(float));
}

/*
        Compute masks
*/
void mobilint::post::YOLOAnchorlessSegPostProcessor::process_mask(
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
void mobilint::post::YOLOAnchorlessSegPostProcessor::decode_conf_thres(
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
            if (npu_out_cls[i * m_nc + j] > m_inverse_conf_thres) {
                float conf = sigmoid(npu_out_cls[i * m_nc + j]);
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
    [20, 20, 32]
    [20, 20, 64]
    [20, 20, 80]
    [40, 40, 32]
    [40, 40, 64]
    [40, 40, 80]
    [80, 80, 32]
    [80, 80, 64]
    [80, 80, 80]
    [160, 160, 32]
*/

void mobilint::post::YOLOAnchorlessSegPostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() != m_nl * 3 + 1)
        throw std::invalid_argument(
            "Size of model outputs does not match "
            "number of detection layers, expected " +
            std::to_string(m_nl * 3 + 1) + " but received " +
            std::to_string(npu_outs.size()));

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<int> pred_labels;
    std::vector<std::vector<float>> pred_extra;
    std::vector<std::array<int, 3>> indices = {{8, 7, 6}, {5, 4, 3}, {2, 1, 0}};

    for (int i = 0; i < m_nl; i++) {
        auto [cls_idx, box_idx, extra_idx] = indices[i];
        decode_conf_thres(npu_outs[box_idx], npu_outs[cls_idx], npu_outs[extra_idx],
                          m_grids[i], m_strides[i], pred_boxes, pred_scores, pred_labels,
                          pred_extra);
    }

    nms(pred_boxes, pred_scores, pred_labels, pred_extra, final_boxes, final_scores,
        final_labels, final_extra);
    process_mask(npu_outs[9], final_extra, final_boxes, final_labels);
    double end = set_timer();
}

void mobilint::post::YOLOAnchorlessSegPostProcessor::get_results(
    int org_h, int org_w, std::vector<std::array<float, 4>>& boxes,
    std::vector<int>& labels, std::vector<std::vector<unsigned int>>& seg_results) {
    float ratio, xpad, ypad;
    compute_ratio_pads(org_w, org_h, m_imw, m_imh, ratio, xpad, ypad);

    seg_results.clear();

    for (int i = 0; i < boxes.size(); ++i) {
        float xmin = (boxes[i][0] - xpad) / ratio;
        float ymin = (boxes[i][1] - ypad) / ratio;
        float xmax = (boxes[i][2] - xpad) / ratio;
        float ymax = (boxes[i][3] - ypad) / ratio;

        // clip the box
        xmin = std::max(xmin, 0.f);
        ymin = std::max(ymin, 0.f);
        xmax = std::min(xmax, (float)org_w);
        ymax = std::min(ymax, (float)org_h);

        float w = xmax - xmin;
        float h = ymax - ymin;

        boxes[i] = {xmin, ymin, w, h};

        cv::Mat mask_up = interpolate(m_instance_masks[i], cv::Size(m_imw, m_imh), 1);
        mask_up = unpad_yolov8_seg(mask_up, xpad, ypad);
        mask_up = interpolate(mask_up, cv::Size(org_w, org_h), 1);

        // binarization
        cv::Mat bin_mask = cv::Mat::zeros(org_h, org_w, CV_8UC1);
        for (int y = ymin; y < ymax; ++y) {
            const float* mrow = mask_up.ptr<float>(y);
            uchar* brow = bin_mask.ptr<uchar>(y);
            for (int x = xmin; x < xmax; ++x) {
                if (mrow[x] > 0.5f) {
                    brow[x] = 1;
                }
            }
        }

        std::vector<unsigned int> rle;
        int count = 0;
        uchar prev = 0;

        for (int col = 0; col < org_w; ++col) {
            for (int row = 0; row < org_h; ++row) {
                uchar px = bin_mask.at<uchar>(row, col);
                if (px == prev) {
                    ++count;
                } else {
                    rle.push_back(count);
                    count = 1;
                    prev = px;
                }
            }
        }
        rle.push_back(count);

        seg_results.push_back(rle);
        int yolo_label = labels[i];
        if (yolo_label >= 0 && yolo_label < YOLO_TO_COCO.size()) {
            labels[i] = YOLO_TO_COCO[yolo_label];
        }
    }
}

void mobilint::post::YOLOAnchorlessSegPostProcessor::plot_results(
    cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores, const std::vector<int>& labels,
    const std::vector<std::vector<float>>& extras) {
    plot_boxes(im, boxes, scores, labels);
    plot_masks(im, boxes, labels);
}

/*
        Plot masks on image
*/
void mobilint::post::YOLOAnchorlessSegPostProcessor::plot_masks(
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

void mobilint::post::YOLOAnchorlessSegPostProcessor::worker() {
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
