#include "post_yolov8_pose.h"

mobilint::post::YOLOv8PosePostProcessor::YOLOv8PosePostProcessor()
    : YOLOv8PostProcessor() {
    m_nc = 1;       // number of classes
    m_nextra = 51;  // number of extra outputs (keypoints => 17 * 3)
    mType = PostType::POSE;

    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);
}

mobilint::post::YOLOv8PosePostProcessor::YOLOv8PosePostProcessor(int nc, int imh, int imw,
                                                                 float conf_thres,
                                                                 float iou_thres,
                                                                 bool verbose)
    : YOLOv8PostProcessor(nc, imh, imw, conf_thres, iou_thres, verbose) {
    m_nextra = 51;  // number of extra outputs (keypoints => 17 * 3)
    mType = PostType::POSE;
}

/*
        Access elements in output related to keypoints and decode them
*/
void mobilint::post::YOLOv8PosePostProcessor::decode_extra(
    const std::vector<float>& extra, const std::vector<int>& grid, int stride, int idx,
    std::vector<float>& pred_extra) {
    pred_extra.clear();
    int num_kpts = m_nextra / 3;  // 51 / 3
    for (int i = 0; i < num_kpts; i++) {
        auto first = extra[idx * m_nextra + 3 * i + 0];
        auto second = extra[idx * m_nextra + 3 * i + 1];
        auto third = extra[idx * m_nextra + 3 * i + 2];

        first = (first * 2 + grid[idx * 2 + 0]) * stride;
        second = (second * 2 + grid[idx * 2 + 1]) * stride;
        third = sigmoid(third);

        pred_extra.push_back(first);
        pred_extra.push_back(second);
        pred_extra.push_back(third);
    }
}

void mobilint::post::YOLOv8PosePostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() / 3 != m_nl) {  // 3 keypoints + 3 dets
        throw std::invalid_argument(
            "YOLOv8 Pose post-processing is "
            "expected to receive 9 NPU outputs, however received " +
            std::to_string(npu_outs.size()));
    }

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<float> pred_conf;
    std::vector<int> pred_label;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<std::vector<float>> pred_extra;

    std::vector<int> order = {6, 8, 5, 4, 7, 3, 1, 2, 0};

    for (int i = 0; i < m_nl; i++) {
        decode_conf_thres(npu_outs[order[3 * i]], npu_outs[order[3 * i + 1]], m_grids[i],
                          m_strides[i], pred_boxes, pred_conf, pred_label, pred_scores,
                          npu_outs[order[3 * i + 2]], pred_extra);
    }

    xywh2xyxy(pred_boxes);

    nms(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra, final_boxes,
        final_scores, final_labels, final_extra);

    double endd = set_timer();
    if (m_verbose) std::cout << "Real C++ Time        : " << endd - start << std::endl;
}

/*
        Draw human keypoints
*/
void mobilint::post::YOLOv8PosePostProcessor::plot_keypoints(
    cv::Mat& im, std::vector<std::vector<float>>& kpts) {
    int radius = 5;               // circle size
    int steps = 3;                // (x, y, conf) * 17
    int num_kpts = m_nextra / 3;  // 51 / 3
    float kpts_conf_thres = 0.4;  // Do not draw low confidence skeleton

    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

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

/*
        Plot extras, in this case plot keypoints
*/
void mobilint::post::YOLOv8PosePostProcessor::plot_extras(
    cv::Mat& im, std::vector<std::vector<float>>& extras) {
    plot_keypoints(im, extras);
}

void mobilint::post::YOLOv8PosePostProcessor::worker() {
    // auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    // auto title = std::to_string(thres_id) + " | Postprocessor Worker | ";

    // print(title + "Start");
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

        // plot_boxes(k.im, k.boxes, k.scores, k.labels);
        plot_extras(k.im, k.extras);

        auto end = set_timer();
        auto elapsed = std::to_string(end - start);

        // print(title + "Postprocessing time: " + elapsed);
        // print(title + "Number of detections " + std::to_string(k.boxes.size()));

        std::unique_lock<std::mutex> lk2(mMutexOut);
        mOut.push_back(k.id);
        lk2.unlock();

        std::unique_lock<std::mutex> lk_(mMutexOut);  // JUST IN CASE
        mCondOut.notify_all();
        lk_.unlock();  // JUST IN CASE
    }
    // print(title + "Finish");
}
