#include "ssd_post.h"

#include "commons.h"

using namespace mobilint::post;

mobilint::post::SSDPostProcessor::SSDPostProcessor(int imh, int imw,
                                                   const ModelInfo& cfg) {
    const auto& post_info = cfg.m_postprocess;
    int nc = post_info.num_classes;
    int nl = post_info.num_layers;
    float conf_thres = post_info.conf_thres;
    float iou_thres = post_info.iou_thres;

    set_params(nc, imh, imw, conf_thres, iou_thres, nl);
}

mobilint::post::SSDPostProcessor::SSDPostProcessor(int nc, int imh, int imw) {
    set_params(nc, imh, imw);
}

mobilint::post::SSDPostProcessor::SSDPostProcessor(int nc, int imh, int imw,
                                                   float conf_thres, float iou_thres,
                                                   int nl, int max_num_threads) {
    set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
}

void mobilint::post::SSDPostProcessor::set_params(int nc, int imh, int imw,
                                                  float conf_thres, float iou_thres,
                                                  int nl, int max_num_threads) {
    PostProcessor::set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
    mType = PostType::BASE;
    m_feat_sizes = {{19, 19}, {10, 10}, {5, 5}, {3, 3}, {2, 2}, {1, 1}};
    m_coder_weights = {0.1, 0.1, 0.2, 0.2};

    int num_boxes = 1917;
    std::vector<std::vector<float>> aspect_ratios = {
        {2.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}};
    prior_generation(m_feat_sizes, aspect_ratios, num_boxes);
}

/*
        Prior Generation.
*/
void mobilint::post::SSDPostProcessor::prior_generation(
    const std::vector<std::pair<int, int>>& feat_sizes,
    const std::vector<std::vector<float>>& aspect_ratios, int num_boxes, float min_scale,
    float max_scale) {
    double start_priors = std::chrono::duration_cast<std::chrono::duration<double>>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();

    std::vector<float> scales(m_nl + 1);
    for (int i = 0; i < m_nl; i++) {
        scales[i] = min_scale + (max_scale - min_scale) * i / (m_nl - 1);
    }
    scales[m_nl] = 1.0f;

    std::vector<float> priors;
    for (int i = 0; i < m_nl; i++) {
        float sk1 = scales[i];
        float sk2 = scales[i + 1];
        float sk3 = sqrt(sk1 * sk2);

        std::vector<std::pair<float, float>> all_sizes;
        if (i == 0)
            all_sizes.emplace_back(std::make_pair(0.1f, 0.1f));
        else
            all_sizes.emplace_back(std::make_pair(sk1, sk1));

        for (float ar : aspect_ratios[i]) {
            if (ar != -1) {
                float w = sk1 * std::sqrt(ar);
                float h = sk1 / std::sqrt(ar);
                all_sizes.emplace_back(w, h);
                all_sizes.emplace_back(h, w);
            }
        }

        if (i != 0) all_sizes.emplace_back(sk3, sk3);

        int range = (int)all_sizes.size();
        int h_size = feat_sizes[i].first;
        int w_size = feat_sizes[i].second;

        for (int k = 0; k < h_size; k++) {
            for (int l = 0; l < w_size; l++) {
                for (int j = 0; j < range; j++) {
                    float cx = (l + 0.5) / w_size;
                    float cy = (k + 0.5) / h_size;
                    float w = all_sizes[j].first;
                    float h = all_sizes[j].second;

                    priors.push_back(cx);
                    priors.push_back(cy);
                    priors.push_back(w);
                    priors.push_back(h);
                }
            }
        }
    }
    std::vector<float> final_priors(num_boxes * 4);
    for (int i = 0; i < num_boxes * 4; i++) {
        final_priors[i] = priors[i];
    }

    double stop_priors = std::chrono::duration_cast<std::chrono::duration<double>>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
    // cout << "Prior Generation Time: " << stop_priors - start_priors << endl;

    m_priors = final_priors;
}

/*
        Decoding and masking with conf threshold
*/
void mobilint::post::SSDPostProcessor::decode_conf_thres(
    const std::vector<float>& boxes_out, const std::vector<float>& cls_out, int idx,
    int prior_offset, std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<std::pair<float, int>>& pred_scores, std::vector<int>& pred_labels) {
    int h = m_feat_sizes[idx].first;
    int w = m_feat_sizes[idx].second;
    int num_boxes = h * w;
    int n = boxes_out.size() / (num_boxes * 4);
#pragma omp parallel for num_threads(m_max_num_threads) \
    shared(pred_boxes, pred_scores, pred_labels)
    for (int i = 0; i < num_boxes; ++i) {
        for (int k = 0; k < n; k++) {
            int prior_idx = prior_offset + (i * n + k);
            int box_base = i * n * 4 + k * 4;
            int cls_base = i * n * m_nc + k * m_nc;

            bool decoded = false;
            std::array<float, 4> final_box;

            for (int cls = 1; cls < m_nc; ++cls) {
                float conf = cls_out[cls_base + cls];
                if (conf > m_inverse_conf_thres) {
                    if (!decoded) {
                        float dx = boxes_out[box_base + 1];
                        float dy = boxes_out[box_base + 0];
                        float dw = boxes_out[box_base + 3];
                        float dh = boxes_out[box_base + 2];

                        float cx = m_priors[prior_idx * 4 + 0];
                        float cy = m_priors[prior_idx * 4 + 1];
                        float pw = m_priors[prior_idx * 4 + 2];
                        float ph = m_priors[prior_idx * 4 + 3];

                        float pred_cx = dx * m_coder_weights[0] * pw + cx;
                        float pred_cy = dy * m_coder_weights[1] * ph + cy;
                        float pred_w = std::exp(dw * m_coder_weights[2]) * pw;
                        float pred_h = std::exp(dh * m_coder_weights[3]) * ph;

                        float xmin = pred_cx - pred_w / 2;
                        float ymin = pred_cy - pred_h / 2;
                        float xmax = pred_cx + pred_w / 2;
                        float ymax = pred_cy + pred_h / 2;

                        final_box = {xmin * m_imw, ymin * m_imh, xmax * m_imw,
                                     ymax * m_imh};
                        // final_box = {xmin, ymin, xmax, ymax};
                        decoded = true;
                    }
#pragma omp critical
                    {
                        pred_boxes.emplace_back(final_box);
                        pred_labels.push_back(cls);
                        pred_scores.emplace_back(sigmoid(conf), pred_scores.size());
                    }
                }
            }
        }
    }
}

/*
    npu_outs shape:
    [19, 19, 273]
    [19, 19, 12]
    [10, 10, 546]
    [10, 10, 24]
    [5, 5, 546]
    [5, 5, 24]
    [3, 3, 546]
    [3, 3, 24]
    [2, 2, 546]
    [2, 2, 24]
    [1, 1, 546]
    [1, 1, 24]
*/
void mobilint::post::SSDPostProcessor::run_postprocess(
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

    // int num_boxes = 0;
    int prior_offset = 0;

    // std::vector<std::vector<float>> aspect_ratios = {
    //     {2.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}};

    // for (int i = 0; i < m_nl; i++) {
    //     num_boxes += npu_outs[2 * i + 1].size() / 4;
    // }
    // prior_generation(m_feat_sizes, aspect_ratios, num_boxes);

    for (int i = 0; i < m_nl; i++) {
        decode_conf_thres(npu_outs[2 * i + 1], npu_outs[2 * i], i, prior_offset,
                          pred_boxes, pred_scores, pred_labels);
        prior_offset += npu_outs[2 * i + 1].size() / 4;
    }
    nms(pred_boxes, pred_scores, pred_labels, final_boxes, final_scores, final_labels);
    double end = set_timer();
}

void mobilint::post::SSDPostProcessor::plot_boxes(
    cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores, const std::vector<int>& labels) {
    int h = im.size().height;
    int w = im.size().width;

    cv::Rect rect;
    for (int i = 0; i < boxes.size(); i++) {
        int xmin = (int)boxes[i][0] * w / m_imw;
        int ymin = (int)boxes[i][1] * h / m_imh;
        int xmax = (int)boxes[i][2] * w / m_imw;
        int ymax = (int)boxes[i][3] * h / m_imh;

        // clip the box
        xmin = std::max(xmin, 0);
        ymin = std::max(ymin, 0);
        xmax = std::min(xmax, im.cols);
        ymax = std::min(ymax, im.rows);

        rect.x = xmin;
        rect.y = ymin;
        rect.width = xmax - xmin;
        rect.height = ymax - ymin;

        std::array<int, 3> bgr = COLORS[labels[i] % 20];
        cv::Scalar clr(bgr[0], bgr[1], bgr[2]);
        cv::rectangle(im, rect, clr, 1);

        double font_scale = std::min(std::max(rect.width / 500.0, 0.35), 0.99);
        std::string desc =
            DET_LABELS[labels[i]] + " " + std::to_string((int)(scores[i] * 100)) + "%";
        cv::putText(im, desc, cv::Point(xmin, ymin - 10), cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, clr, 1, false);
    }
}

void mobilint::post::SSDPostProcessor::rescale_boxes(
    int org_h, int org_w, std::vector<std::array<float, 4>>& boxes,
    std::vector<int>& labels) {
    for (int i = 0; i < boxes.size(); i++) {
        float xmin = boxes[i][0] * org_w / m_imw;
        float ymin = boxes[i][1] * org_h / m_imh;
        float xmax = boxes[i][2] * org_w / m_imw;
        float ymax = boxes[i][3] * org_h / m_imh;

        // clip the box
        xmin = std::max(xmin, 0.f);
        ymin = std::max(ymin, 0.f);
        xmax = std::min(xmax, (float)org_w);
        ymax = std::min(ymax, (float)org_h);

        float w = xmax - xmin;
        float h = ymax - ymin;

        boxes[i] = {xmin, ymin, w, h};
    }
}

void mobilint::post::SSDPostProcessor::worker() {
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
