#include "cls_post.h"

#include <chrono>

#include "commons.h"
using namespace std::chrono;

using namespace mobilint::post;

mobilint::post::CLSPostProcessor::CLSPostProcessor(int imh, int imw,
                                                   const ModelInfo& cfg) {
    const auto& post_info = cfg.m_postprocess;
    int nc = post_info.num_classes;
    set_params(nc, imh, imw);
}

mobilint::post::CLSPostProcessor::CLSPostProcessor(int nc, int imh, int imw,
                                                   int max_num_threads) {
    set_params(nc, imh, imw, 0.0, 0.0, 0, max_num_threads);
}

void mobilint::post::CLSPostProcessor::set_params(int nc, int imh, int imw,
                                                  float conf_thres, float iou_thres,
                                                  int nl, int max_num_threads) {
    PostProcessor::set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
    mType = PostType::BASE;
    load_labels();
}

void CLSPostProcessor::load_labels(std::string label_file) {
    std::ifstream infile(label_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open label file: " << label_file << std::endl;
        return;
    }
    m_labels.clear();
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty()) {
            m_labels.push_back(line);
        }
    }
}

void mobilint::post::CLSPostProcessor::decode_conf_thres(
    const std::vector<float>& npu_out, std::vector<float>& pred_scores,
    std::vector<int>& pred_label) {
    pred_scores.clear();
    pred_label.clear();
    if (npu_out.empty()) return;

    std::vector<float> probs = npu_out;
    const auto max_it = std::max_element(probs.begin(), probs.end());
    const float max_v = (max_it != probs.end()) ? *max_it : 0.0f;
    for (auto& v : probs) v -= max_v;
    softmax_inplace(probs);

    std::vector<int> idx(m_nc);
    std::iota(idx.begin(), idx.end(), 0);

    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });

    pred_scores.reserve(m_nc);
    pred_label.reserve(m_nc);
    for (int i = 0; i < m_nc; ++i) {
        int c = idx[i];
        pred_label.push_back(c);
        pred_scores.push_back(probs[c]);
    }
}

void mobilint::post::CLSPostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    // if (npu_outs.size() != m_nc)
    //     throw std::invalid_argument(
    //         "Size of model outputs does not match "
    //         "number of classes, expected " +
    //         std::to_string(m_nc) + " but received " +
    //         std::to_string(npu_outs.size()));

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    decode_conf_thres(npu_outs[0], final_scores, final_labels);
    double end = set_timer();
}

void mobilint::post::CLSPostProcessor::plot_results(cv::Mat& im,
                                                    const std::vector<float>& scores,
                                                    const std::vector<int>& labels) {
    if (im.empty() || scores.empty() || labels.empty()) return;
    const int n = std::min<int>({m_topk, (int)scores.size(), (int)labels.size()});
    if (n <= 0) return;

    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.4;
    const int thickness = 1;
    const int pad = 3;
    const int gap = 2;
    const cv::Scalar fg(255, 255, 255);
    const cv::Scalar bg(0, 0, 0);
    const int start_x = 8;
    int y = 8;

    for (int i = 0; i < n; ++i) {
        const int cls = labels[i];
        std::string name;
        if (cls >= 0 && cls < (int)m_labels.size()) {
            name = m_labels[cls];
        } else {
            name = "class_" + std::to_string(cls);
        }

        std::ostringstream oss;
        oss << name << " : " << std::fixed << std::setprecision(3) << scores[i];
        const std::string line = oss.str();

        int baseline = 0;
        const cv::Size text_size =
            cv::getTextSize(line, font_face, font_scale, thickness, &baseline);

        const int box_h = text_size.height + baseline + pad * 2;
        if (y + box_h > im.rows) break;

        const cv::Point p1(start_x, y);
        const cv::Point p2(start_x + text_size.width + pad * 2, y + box_h);
        cv::rectangle(im, p1, p2, bg, cv::FILLED);

        const int text_x = start_x + pad;
        const int text_y = y + pad + text_size.height;
        cv::putText(im, line, cv::Point(text_x, text_y), font_face, font_scale, fg,
                    thickness, cv::LINE_AA);

        y += box_h + gap;
    }
}

void mobilint::post::CLSPostProcessor::worker() {
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
        // k.boxes = get_result_box();
        k.scores = get_result_score();
        k.labels = get_result_label();
        // k.extras = get_result_extra();

        auto end = set_timer();
        auto elapsed = std::to_string(end - start);

        print(title + "Postprocessing time: " + elapsed);

        std::unique_lock<std::mutex> lk2(mMutexOut);
        mOut.push_back(k.id);
        lk2.unlock();

        std::unique_lock<std::mutex> lk_(mMutexOut);  // JUST IN CASE
        mCondOut.notify_all();
        lk_.unlock();  // JUST IN CASE
    }
    print(title + "Finish");
}
