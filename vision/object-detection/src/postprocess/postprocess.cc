#include "postprocess.h"

#include "commons.h"

mobilint::post::PostProcessor::PostProcessor(int imh, int imw, const ModelInfo& cfg) {
    const auto& post_info = cfg.m_postprocess;
    int nc = post_info.num_classes;
    int nl = post_info.num_layers;
    float conf_thres = post_info.conf_thres;
    float iou_thres = post_info.iou_thres;

    set_params(nc, imh, imw, conf_thres, iou_thres, nl);
}

mobilint::post::PostProcessor::PostProcessor(int nc, int imh, int imw) {
    set_params(nc, imh, imw);
}

mobilint::post::PostProcessor::PostProcessor(int nc, int imh, int imw, float conf_thres,
                                             float iou_thres, int nl,
                                             int max_num_threads) {
    set_params(nc, imh, imw, conf_thres, iou_thres, nl, max_num_threads);
}

mobilint::post::PostProcessor::~PostProcessor() noexcept {
    {
        std::lock_guard<std::mutex> lk(mMutexIn);
        destroyed = true;
    }

    mCondIn.notify_all();
    mCondOut.notify_all();

    if (mThread.joinable()) {
        if (std::this_thread::get_id() == mThread.get_id()) {
            mThread.detach();
        } else {
            mThread.join();
        }
    }
}

void mobilint::post::PostProcessor::set_params(int nc, int imh, int imw, float conf_thres,
                                               float iou_thres, int nl,
                                               int max_num_threads) {
    m_max_num_threads = max_num_threads;
    m_nc = nc;                  // number of classes
    m_nl = nl;                  // number of detection layers
    m_nextra = 0;               // number of extra outputs
    m_imh = imh;                // model input image height
    m_imw = imw;                // model input image width
    m_conf_thres = conf_thres;  // confidence threshold, used in decoding
    m_inverse_conf_thres = inverse_sigmoid(m_conf_thres);
    m_iou_thres = iou_thres;  // iou threshold, used in nms
    m_max_det_num = 300;
    mType = PostType::BASE;

    ticket = 0;
    destroyed = false;
    mThread = std::thread(&mobilint::post::PostProcessor::worker, this);
}

int mobilint::post::PostProcessor::get_nl() const { return m_nl; }

int mobilint::post::PostProcessor::get_nc() const { return m_nc; }

int mobilint::post::PostProcessor::get_imh() const { return m_imh; }

int mobilint::post::PostProcessor::get_imw() const { return m_imw; }

/*
                Return the type of Post Processing
*/
mobilint::post::PostType mobilint::post::PostProcessor::getType() const { return mType; }

/*
                Apply NMS
*/
void mobilint::post::PostProcessor::nms(
    const std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<std::pair<float, int>>& pred_scores, const std::vector<int>& pred_labels,
    const std::vector<std::vector<float>>& pred_extra,
    std::vector<std::array<float, 4>>& final_boxes, std::vector<float>& final_scores,
    std::vector<int>& final_labels, std::vector<std::vector<float>>& final_extra) {
    // sort the scores(predicted confidence * predicted class score)
    sort(pred_scores.begin(), pred_scores.end(), std::greater<>());

    for (int i = 0; i < (int)pred_scores.size(); i++) {
        float temp_score = pred_scores[i].first;
        if (pred_scores[i].first != -99) {  // check if the box valid or not
            int idx = pred_scores[i].second;
            const std::array<float, 4>& max_box = pred_boxes[idx];

#pragma omp parallel for num_threads(m_max_num_threads) \
    shared(pred_boxes, pred_labels, pred_scores, pred_extra, i, idx, max_box)
            for (int j = i; j < (int)pred_scores.size(); j++) {
                int temp_idx = pred_scores[j].second;
                const std::array<float, 4>& temp_box = pred_boxes[temp_idx];
                if (pred_labels[idx] == pred_labels[temp_idx]) {
                    float iou = get_iou(max_box, temp_box);
                    if (iou > m_iou_thres) pred_scores[j].first = -99;
                }
            }

            final_boxes.push_back(max_box);
            final_scores.push_back(temp_score);
            final_labels.push_back(pred_labels[idx]);
            final_extra.push_back(pred_extra[idx]);

            if (final_boxes.size() >= m_max_det_num) {
                break;
            }
        }
    }
}

void mobilint::post::PostProcessor::nms(
    const std::vector<std::array<float, 4>>& pred_boxes,
    std::vector<std::pair<float, int>>& pred_scores, const std::vector<int>& pred_labels,
    std::vector<std::array<float, 4>>& final_boxes, std::vector<float>& final_scores,
    std::vector<int>& final_labels) {
    sort(pred_scores.begin(), pred_scores.end(), std::greater<>());

    for (int i = 0; i < (int)pred_scores.size(); i++) {
        float temp_score = pred_scores[i].first;
        if (pred_scores[i].first != -99) {  // check if the box valid or not
            int idx = pred_scores[i].second;
            const std::array<float, 4>& max_box = pred_boxes[idx];

#pragma omp parallel for num_threads(m_max_num_threads) \
    shared(pred_boxes, pred_scores, pred_labels, i, idx, max_box)
            for (int j = i; j < (int)pred_scores.size(); j++) {
                int temp_idx = pred_scores[j].second;
                const std::array<float, 4>& temp_box = pred_boxes[temp_idx];
                if (pred_labels[idx] == pred_labels[temp_idx]) {
                    float iou = get_iou(max_box, temp_box);
                    if (iou > m_iou_thres) pred_scores[j].first = -99;
                }
            }

            final_boxes.push_back(max_box);
            final_scores.push_back(temp_score);
            final_labels.push_back(pred_labels[idx]);

            if (final_boxes.size() >= m_max_det_num) {
                break;
            }
        }
    }
}

/*
                Returns current time, used to measure performance time
*/
double mobilint::post::PostProcessor::set_timer() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::vector<std::array<float, 4>>& mobilint::post::PostProcessor::get_result_box() {
    return final_boxes;
}

std::vector<float>& mobilint::post::PostProcessor::get_result_score() {
    return final_scores;
}

std::vector<int>& mobilint::post::PostProcessor::get_result_label() {
    return final_labels;
}

std::vector<std::vector<float>>& mobilint::post::PostProcessor::get_result_extra() {
    return final_extra;
}

void mobilint::post::PostProcessor::plot_results(
    cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores, const std::vector<int>& labels,
    const std::vector<std::vector<float>>& extras) {
    plot_boxes(im, boxes, scores, labels);
    plot_extras(im, extras);
}
/*
                Draw detected box and write the it's label & score
*/
void mobilint::post::PostProcessor::plot_boxes(
    cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
    const std::vector<float>& scores, const std::vector<int>& labels) {
    float ratio, xpad, ypad;
    compute_ratio_pads(im, m_imw, m_imh, ratio, xpad, ypad);

    cv::Rect rect;
    for (int i = 0; i < boxes.size(); i++) {
        int xmin = (int)(boxes[i][0] - xpad) / ratio;
        int ymin = (int)(boxes[i][1] - ypad) / ratio;
        int xmax = (int)(boxes[i][2] - xpad) / ratio;
        int ymax = (int)(boxes[i][3] - ypad) / ratio;

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

void mobilint::post::PostProcessor::rescale_boxes(
    int org_h, int org_w, std::vector<std::array<float, 4>>& boxes,
    std::vector<int>& labels) {
    float ratio, xpad, ypad;
    compute_ratio_pads(org_w, org_h, m_imw, m_imh, ratio, xpad, ypad);

    for (int i = 0; i < boxes.size(); i++) {
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

        int yolo_label = labels[i];
        if (yolo_label >= 0 && yolo_label < YOLO_TO_COCO.size()) {
            labels[i] = YOLO_TO_COCO[yolo_label];
        }
    }
}

/*
                Plot extra data such as landmarks. keypoins, masks.
                Simple Object Detection does not any extras.
*/
void mobilint::post::PostProcessor::plot_extras(
    cv::Mat& im, const std::vector<std::vector<float>>& extras) {
    // do nothing
    int do_nothing;
}

/*
                Print out given message in DEBUG mode using locks
*/
void mobilint::post::PostProcessor::print(std::string msg) {
#ifdef DEBUG
    std::lock_guard<std::mutex> lk(mPrintMutex);
    std::cout << msg << endl;
#endif
}

uint64_t mobilint::post::PostProcessor::enqueue(std::vector<std::vector<float>>& npu_outs,
                                                std::vector<std::array<float, 4>>& boxes,
                                                std::vector<float>& scores,
                                                std::vector<int>& labels,
                                                std::vector<std::vector<float>>& extras) {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Enqueue | ";

    print(title + "Start");
    uint64_t ticket_save = 0;

    {
        std::lock_guard<std::mutex> lk(mMutexIn);
        mQueueIn.push({++ticket, npu_outs, boxes, scores, labels, extras});
        ticket_save = ticket;

        print(title + "Input Queue size " + std::to_string(mQueueIn.size()));
    }
    mCondIn.notify_all();  // JUST IN CASE

    print(title + "Finish");
    return ticket_save;
}

void mobilint::post::PostProcessor::receive(uint64_t receipt_no) {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Receive | ";

    while (!destroyed) {
        print(title + "Start | Receipt = " + std::to_string(receipt_no));
        std::unique_lock<std::mutex> lk(mMutexOut);

        if (mOut.empty()) {
            mCondOut.wait(lk, [this] { return !(mOut.empty()) || destroyed; });
        }

        print(title + "Received Output Queue of size " + std::to_string(mOut.size()));

        if (destroyed) {
            break;
        }

        for (int i = 0; i < mOut.size(); i++) {
            if (mOut[i] == receipt_no) {
                print(title + "Got my output | Receipt = " + std::to_string(mOut[i]));
                mOut.erase(mOut.begin() + i);
                return;
            }
        }

        lk.unlock();
        print(title + "Finish");
    }
}
