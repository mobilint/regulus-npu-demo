#include "post_yolov8.h"

using namespace std;

namespace {

/*
        Apply softmax function to the input vector in-place
*/
template <typename FloatContainer>
void softmax_inplace(FloatContainer& con) {
    float sum = 0;
    for (auto v : con) {
        sum += exp(v);
    }
    for (auto& v : con) {
        v = exp(v) / sum;
    }
}

}  // namespace

mobilint::post::YOLOv8PostProcessor::YOLOv8PostProcessor() {
    m_nc = 80;           // number of classes
    m_nl = 3;            // number of detection layers
    m_nextra = 0;        // number of extra outputs
    m_imh = 384;         // model input image height
    m_imw = 576;         // model input image width
    m_conf_thres = 0.5;  // confidence threshold, used in decoding
    // m_inverse_conf_thres = inverse_sigmoid(m_conf_thres);
    m_iou_thres = 0.45;  // iou threshold, used in nms
    m_verbose = false;
    m_max_det_num = 300;
    mType = PostType::BASE;
    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);

    ticket = 0;
    destroyed = false;
    mThread = std::thread(&mobilint::post::YOLOv8PostProcessor::worker, this);
}

mobilint::post::YOLOv8PostProcessor::YOLOv8PostProcessor(int nc, int imh, int imw,
                                                         float conf_thres,
                                                         float iou_thres, bool verbose) {
    m_nc = nc;                  // number of classes
    m_nl = 3;                   // number of detection layers
    m_nextra = 0;               // number of extra outputs
    m_imh = imh;                // model input image height
    m_imw = imw;                // model input image width
    m_conf_thres = conf_thres;  // confidence threshold, used in decoding
    // m_inverse_conf_thres = inverse_sigmoid(m_conf_thres);
    m_iou_thres = iou_thres;  // iou threshold, used in nms
    m_verbose = verbose;
    m_max_det_num = 300;
    mType = PostType::BASE;
    m_strides = generate_strides(m_nl);
    m_grids = generate_grids(m_imh, m_imw, m_strides);

    ticket = 0;
    destroyed = false;
    mThread = std::thread(&mobilint::post::YOLOv8PostProcessor::worker, this);
}

mobilint::post::YOLOv8PostProcessor::~YOLOv8PostProcessor() {
    destroyed = true;
    mCondIn.notify_all();
    mCondOut.notify_all();
    mThread.join();
}

/*
        Generate strides, needed for decoding
*/
std::vector<int> mobilint::post::YOLOv8PostProcessor::generate_strides(int nl) {
    std::vector<int> strides;
    for (int i = 0; i < nl; i++) {
        strides.push_back(pow(2, 3 + i));
    }
    return strides;
}

/*
        Generate grids, needed for decoding
*/
std::vector<std::vector<int>> mobilint::post::YOLOv8PostProcessor::generate_grids(
    int imh, int imw, const std::vector<int>& strides) {
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

int mobilint::post::YOLOv8PostProcessor::get_nl() const { return m_nl; }

int mobilint::post::YOLOv8PostProcessor::get_nc() const { return m_nc; }

/*
        Return the type of Post Processing
*/
mobilint::post::PostType mobilint::post::YOLOv8PostProcessor::getType() const {
    return mType;
}

/*
        Apply sigmoid function to the input float number
*/
float mobilint::post::YOLOv8PostProcessor::sigmoid(float num) {
    return 1 / (1 + exp(-(float)num));
}

/*
        Applying inverse sigmoid function to the input float number
*/
float mobilint::post::YOLOv8PostProcessor::inverse_sigmoid(float num) {
    return -log(1 / num - 1);
}

/*
        Apply softmax function to the input vector
*/
std::vector<float> mobilint::post::YOLOv8PostProcessor::softmax(
    const std::vector<float>& vec) {
    std::vector<float> result;
    float sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += exp(vec[i]);
    }

    for (int i = 0; i < vec.size(); i++) {
        result.push_back(exp(vec[i]) / sum);
    }
    return result;
}

/*
        Calculate the area of the box
*/
float mobilint::post::YOLOv8PostProcessor::area(float xmin, float ymin, float xmax,
                                                float ymax) {
    float width = xmax - xmin;
    float height = ymax - ymin;

    if (width < 0) return 0;

    if (height < 0) return 0;

    return width * height;
}

/*
        Calculate IoU of two input boxes
*/
float mobilint::post::YOLOv8PostProcessor::get_iou(const std::array<float, 4>& box1,
                                                   const std::array<float, 4>& box2) {
    float epsilon = 1e-6;

    // Coordinated of the overlapped region(intersection of two boxes)
    float overlap_xmin = std::max(box1[0], box2[0]);
    float overlap_ymin = std::max(box1[1], box2[1]);
    float overlap_xmax = std::min(box1[2], box2[2]);
    float overlap_ymax = std::min(box1[3], box2[3]);

    // Calculate areas
    float overlap_area = area(overlap_xmin, overlap_ymin, overlap_xmax, overlap_ymax);
    float area1 = area(box1[0], box1[1], box1[2], box1[3]);
    float area2 = area(box2[0], box2[1], box2[2], box2[3]);
    float iou = overlap_area / (area1 + area2 - overlap_area + epsilon);

    return iou;
}

/*
        Convert boxes from Center Form(CxCyWiHe) to Corner Form(XminYminXmaxYmax)
*/
void mobilint::post::YOLOv8PostProcessor::xywh2xyxy(
    std::vector<std::array<float, 4>>& pred_boxes) {
    for (uint32_t i = 0; i < pred_boxes.size(); i++) {
        float cx = pred_boxes[i][0];
        float cy = pred_boxes[i][1];

        pred_boxes[i][0] = cx - pred_boxes[i][2] * 0.5;
        pred_boxes[i][1] = cy - pred_boxes[i][3] * 0.5;
        pred_boxes[i][2] = cx + pred_boxes[i][2] * 0.5;
        pred_boxes[i][3] = cy + pred_boxes[i][3] * 0.5;
    }
}

/*
        Access elements in output related to box coordinates and decode them
*/
void mobilint::post::YOLOv8PostProcessor::decode_boxes(const std::vector<float>& npu_out,
                                                       const std::vector<int>& grid,
                                                       int stride, int idx,
                                                       std::array<float, 4>& pred_box) {
    std::array<float, 4> box;
    std::array<float, 16> tmp;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 16; k++) {
            tmp[k] = npu_out[idx * 64 + j * 16 + k];
        }
        softmax_inplace(tmp);

        float box_value = 0;
        for (int k = 0; k < 16; k++) box_value += tmp[k] * k;
        box[j] = box_value;
    }

    float xmin = grid[idx * 2 + 0] - box[0] + 0.5;
    float ymin = grid[idx * 2 + 1] - box[1] + 0.5;
    float xmax = grid[idx * 2 + 0] + box[2] + 0.5;
    float ymax = grid[idx * 2 + 1] + box[3] + 0.5;

    float x = (xmin + xmax) / 2 * stride;
    float y = (ymin + ymax) / 2 * stride;
    float w = (xmax - xmin) * stride;
    float h = (ymax - ymin) * stride;

    pred_box = {x, y, w, h};
}

/*
        Access elements in output related to extra and decode them
*/
void mobilint::post::YOLOv8PostProcessor::decode_extra(const std::vector<float>& extra,
                                                       const std::vector<int>& grid,
                                                       int stride, int idx,
                                                       std::vector<float>& pred_extra) {
    // Do nothing for Object Detection
}

/*
        Decoding and masking with conf threshold
*/
void mobilint::post::YOLOv8PostProcessor::decode_conf_thres(
    const std::vector<float>& npu_out_box, const std::vector<float>& npu_out_cls,
    const std::vector<int>& grid, int stride,
    std::vector<std::array<float, 4>>& pred_boxes, std::vector<float>& pred_conf,
    std::vector<int>& pred_label, std::vector<std::pair<float, int>>& pred_scores,
    const std::vector<float>& extra, std::vector<std::vector<float>>& pred_extra) {
    int grid_h = m_imh / stride;
    int grid_w = m_imw / stride;
    int tmp_no = npu_out_cls.size() / (grid_h * grid_w);

    // if (tmp_no != m_nc + 64)  // 64 = 16 * 4
    //     throw std::invalid_argument(
    //         "Number of outputs per grid should be 114, "
    //         "however post-processor received " +
    //         std::to_string(tmp_no));

#pragma omp parallel for num_threads(mOpenmpThreadCount) \
    shared(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra)
    for (int i = 0; i < grid_h * grid_w; i++) {
        std::array<float, 4> pred_box = {-999, -999, -999, -999};
        std::vector<float> pred_extra_values;
        for (int j = 0; j < m_nc; j++) {
            if (npu_out_cls[i * tmp_no + j] > m_conf_thres) {
                float conf = npu_out_cls[i * tmp_no + j];
                if (pred_box[0] == -999) {  // decode box only once
                    decode_boxes(npu_out_box, grid, stride, i, pred_box);
                    decode_extra(extra, grid, stride, i, pred_extra_values);
                }

#pragma omp critical
                {
                    pred_conf.push_back(conf);
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
        Apply NMS
*/
void mobilint::post::YOLOv8PostProcessor::nms(
    const std::vector<std::array<float, 4>>& pred_boxes,
    const std::vector<float>& pred_conf, const std::vector<int>& pred_label,
    std::vector<std::pair<float, int>>& scores,
    const std::vector<std::vector<float>>& pred_extra,
    std::vector<std::array<float, 4>>& final_boxes, std::vector<float>& final_scores,
    std::vector<int>& final_labels, std::vector<std::vector<float>>& final_extra) {
    // sort the scores(predicted confidence * predicted class score)
    sort(scores.begin(), scores.end(), std::greater<>());

    for (int i = 0; i < (int)scores.size(); i++) {
        float temp_score = scores[i].first;
        if (scores[i].first != -99) {  // check if the box valid or not
            int idx = scores[i].second;
            std::array<float, 4> max_box = pred_boxes[idx];

#pragma omp parallel for num_threads(mOpenmpThreadCount) \
    shared(pred_boxes, pred_conf, pred_label, scores, i, idx, max_box)
            for (int j = i; j < (int)scores.size(); j++) {
                int temp_idx = scores[j].second;
                const std::array<float, 4>& temp_box = pred_boxes[temp_idx];
                float iou = get_iou(max_box, temp_box);

                if (iou > m_iou_thres && pred_label[idx] == pred_label[temp_idx]) {
                    scores[j].first = -99;  // mark the invalid boxes
                }
            }

            final_boxes.push_back(max_box);
            final_scores.push_back(temp_score);
            final_labels.push_back(pred_label[idx]);
            final_extra.push_back(pred_extra[idx]);

            if (final_boxes.size() >= m_max_det_num) {
                break;
            }
        }
    }
}

/*
        Returns current time, used to measure performance time
*/
double mobilint::post::YOLOv8PostProcessor::set_timer() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::vector<std::array<float, 4>>& mobilint::post::YOLOv8PostProcessor::get_result_box() {
    return final_boxes;
}

std::vector<float>& mobilint::post::YOLOv8PostProcessor::get_result_score() {
    return final_scores;
}

std::vector<int>& mobilint::post::YOLOv8PostProcessor::get_result_label() {
    return final_labels;
}

std::vector<std::vector<float>>& mobilint::post::YOLOv8PostProcessor::get_result_extra() {
    return final_extra;
}

void mobilint::post::YOLOv8PostProcessor::run_postprocess(
    const std::vector<std::vector<float>>& npu_outs) {
    double start = set_timer();

    if (npu_outs.size() != 2 * m_nl)
        throw std::invalid_argument(
            "Size of model outputs does not match "
            "number of detection layers, expected " +
            std::to_string(m_nl) + " but received " + std::to_string(npu_outs.size()));

    final_boxes.clear();
    final_scores.clear();
    final_labels.clear();
    final_extra.clear();

    std::vector<std::array<float, 4>> pred_boxes;
    std::vector<float> pred_conf;
    std::vector<int> pred_label;
    std::vector<std::pair<float, int>> pred_scores;
    std::vector<std::vector<float>> pred_extra;
    std::vector<float> extra;

    for (int i = 0; i < m_nl; i++) {
        decode_conf_thres(npu_outs[4 - 2 * i], npu_outs[5 - 2 * i], m_grids[i],
                          m_strides[i], pred_boxes, pred_conf, pred_label, pred_scores,
                          extra, pred_extra);
    }

    xywh2xyxy(pred_boxes);

    nms(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra, final_boxes,
        final_scores, final_labels, final_extra);
    double end = set_timer();
    if (m_verbose) std::cout << "Real C++ Time        : " << end - start << std::endl;
}

/*
        Compute the ratio and pads needed to switch between
        original image size and model input image size
*/
void mobilint::post::YOLOv8PostProcessor::compute_ratio_pads(cv::Mat im, float& ratio,
                                                             float& xpad, float& ypad) {
    cv::Size size = im.size();
    if (size.width > size.height) {
        ratio = (float)m_imw / size.width;
        xpad = 0;
        ypad = (m_imh - ratio * size.height) / 2;
    } else {
        ratio = (float)m_imh / size.height;
        xpad = (m_imw - ratio * size.width) / 2;
        ypad = 0;
    }
}

/*
        Draw detected box and write the it's label & score
*/
void mobilint::post::YOLOv8PostProcessor::plot_boxes(
    cv::Mat& im, std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
    std::vector<int>& labels) {
    float ratio, xpad, ypad;
    compute_ratio_pads(im, ratio, xpad, ypad);

    cv::Rect rect;
    for (int i = 0; i < boxes.size(); i++) {
        // if (labels[i] != 0) {
        //     continue;
        // }
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
        cv::rectangle(im, rect, clr, 2);

        if (m_nc != 1) {
            double font_scale = std::min(std::max(rect.width / 500.0, 0.35), 0.99) * 1.3;
            std::string desc = COCO_LABELS[labels[i]] + " " +
                               std::to_string((int)(scores[i] * 100)) + "%";
            cv::putText(im, desc, cv::Point(xmin, ymin - 10), cv::FONT_HERSHEY_SIMPLEX,
                        font_scale, clr, 2, false);
        }
    }
}

/*
        Plot extra data such as landmarks. keypoins, masks.
        Simple Object Detection does not any extras.
*/
void mobilint::post::YOLOv8PostProcessor::plot_extras(
    cv::Mat& im, std::vector<std::vector<float>>& extras) {
    // do nothing
    int do_nothing;
}

/*
        Print out given message in DEBUG mode using locks
*/
void mobilint::post::YOLOv8PostProcessor::print(std::string msg) {
#ifdef DEBUG
    std::lock_guard<std::mutex> lk(mPrintMutex);
    std::cout << msg << endl;
#endif
}

void mobilint::post::YOLOv8PostProcessor::compute_ratio(int srcHeight, int srcWidth,
                                                        float& h_ratio, float& w_ratio) {
    h_ratio = (float)m_imh / srcHeight;
    w_ratio = (float)m_imw / srcWidth;
}

void mobilint::post::YOLOv8PostProcessor::compute_ratio_pads(int srcHeight, int srcWidth,
                                                             float& ratio, int& xpad,
                                                             int& ypad) {
    float w_ratio = (float)m_imw / srcWidth;
    float h_ratio = (float)m_imh / srcHeight;
    if (w_ratio > h_ratio) {
        ratio = h_ratio;
        ypad = 0;
        xpad = (int)(m_imw - (int)floor(ratio * srcWidth + 0.5)) / 2;
    } else {
        ratio = w_ratio;
        xpad = 0;
        ypad = (int)(m_imh - (int)floor(ratio * srcHeight + 0.5)) / 2;
    }
}

void mobilint::post::YOLOv8PostProcessor::rescaleImg(
    std::vector<std::array<int, 4>>& rescaled_boxes, int srcHeight, int srcWidth) {
    float h_ratio, w_ratio;
    compute_ratio(srcHeight, srcWidth, h_ratio, w_ratio);

    for (int i = 0; i < final_boxes.size(); i++) {
        int xmin = (int)(final_boxes[i][0]) / w_ratio;
        int ymin = (int)(final_boxes[i][1]) / h_ratio;
        int xmax = (int)(final_boxes[i][2]) / w_ratio;
        int ymax = (int)(final_boxes[i][3]) / h_ratio;

        xmin = std::max(0, xmin);
        ymin = std::max(0, ymin);
        xmax = std::min(srcWidth, xmax);
        ymax = std::min(srcHeight, ymax);

        int width = xmax - xmin;
        int height = ymax - ymin;

        std::array<int, 4> tmp = {xmin, ymin, width, height};
        rescaled_boxes.push_back(tmp);
    }
}

void mobilint::post::YOLOv8PostProcessor::rescaleImgWithPadding(
    std::vector<std::array<float, 4>>& rescaled_boxes, int srcHeight, int srcWidth) {
    float ratio;
    int xpad, ypad;
    compute_ratio_pads(srcHeight, srcWidth, ratio, xpad, ypad);

    for (int i = 0; i < final_boxes.size(); i++) {
        float xmin = (float)(final_boxes[i][0] - xpad) / ratio;
        float ymin = (float)(final_boxes[i][1] - ypad) / ratio;
        float xmax = (float)(final_boxes[i][2] - xpad) / ratio;
        float ymax = (float)(final_boxes[i][3] - ypad) / ratio;

        xmin = std::max((float)0, xmin);
        ymin = std::max((float)0, ymin);
        xmax = std::min((float)srcWidth, xmax);
        ymax = std::min((float)srcHeight, ymax);

        float width = xmax - xmin;
        float height = ymax - ymin;

        std::array<float, 4> tmp = {xmin, ymin, width, height};
        rescaled_boxes.push_back(tmp);
    }
}

void mobilint::post::YOLOv8PostProcessor::rescaleImgWithPaddingV2(
    std::vector<std::array<float, 4>>& rescaled_boxes, int srcHeight, int srcWidth) {
    float gain;
    int xpad, ypad;

    gain = std::min((float)m_imh / srcHeight, (float)m_imw / srcWidth);

    xpad = (int)std::round(((float)m_imw - srcWidth * gain) / 2.0f - 0.1f);
    ypad = (int)std::round(((float)m_imh - srcHeight * gain) / 2.0f - 0.1f);

    for (int i = 0; i < final_boxes.size(); i++) {
        float xmin = (float)(final_boxes[i][0] - xpad) / gain;
        float ymin = (float)(final_boxes[i][1] - ypad) / gain;
        float xmax = (float)(final_boxes[i][2] - xpad) / gain;
        float ymax = (float)(final_boxes[i][3] - ypad) / gain;

        xmin = std::max((float)0, std::min(xmin, (float)srcWidth));
        xmax = std::max((float)0, std::min(xmax, (float)srcWidth));
        ymin = std::max((float)0, std::min(ymin, (float)srcHeight));
        ymax = std::max((float)0, std::min(ymax, (float)srcHeight));

        float width = xmax - xmin;
        float height = ymax - ymin;

        std::array<float, 4> tmp = {xmin, ymin, width, height};
        rescaled_boxes.push_back(tmp);
    }
}

void mobilint::post::YOLOv8PostProcessor::worker() {
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

        plot_boxes(k.im, k.boxes, k.scores, k.labels);
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

uint64_t mobilint::post::YOLOv8PostProcessor::enqueue(
    cv::Mat& im, std::vector<std::vector<float>>& npu_outs,
    std::vector<std::array<float, 4>>& boxes, std::vector<float>& scores,
    std::vector<int>& labels, std::vector<std::vector<float>>& extras) {
    auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto title = std::to_string(thres_id) + " | Postprocessor Enqueue | ";

    // print(title + "Start");
    uint64_t ticket_save = 0;

    {
        std::lock_guard<std::mutex> lk(mMutexIn);
        mQueueIn.push({++ticket, im, npu_outs, boxes, scores, labels, extras});
        ticket_save = ticket;

        mCondIn.notify_all();  // JUST IN CASE
        // print(title + "Input Queue size " + std::to_string(mQueueIn.size()));
    }

    // print(title + "Finish");
    return ticket_save;
}

void mobilint::post::YOLOv8PostProcessor::receive(uint64_t receipt_no) {
    // auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
    // auto title = std::to_string(thres_id) + " | Postprocessor Receive | ";

    while (!destroyed) {
        // print(title + "Start | Receipt = " + std::to_string(receipt_no));
        std::unique_lock<std::mutex> lk(mMutexOut);

        if (mOut.empty()) {
            mCondOut.wait(lk, [this] { return !(mOut.empty()) || destroyed; });
        }

        // print(title + "Received Output Queue of size " + std::to_string(mOut.size()));

        if (destroyed) {
            break;
        }

        for (int i = 0; i < mOut.size(); i++) {
            if (mOut[i] == receipt_no) {
                // print(title + "Got my output | Receipt = " + std::to_string(mOut[i]));
                mOut.erase(mOut.begin() + i);
                return;
            }
        }

        lk.unlock();
        // print(title + "Finish");
    }
}
