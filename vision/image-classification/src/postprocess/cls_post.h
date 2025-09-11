#include <maccel/maccel.h>
#include <math.h>
#include <omp.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "postprocess.h"

namespace mobilint::post {

class CLSPostProcessor : public PostProcessor {
public:
    CLSPostProcessor();
    CLSPostProcessor(int imh, int imw, const ModelInfo& cfg);
    CLSPostProcessor(int nc, int imh, int imw, int max_num_threads = 4);

public:
    void set_label_file(std::string label_file) { m_label_file = label_file; }
    void load_labels() { load_labels(m_label_file); }
    void load_labels(std::string label_file);
    void set_params(int nc, int imh, int imw, float conf_thres = 0.0,
                    float iou_thres = 0.0, int nl = 0, int max_num_threads = 4) override;
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs);

    void decode_conf_thres(const std::vector<float>& npu_out,
                           std::vector<float>& pred_scores, std::vector<int>& pred_label);

    void plot_results(cv::Mat& im, const std::vector<float>& scores,
                      const std::vector<int>& labels) override;
    void plot_results(cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
                      const std::vector<float>& scores, const std::vector<int>& labels,
                      const std::vector<std::vector<float>>& extras = {}) override {
        plot_results(im, scores, labels);
    }
    void worker() override;

protected:
    std::string m_label_file = "../imagenet.txt";
    std::vector<std::string> m_labels;
    int m_topk = 5;
};
};  // namespace mobilint::post
