#include <math.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "./maccel/maccel.h"
#include "postprocess.h"

namespace mobilint::post {

class YOLOAnchorlessSegPostProcessor : public PostProcessor {
public:
    YOLOAnchorlessSegPostProcessor();
    YOLOAnchorlessSegPostProcessor(int nc, int imh, int imw);
    YOLOAnchorlessSegPostProcessor(int nc, int imh, int imw, float conf_thres,
                                   float iou_thres, int max_num_threads = 4);
    ~YOLOAnchorlessSegPostProcessor();

public:
    void set_params(int nc, int imh, int imw, float conf_thres = 0.3,
                    float iou_thres = 0.6, int max_num_threads = 4) override;
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs);

    std::vector<std::vector<int>> generate_grids(int imh, int imw,
                                                 std::vector<int> strides);
    std::vector<int> generate_strides(int nl);

    std::vector<std::array<float, 4>> downsample_boxes(
        const std::vector<std::array<float, 4>>& boxes);
    void process_mask(const std::vector<float>& proto,
                      const std::vector<std::vector<float>>& masks,
                      const std::vector<std::array<float, 4>>& boxes,
                      const std::vector<int>& labels);

    void decode_boxes(const std::vector<float>& npu_out_box, const std::vector<int>& grid,
                      int stride, int idx, std::array<float, 4>& pred_box);

    void decode_extra(const std::vector<float>& npu_out_extra,
                      const std::vector<int>& grid, int stride, int idx,
                      std::vector<float>& pred_extra);
    void decode_conf_thres(const std::vector<float>& npu_out_box,
                           const std::vector<float>& npu_out_cls,
                           const std::vector<float>& npu_out_extra,
                           const std::vector<int>& grid, int stride,
                           std::vector<std::array<float, 4>>& pred_boxes,
                           std::vector<std::pair<float, int>>& pred_scores,
                           std::vector<int>& pred_label,
                           std::vector<std::vector<float>>& pred_extra);
    void get_results(int org_h, int org_w, std::vector<std::array<float, 4>>& boxes,
                     std::vector<int>& labels,
                     std::vector<std::vector<unsigned int>>& seg_results);
    void plot_masks(cv::Mat& im, const std::vector<std::array<float, 4>>& boxes,
                    const std::vector<int>& labels);
    void worker() override;

protected:
    std::vector<int> m_strides;
    std::vector<std::vector<int>> m_grids;
    int m_proto_stride;
    int m_proto_h;
    int m_proto_w;

    int m_reg_max = 16;
    std::vector<cv::Mat> m_instance_masks;
};
};  // namespace mobilint::post
