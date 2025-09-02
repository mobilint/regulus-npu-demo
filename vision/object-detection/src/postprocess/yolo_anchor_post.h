#include <maccel/maccel.h>
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

#include "postprocess.h"

namespace mobilint::post {

class YOLOAnchorPostProcessor : public PostProcessor {
public:
    YOLOAnchorPostProcessor();
    YOLOAnchorPostProcessor(int nc, int imh, int imw);
    YOLOAnchorPostProcessor(int nc, int imh, int imw, float conf_thres, float iou_thres,
                            int max_num_threads = 4);
    ~YOLOAnchorPostProcessor();

public:
    void set_params(int nc, int imh, int imw, float conf_thres = 0.3,
                    float iou_thres = 0.6, int max_num_threads = 4) override;
    std::vector<std::vector<int>> generate_grids(int imh, int imw,
                                                 std::vector<int> strides);
    std::vector<int> generate_strides(int nl);
    void set_default_anchors();
    void decode_boxes(const std::vector<float>& npu_out_box, const std::vector<int>& grid,
                      int stride, int idx, std::array<float, 4>& pred_box);
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs);

    void decode_conf_thres(const std::vector<float>& npu_out,
                           const std::vector<int>& grid, int stride,
                           const std::vector<std::vector<int>>& anchor,
                           std::vector<std::array<float, 4>>& pred_boxes,
                           std::vector<std::pair<float, int>>& pred_scores,
                           std::vector<int>& pred_label);
    void worker() override;

protected:
    int m_no;  // number outputs per anchor (5 + nc + keypts/lmarks/masks)
    int m_na;  // number of anchors
    std::vector<int> m_strides;
    std::vector<std::vector<int>> m_grids;
    std::vector<std::vector<std::vector<int>>> m_anchors;

    int m_reg_max = 16;
};
};  // namespace mobilint::post
