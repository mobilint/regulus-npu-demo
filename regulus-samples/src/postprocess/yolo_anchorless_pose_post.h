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

class YOLOAnchorlessPosePostProcessor : public PostProcessor {
public:
    YOLOAnchorlessPosePostProcessor();
    YOLOAnchorlessPosePostProcessor(int nc, int imh, int imw);
    YOLOAnchorlessPosePostProcessor(int nc, int imh, int imw, float conf_thres,
                                    float iou_thres, int max_num_threads = 4);
    ~YOLOAnchorlessPosePostProcessor();

public:
    void setParams(int nc, int imh, int imw, float conf_thres = 0.3,
                   float iou_thres = 0.6, int max_num_threads = 4) override;
    void run_postprocess(const std::vector<std::vector<float>>& npu_outs);

    std::vector<std::vector<int>> generate_grids(int imh, int imw,
                                                 std::vector<int> strides);
    std::vector<int> generate_strides(int nl);

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
    void plot_keypoints(cv::Mat& im, std::vector<std::vector<float>>& kpts);
    void worker() override;

protected:
    std::vector<int> m_strides;
    std::vector<std::vector<int>> m_grids;

    int m_reg_max = 16;

    const std::vector<std::array<int, 2>> m_skeleton = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
        {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
        {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}};

    const std::vector<std::array<int, 3>> m_pose_limb_color = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255},
        {255, 51, 255}, {255, 51, 255}, {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}};

    const std::vector<std::array<int, 3>> m_pose_kpt_color = {
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},
        {255, 128, 0},  {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255},
        {51, 153, 255}, {51, 153, 255}};
};
};  // namespace mobilint::post
