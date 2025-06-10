#ifndef YOLOV8_POSE_POSTPROCESSOR
#define YOLOV8_POSE_POSTPROCESSOR

#include "post_yolov8.h"

namespace mobilint::post {
class YOLOv8PosePostProcessor : public YOLOv8PostProcessor {
public:
    YOLOv8PosePostProcessor();
    YOLOv8PosePostProcessor(int nc, int imh, int imw, float conf_thres, float iou_thres,
                            bool verbose);

    void run_postprocess(const std::vector<std::vector<float>>& npu_outs);
    void decode_extra(const std::vector<float>& extra, const std::vector<int>& grid,
                      int stride, int idx, std::vector<float>& pred_extra);
    void plot_keypoints(cv::Mat& im, std::vector<std::vector<float>>& kpts);
    void plot_extras(cv::Mat& im, std::vector<std::vector<float>>& extras);
    void worker();

    // not the best practice, should find better way
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
}  // namespace mobilint::post

#endif
