#ifndef YOLOV8_FACE_POSTPROCESSOR
#define YOLOV8_FACE_POSTPROCESSOR

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <queue>
#include <thread>
#include "./maccel/maccel.h"
#include <opencv2/opencv.hpp>

namespace mobilint::postface {
	enum class PostType {BASE, FACE, POSE, SEG};

	struct YOLOv8PostItem {
		uint64_t id;
		cv::Mat& im;
		std::vector<mobilint::NDArray<float>>& npu_outs;
		std::vector<std::array<float, 4>>& boxes;
		std::vector<float>& scores;
		std::vector<int>& labels;
		std::vector<mobilint::NDArray<float>>& npu_outs2;
		std::vector<std::vector<float>>& extras;
	};

	class YOLOv8FacePostProcessor
	{
		public:
			YOLOv8FacePostProcessor();
			YOLOv8FacePostProcessor(
				int imh, int imw);
			YOLOv8FacePostProcessor(
				int nc, int imh, int imw, 
				float conf_thres, 
				float iou_thres, 
				int max_num_threads, 
				bool verbose);
			~YOLOv8FacePostProcessor();

		public:
			std::vector<int> generate_strides(int nl);
			std::vector<std::vector<int>> generate_grids(
				int imh, int imw, std::vector<int> strides);
			void run_postprocess(const std::vector<mobilint::NDArray<float>> &npu_outs, const std::vector<mobilint::NDArray<float>> &npu_outs2);

			int get_nl() const;
			int get_nc() const;
			int get_imh() const;
			int get_imw() const;
			PostType getType() const;
			float sigmoid(const float &num);
			float inverse_sigmoid(const float &num);
			std::vector<float> softmax(const std::vector<float> &vec);
			float area(const float &xmin, const float &ymin, const float &xmax, const float &ymax);
			float get_iou(const std::array<float, 4> &box1, const std::array<float, 4> &box2);
			void xywh2xyxy(std::vector<std::array<float, 4>> &pred_boxes);
			virtual void decode_extra(
				const float* npu_out,
				const std::vector<int> &grid,
				const int &stride, const int &idx,
				std::vector<float> &pred_extra);
			void decode_boxes(
				const float* npu_out,
				const std::vector<int> &grid,
				const int &stride, const int &idx,
				std::array<float, 4> &pred_box,
				std::vector<float> &pred_extra);
			void decode_conf_thres(
				const float* npu_out,
				const std::vector<int> &grid,
				const int &stride, 
				std::vector<std::array<float,4>> &pred_boxes, 
				std::vector<float> &pred_conf, 
				std::vector<int> &pred_label, 
				std::vector<std::pair<float, int>> &pred_scores, 
				const float* npu_out2,
				std::vector<std::vector<float>> &pred_extra);
			void nms(
				const std::vector<std::array<float, 4>> &pred_boxes, 
				const std::vector<float> &pred_conf, 
				const std::vector<int> &pred_label, 
				std::vector<std::pair<float,int>> &scores, 
				const std::vector<std::vector<float>> &pred_extra, 
				std::vector<std::array<float, 4>> &final_boxes, 
				std::vector<float> &final_scores, 
				std::vector<int> &final_labels, 
				std::vector<std::vector<float>> &final_extra);
			double set_timer();

			std::vector<std::array<float, 4>>& get_result_box();
			std::vector<float>& get_result_score();
			std::vector<int>& get_result_label();
			std::vector<std::vector<float>>& get_result_extra();
			void compute_ratio_pads(const cv::Mat &im, float& ratio,
				float& xpad, float& ypad);
			void compute_ratio_pads(const float &width, const float &height, 
				float& ratio, float& xpad, float& ypad);
			void plot_boxes(cv::Mat& im, 
				const std::vector<std::array<float, 4>>& boxes,
				const std::vector<float>& scores, 
				const std::vector<int>& labels);
			virtual void plot_extras(cv::Mat& im, 
				const std::vector<std::vector<float>>& extras);

			void print(std::string msg);
			virtual void worker();
			uint64_t enqueue(
				cv::Mat& im,
				std::vector<mobilint::NDArray<float>>& npu_outs, 
				std::vector<std::array<float, 4>>& boxes,
				std::vector<float>& scores, 
				std::vector<int>& labels,
				std::vector<mobilint::NDArray<float>>& npu_outs2);
			void receive(uint64_t receipt_no);

		protected:
			int m_max_num_threads;
			int m_nextra; // number of keypoints/landmarks/masks
			int m_nl; // number of detection layers
			int m_nc; // number of classes
			uint32_t m_imh; // model input image height
			uint32_t m_imw; // model input image width
			float m_conf_thres; // confidence threshold, used in decoding
			float m_inverse_conf_thres; // inverse confidence threshold, used in decoding
			float m_iou_thres; // iou threshold, used in nms
			int m_max_det_num;
			bool m_verbose;
			PostType mType;

			std::vector<std::array<float, 4>> final_boxes;
			std::vector<float> final_scores;
			std::vector<int> final_labels;
			std::vector<std::vector<float>> final_extra; // keypoints/landmarks/masks

			std::vector<int> m_strides;
			std::vector<std::vector<int>> m_grids;


			std::thread mThread;
			std::queue<YOLOv8PostItem> mQueueIn;
			std::vector<uint64_t> mOut;
			uint64_t ticket;

			std::mutex mPrintMutex;
			std::mutex mMutexIn;
			std::mutex mMutexOut;
			std::condition_variable mCondIn;
			std::condition_variable mCondOut;
			bool destroyed;

			// not the best practice, should find better way
			const std::vector<std::string> COCO_LABELS = {
				"face"
			};

			const std::vector<std::array<int, 3>> COLORS = {
				{56, 56, 255},
				{151, 157, 255},
				{31, 112, 255},
				{29, 178, 255},
				{49, 210, 207},
				{10, 249, 72},
				{23, 204, 146},
				{134, 219, 61},
				{52, 147, 26},
				{187, 212, 0},
				{168, 153, 44},
				{255, 194, 0},
				{147, 69, 52},
				{255, 115, 100},
				{236, 24, 0},
				{255, 56, 132},
				{133, 0, 82},
				{255, 56, 203},
				{200, 149, 255},
				{199, 55, 255}
			};

			const std::vector<std::array<int, 3>> LMARK_COLORS = {
                {255, 255, 0},
				{255, 255, 0},
				{255, 255, 0},
                {255, 255, 0},
				{255, 255, 0}
            };
	};
};

#endif
