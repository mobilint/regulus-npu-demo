#include "post_yolov8_face.h"

namespace {

/*
	Apply softmax function to the input vector in-place
*/
template <typename FloatContainer>
void softmax_inplace(
	FloatContainer &container){

	float sum = 0;
	for (auto v : container) {
		sum += exp(v);
	}
	for (auto& v: container) {
		v = exp(v) / sum;
	}
}

float softmax_inplace_idx(
	const float* npu_out, int start_idx, int end_idx){
	
	float sum = 0, result = 0;
	for (int i = start_idx; i < end_idx; i++){
		sum += exp(npu_out[i]);
	}
	for (int i = start_idx; i < end_idx; i++){
		result += exp(npu_out[i]) / sum * (i - start_idx);
	}
	return result;
}


}  // namespace

mobilint::postface::YOLOv8FacePostProcessor::YOLOv8FacePostProcessor(){
	m_max_num_threads = 4;
	m_nc = 1; // number of classes
	m_nl = 3; // number of detection layers
	m_nextra = 15; // number of extra outputs
	m_imh = 640; // model input image height
	m_imw = 640; // model input image width
	m_conf_thres = 0.5; // confidence threshold, used in decoding
	m_inverse_conf_thres = inverse_sigmoid(m_conf_thres);
	m_iou_thres = 0.45; // iou threshold, used in nms
	m_verbose = false;
	m_max_det_num = 300;
	mType = PostType::BASE;
	m_strides = generate_strides(m_nl);
	m_grids = generate_grids(m_imh, m_imw, m_strides);

	ticket = 0;
	destroyed = false;
	mThread = std::thread(&mobilint::postface::YOLOv8FacePostProcessor::worker, this);
}

mobilint::postface::YOLOv8FacePostProcessor::YOLOv8FacePostProcessor(
	int imh, int imw){
	
	m_imh = imh; // model input image height
	m_imw = imw; // model input image width

	m_max_num_threads = 4;
	m_nc = 1; // number of classes
	m_nl = 3; // number of detection layers
	m_nextra = 15; // number of extra outputs
	m_conf_thres = 0.5; // confidence threshold, used in decoding
	m_inverse_conf_thres = inverse_sigmoid(m_conf_thres);
	m_iou_thres = 0.45; // iou threshold, used in nms
	m_verbose = false;
	m_max_det_num = 300;
	mType = PostType::BASE;
	m_strides = generate_strides(m_nl);
	m_grids = generate_grids(m_imh, m_imw, m_strides);

	ticket = 0;
	destroyed = false;
	mThread = std::thread(&mobilint::postface::YOLOv8FacePostProcessor::worker, this);
} 

mobilint::postface::YOLOv8FacePostProcessor::YOLOv8FacePostProcessor(
	int nc, int imh, int imw, 
	float conf_thres, 
	float iou_thres, 
	int max_num_threads, 
	bool verbose){

	m_max_num_threads = max_num_threads;
	m_nc = nc; // number of classes
	m_nl = 3; // number of detection layers
	m_nextra = 15; // number of extra outputs
	m_imh = imh; // model input image height
	m_imw = imw; // model input image width
	m_conf_thres = conf_thres; // confidence threshold, used in decoding
	m_inverse_conf_thres = inverse_sigmoid(m_conf_thres);
	m_iou_thres = iou_thres; // iou threshold, used in nms
	m_verbose = verbose;
	m_max_det_num = 300;
	mType = PostType::BASE;
	m_strides = generate_strides(m_nl);
	m_grids = generate_grids(m_imh, m_imw, m_strides);

	ticket = 0;
	destroyed = false;
	mThread = std::thread(&mobilint::postface::YOLOv8FacePostProcessor::worker, this);
}

mobilint::postface::YOLOv8FacePostProcessor::~YOLOv8FacePostProcessor(){
	destroyed = true;
	mCondIn.notify_all();
	mCondOut.notify_all();
	mThread.join();
}

/*
	Generate strides, needed for decoding
*/
std::vector<int> mobilint::postface::YOLOv8FacePostProcessor::generate_strides(int nl){
	std::vector<int> strides;
	for(int i=0; i<nl; i++){
		strides.push_back(pow(2, 3 + i));
	}
	return strides;
}

/*
	Generate grids, needed for decoding
*/
std::vector<std::vector<int>> mobilint::postface::YOLOv8FacePostProcessor::generate_grids(
	int imh, int imw, 
	std::vector<int> strides){

	std::vector<std::vector<int>> all_grids;
	for(int i = 0; i < strides.size(); i++){
		int grid_h = imh  / strides[i];
		int grid_w = imw / strides[i];
		int grid_size =  grid_h * grid_w * 2;

		std::vector<int> grids;
		for(int j = 0; j < grid_size; j++){
			if(j % 2 == 0){
				grids.push_back(((int) j / 2) % grid_w);
			} else{
				grids.push_back(((int) j / 2) / grid_w);
			}
		}

		all_grids.push_back(grids);
	}
	return all_grids;
}

int mobilint::postface::YOLOv8FacePostProcessor::get_nl() const{
	return m_nl;
}

int mobilint::postface::YOLOv8FacePostProcessor::get_nc() const{
	return m_nc;
}

int mobilint::postface::YOLOv8FacePostProcessor::get_imh() const{
	return m_imh;
}

int mobilint::postface::YOLOv8FacePostProcessor::get_imw() const{
	return m_imw;
}

/*
	Return the type of Post Processing
*/
mobilint::postface::PostType mobilint::postface::YOLOv8FacePostProcessor::getType() const{
	return mType;
}

/*
	Apply sigmoid function to the input float number
*/
float mobilint::postface::YOLOv8FacePostProcessor::sigmoid(const float &num){
	return 1 / (1 + exp(-(float) num));
}

/*
	Applying inverse sigmoid function to the input float number
*/
float mobilint::postface::YOLOv8FacePostProcessor::inverse_sigmoid(const float &num){
	return -log(1 / num - 1);
}

/*
	Apply softmax function to the input vector
*/
std::vector<float> mobilint::postface::YOLOv8FacePostProcessor::softmax(
	const std::vector<float> &vec){
	
	std::vector<float> result;
	float sum = 0;
	for(int i=0; i<vec.size(); i++){
		sum += exp(vec[i]);
	}

	for(int i=0; i<vec.size(); i++){
		result.push_back(exp(vec[i]) / sum);
	}
	return result;
}

/*
	Calculate the area of the box
*/
float mobilint::postface::YOLOv8FacePostProcessor::area(
	const float &xmin, const float &ymin, const float &xmax, const float &ymax){
	
	float width = xmax - xmin;
	float height = ymax - ymin;

	if(width < 0)
		return 0;

	if(height < 0)
		return 0;

	return width * height;
}

/*
	Calculate IoU of two input boxes
*/
float mobilint::postface::YOLOv8FacePostProcessor::get_iou(
	const std::array<float, 4> &box1, const std::array<float, 4> &box2){
	
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
void mobilint::postface::YOLOv8FacePostProcessor::xywh2xyxy(
	std::vector<std::array<float, 4>> &pred_boxes){

	for(uint32_t i=0; i<pred_boxes.size(); i++){
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
void mobilint::postface::YOLOv8FacePostProcessor::decode_boxes(
	const float* npu_out,
	const std::vector<int> &grid,
	const int &stride, const int &idx,
	std::array<float, 4> &pred_box,
	std::vector<float> &pred_extra){
	
	std::array<float, 4> box;
	// std::array<float, 16> tmp;
	int base_idx = idx * (m_nc + 64 + m_nextra);
	for (int j = 0; j < 4; j++){
		// for (int k = 0; k < 16; k++)
		// 	tmp[k] = npu_out[idx * (m_nc + 64) + j * 16 + k];
		// softmax_inplace(tmp);

		// float box_value = 0;
		// for (int k = 0; k < 16; k++)
		// 	box_value += tmp[k] * k;
		// box[j] = box_value;

		int curr_idx = base_idx + j * 16;
		box[j] = softmax_inplace_idx(npu_out, curr_idx, curr_idx + 16);

		// Do nothing for Object Detection
	}

	float xmin = grid[idx * 2 + 0] - box[0] + 0.5;
	float ymin = grid[idx * 2 + 1] - box[1] + 0.5;
	float xmax = grid[idx * 2 + 0] + box[2] + 0.5;
	float ymax = grid[idx * 2 + 1] + box[3] + 0.5;

	pred_box = {xmin * stride, ymin * stride, xmax * stride, ymax * stride};

	pred_extra.clear();
	int num_kpts = m_nextra / 3;
	for (int i = 0; i < num_kpts; i++){
		auto first  = npu_out[base_idx + 3 * i + 65];
		auto second = npu_out[base_idx + 3 * i + 66];
		auto third  = npu_out[base_idx + 3 * i + 67];

		first =  (first  * 2 + grid[idx * 2 + 0]) * stride;
		second = (second * 2 + grid[idx * 2 + 1]) * stride;
		third = sigmoid(third);

		pred_extra.push_back(first);
		pred_extra.push_back(second);
		pred_extra.push_back(third);
	}

	// float x = (xmin + xmax) / 2 * stride;
	// float y = (ymin + ymax) / 2 * stride;
	// float w = (xmax - xmin) * stride;
	// float h = (ymax - ymin) * stride;

	// pred_box = {x, y, w, h};
}

/*
	Access elements in output related to extra and decode them
*/
void mobilint::postface::YOLOv8FacePostProcessor::decode_extra(
	const float* npu_out,
	const std::vector<int> &grid,
	const int &stride, const int &idx,
	std::vector<float> &pred_extra){
	
	// Do nothing for Object Detection
	pred_extra.clear();
	int num_kpts = m_nextra / 3;
	for (int i = 0; i < num_kpts; i++){
		auto first  = npu_out[idx * m_nextra + 3 * i + 0];
		auto second = npu_out[idx * m_nextra + 3 * i + 1];
		auto third  = npu_out[idx * m_nextra + 3 * i + 2];

		first =  (first  * 2 + grid[idx * 2 + 0]) * stride;
		second = (second * 2 + grid[idx * 2 + 1]) * stride;
		third = sigmoid(third);

		pred_extra.push_back(first);
		pred_extra.push_back(second);
		pred_extra.push_back(third);
	}

}

/*
	Decoding and masking with conf threshold
*/
void mobilint::postface::YOLOv8FacePostProcessor::decode_conf_thres(
	const float* npu_out,
	const std::vector<int> &grid,
	const int &stride, 
	std::vector<std::array<float,4>> &pred_boxes, 
	std::vector<float> &pred_conf, 
	std::vector<int> &pred_label, 
	std::vector<std::pair<float, int>> &pred_scores,
	const float* npu_out2,
	std::vector<std::vector<float>> &pred_extra){

	int grid_h = m_imh  / stride;
	int grid_w = m_imw / stride;
	// int tmp_no = npu_out.size() / (grid_h * grid_w);

	// if (tmp_no != m_nc + 64) // 64 = 16 * 4
	// 	throw std::invalid_argument("Number of outputs per grid should be 114, "
	// 		"however post-processor received " + std::to_string(tmp_no));

	#pragma omp parallel for num_threads(m_max_num_threads) \
			shared(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra)
	for (int i = 0; i < grid_h * grid_w; i++){
		std::array<float, 4> pred_box = {-999, -999, -999, -999};
		std::vector<float> pred_extra_values;
		float conf;
		int curr_idx = i * (m_nc + 64 + m_nextra) + 64;
		for (int j = 0; j < m_nc; j++){
			if (npu_out[curr_idx + j] > m_inverse_conf_thres){
				if (pred_box[0] == -999){ // decode box only once
					conf = sigmoid(npu_out[curr_idx + j]);
					decode_boxes(npu_out, grid, stride, i, pred_box, pred_extra_values);
					// decode_extra(npu_out2, grid, stride, i, pred_extra_values);
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
void mobilint::postface::YOLOv8FacePostProcessor::nms(
	const std::vector<std::array<float, 4>> &pred_boxes, 
	const std::vector<float> &pred_conf, 
	const std::vector<int> &pred_label, 
	std::vector<std::pair<float,int>> &scores, 
	const std::vector<std::vector<float>> &pred_extra, 
	std::vector<std::array<float, 4>> &final_boxes, 
	std::vector<float> &final_scores, 
	std::vector<int> &final_labels, 
	std::vector<std::vector<float>> &final_extra){

	// sort the scores(predicted confidence * predicted class score)
	sort(scores.begin(), scores.end(), std::greater <>());

	for(int i=0; i<(int)scores.size(); i++){
		float temp_score = scores[i].first;
		if(scores[i].first != -99){ // check if the box valid or not
			int idx = scores[i].second;
			const std::array<float, 4> &max_box = pred_boxes[idx];

			#pragma omp parallel for num_threads(m_max_num_threads) \
				shared (pred_boxes, pred_conf, pred_label, scores, i, idx, max_box)
			for(int j=i; j<(int)scores.size(); j++){
				int temp_idx = scores[j].second;
				const std::array<float, 4> &temp_box = pred_boxes[temp_idx];
				float iou = get_iou(max_box, temp_box);
				
				if(iou > m_iou_thres && pred_label[idx] == pred_label[temp_idx]){
					scores[j].first = -99; // mark the invalid boxes
				}
			}

			final_boxes.push_back(max_box);
			final_scores.push_back(temp_score);
			final_labels.push_back(pred_label[idx]);
			final_extra.push_back(pred_extra[idx]);

			if (final_boxes.size() >= m_max_det_num){
				break;
			}
		}
	}
}

/*
	Returns current time, used to measure performance time
*/
double mobilint::postface::YOLOv8FacePostProcessor::set_timer(){
	return std::chrono::duration_cast<std::chrono::duration<double>>
		(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::vector<std::array<float, 4>>& mobilint::postface::YOLOv8FacePostProcessor::get_result_box() {
    return final_boxes;
}

std::vector<float>& mobilint::postface::YOLOv8FacePostProcessor::get_result_score() {
    return final_scores;
}

std::vector<int>& mobilint::postface::YOLOv8FacePostProcessor::get_result_label() {
    return final_labels;
}

std::vector<std::vector<float>>& mobilint::postface::YOLOv8FacePostProcessor::get_result_extra() {
    return final_extra;
}

void mobilint::postface::YOLOv8FacePostProcessor::run_postprocess(
	const std::vector<mobilint::NDArray<float>> &npu_outs, const std::vector<mobilint::NDArray<float>> &npu_outs2){

	double start = set_timer();
	
	if (npu_outs.size() != m_nl)
		throw std::invalid_argument("Size of model outputs does not match "
			"number of detection layers, expected " + std::to_string(m_nl) +
			" but received " + std::to_string(npu_outs.size()));

	final_boxes.clear();
	final_scores.clear();
	final_labels.clear();
	final_extra.clear();

	std::vector<std::array<float, 4>> pred_boxes;
	std::vector<float> pred_conf;
	std::vector<int> pred_label;
	std::vector<std::pair<float, int>> pred_scores;
	std::vector<std::vector<float>> pred_extra;

	for(int i=0; i<m_nl; i++){
		decode_conf_thres(npu_outs[i].data(), m_grids[i], m_strides[i], 
				pred_boxes, pred_conf, pred_label, pred_scores, 
				npu_outs2[i].data(), pred_extra);
	}

	// xywh2xyxy(pred_boxes);

	nms(pred_boxes, pred_conf, pred_label, pred_scores, pred_extra, 
		final_boxes, final_scores, final_labels, final_extra);
	double end = set_timer();
	if(m_verbose)
		std::cout << "Real C++ Time        : " << end - start << std::endl;
}

/*
	Compute the ratio and pads needed to switch between 
	original image size and model input image size
*/
void mobilint::postface::YOLOv8FacePostProcessor::compute_ratio_pads(const cv::Mat &im, 
	float& ratio, float& xpad, float& ypad){
	
	cv::Size size = im.size();
	if(size.width > size.height){
		ratio = (float) m_imw / size.width;
		xpad = 0;
		ypad = (m_imh - ratio * size.height) / 2;
	} else{
		ratio = (float) m_imh / size.height;
		xpad = (m_imw - ratio * size.width) / 2;
		ypad = 0;
	}
}

/*
	Compute the ratio and pads needed to switch between 
	original image size and model input image size
*/
void mobilint::postface::YOLOv8FacePostProcessor::compute_ratio_pads(
	const float &width, const float &height, float& ratio, float& xpad, float& ypad){

	if(width > height){
		ratio = (float) m_imw / width;
		xpad = 0;
		ypad = (m_imh - ratio * height) / 2;
	} else{
		ratio = (float) m_imh / height;
		xpad = (m_imw - ratio * width) / 2;
		ypad = 0;
	}
}

/*
	Draw detected box and write the it's label & score
*/
void mobilint::postface::YOLOv8FacePostProcessor::plot_boxes(cv::Mat& im, 
	const std::vector<std::array<float, 4>>& boxes,
	const std::vector<float>& scores, 
	const std::vector<int>& labels){

	float ratio, xpad, ypad;
	compute_ratio_pads(im, ratio, xpad, ypad);

	cv::Rect rect;
	for(int i = 0; i < boxes.size(); i++){
		int xmin = (int) (boxes[i][0] - xpad) / ratio;
		int ymin = (int) (boxes[i][1] - ypad) / ratio;
		int xmax = (int) (boxes[i][2] - xpad) / ratio;
		int ymax = (int) (boxes[i][3] - ypad) / ratio;

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
		std::string desc = COCO_LABELS[labels[i]] + " " + 
			std::to_string((int)(scores[i] * 100)) + "%";
		cv::putText(im, desc, cv::Point(xmin, ymin-10), 
			cv::FONT_HERSHEY_SIMPLEX, 
			font_scale, clr, 1, false);
	}
}

/*
	Plot extra data such as landmarks. keypoins, masks.
	Simple Object Detection does not any extras.
*/
void mobilint::postface::YOLOv8FacePostProcessor::plot_extras(cv::Mat& im, 
	const std::vector<std::vector<float>>& extras){

	float ratio, xpad, ypad;
	compute_ratio_pads(im, ratio, xpad, ypad);

	for(int i = 0; i < extras.size(); i++){
		cv::Point pt;
		for(int j = 0; j < m_nextra / 3; j++){
			pt.x = (int)(extras[i][3 * j + 0] - xpad) / ratio;
			pt.y = (int)(extras[i][3 * j + 1] - ypad) / ratio;

			std::array<int, 3> bgr = LMARK_COLORS[j];
			cv::Scalar clr(bgr[0], bgr[1], bgr[2]);
			cv::circle(im, pt, 3, clr, -1);
		}
	}
}

/*
	Print out given message in DEBUG mode using locks
*/
void mobilint::postface::YOLOv8FacePostProcessor::print(std::string msg){
#ifdef DEBUG
	std::lock_guard<std::mutex> lk(mPrintMutex);
	std::cout << msg << endl;
#endif
}

void mobilint::postface::YOLOv8FacePostProcessor::worker() {
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

		run_postprocess(k.npu_outs, k.npu_outs2);
		k.boxes = get_result_box();
		k.scores = get_result_score();
		k.labels = get_result_label();
		auto extra_landmarks = get_result_extra();

		plot_boxes(k.im, k.boxes, k.scores, k.labels);
		plot_extras(k.im, extra_landmarks);

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

uint64_t mobilint::postface::YOLOv8FacePostProcessor::enqueue(
	cv::Mat& im,
	std::vector<mobilint::NDArray<float>>& npu_outs, 
	std::vector<std::array<float, 4>>& boxes,
	std::vector<float>& scores, 
	std::vector<int>& labels,
	std::vector<mobilint::NDArray<float>>& npu_outs2){


	auto thres_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
	auto title = std::to_string(thres_id) + " | Postprocessor Enqueue | ";

	print(title + "Start");
	uint64_t ticket_save = 0;
	std::vector<std::vector<float>> extras;

	{
		std::lock_guard<std::mutex> lk(mMutexIn);
		mQueueIn.push(
			{ ++ticket, im, npu_outs, boxes, scores, labels, npu_outs2, extras});
		ticket_save = ticket;

		mCondIn.notify_all();  // JUST IN CASE
		print(title + "Input Queue size " + std::to_string(mQueueIn.size()));
	}

	print(title + "Finish");
	return ticket_save;
}

void mobilint::postface::YOLOv8FacePostProcessor::receive(uint64_t receipt_no) {
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
