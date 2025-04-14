#include <thread>
#include "app_module_construct_image.h"

#include <atomic>
#include "app_type.h"
#include "app_image.h"
#include "benchmarker.h"

namespace app {
    std::thread module_construct_image(Buffer &buffer_in_org, Queue &queue_in_face, Queue &queue_in_od, Queue &queue_in_pose, Queue &queue_out, std::atomic<bool> &push_on) {
	auto func = [&]() {
	    cv::Mat background = cv::imread("./img/[Regulus] Smart Camera_Demo_250226.png", cv::IMREAD_COLOR);

	    std::vector<cv::Mat> displays;
	    for (int i = 0; i < 3; i++) {
		cv::Mat display = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC3);
		background.copyTo(display({ 0, 0, 1920, 1080 }));
		displays.push_back(display);
	    }
	
	    cv::Mat mask = img::get_mask(background);

	    cv::Mat section_org = cv::imread("./img/@Section_org.png", cv::IMREAD_UNCHANGED);
	    cv::Mat section_face = cv::imread("./img/@Section_face.png", cv::IMREAD_UNCHANGED);
	    cv::Mat section_od = cv::imread("./img/@Section_od.png", cv::IMREAD_UNCHANGED);
	    cv::Mat section_pose = cv::imread("./img/@Section_pose.png", cv::IMREAD_UNCHANGED);

	    cv::Mat section_org_alpha = img::get_alpha(section_org);
	    cv::Mat section_face_alpha = img::get_alpha(section_face);
	    cv::Mat section_od_alpha = img::get_alpha(section_od);
	    cv::Mat section_pose_alpha = img::get_alpha(section_pose);
	
	    cv::Mat section_org_bgr = cv::imread("./img/@Section_org.png", cv::IMREAD_COLOR);
	    cv::Mat section_face_bgr = cv::imread("./img/@Section_face.png", cv::IMREAD_COLOR);
	    cv::Mat section_od_bgr = cv::imread("./img/@Section_od.png", cv::IMREAD_COLOR);
	    cv::Mat section_pose_bgr = cv::imread("./img/@Section_pose.png", cv::IMREAD_COLOR);

	    int frame_i = 0;
	    double ratio = 0;
	    bool ratio_inc = true;
	    Benchmarker bc;
	
	    while (push_on) {
		if (frame_i == 3) {
		    frame_i = 0;
		}
		cv::Mat display = displays[frame_i++];	    

		int64_t index = 0;
		app::InterThreadData data_org;
		auto qsc = buffer_in_org.get(data_org, index);

		app::InterThreadData data_face, data_od, data_pose;
		queue_in_face.pop(data_face);
		queue_in_od.pop(data_od);
		queue_in_pose.pop(data_pose);

		cv::Mat frame_org = data_org.frame;
		cv::Mat frame_face = data_face.frame;
		cv::Mat frame_od = data_od.frame;
		cv::Mat frame_pose = data_pose.frame;
	    
		frame_org.copyTo(display({ 70, 212, 576, 384 }), mask({ 70, 212, 576, 384 }));	    
		frame_face.copyTo(display({ 70, 212 + 384 + 30, 576, 384 }), mask({ 70, 212, 576, 384 }));	    
		frame_od.copyTo(display({ 70 + 576 + 30, 212, 576, 384 }), mask({ 70, 212, 576, 384 }));	    
		frame_pose.copyTo(display({ 70 + 576 + 30, 212 + 384 + 30, 576, 384 }), mask({ 70, 212, 576, 384 }));

		// draw live circle
		cv::Scalar color = cv::Scalar(255, 255, 255) * (1 - ratio) + cv::Scalar(33, 33, 223) * ratio;

		if (ratio_inc) {
		    ratio += 0.066;
		    if (ratio >= 1) ratio_inc = false;
		} else {
		    ratio -= 0.066;
		    if (ratio <= 0) ratio_inc = true;
		}

		cv::Mat circle_area(cv::Size(20, 20), CV_8UC3, cv::Scalar(255, 255, 255));
		cv::circle(circle_area, cv::Point(10, 10), 7, color, cv::FILLED, cv::LINE_AA, 0);

		circle_area.copyTo(display(cv::Rect(1643 + 29, 57 + 22, 20, 20)));

		auto draw_section = [&display](cv::Mat section_bgr, cv::Mat section_alpha, int x, int y) {
		    cv::Mat section_area_org = display({ x, y, section_bgr.cols, section_bgr.rows });
		    for (int y = 0; y < section_bgr.rows; y++) {
			for (int x = 0; x < section_bgr.cols; x++) {
			    cv::Vec3b disp = section_area_org.at<cv::Vec3b>(y, x);
			    cv::Vec3b sect = section_bgr.at<cv::Vec3b>(y, x);
			    float alpha = section_alpha.at<uchar>(y, x) / 255.0f;
			
			    for (int c = 0; c < 3; c++) {
				disp[c] = disp[c] * (1 - alpha) + sect[c] * alpha;
			    }
		    
			    section_area_org.at<cv::Vec3b>(y, x) = disp;
			}
		    }		
		};

		draw_section(section_org_bgr, section_org_alpha, 70 + 30, 212 + 300);
		draw_section(section_face_bgr, section_face_alpha, 70 + 30, 212 + 384 + 30 + 300);
		draw_section(section_od_bgr, section_od_alpha, 70 + 576 + 30 + 30, 212 + 300);
		draw_section(section_pose_bgr, section_pose_alpha, 70 + 576 + 30 + 30, 212 + 384 + 30 + 300);
		
		queue_out.push({ display });
	    }
	};
    
	return std::thread(func);
    }    
}
