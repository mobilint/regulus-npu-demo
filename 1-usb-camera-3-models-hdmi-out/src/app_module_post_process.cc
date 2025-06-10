#include "app_module_post_process.h"

#include <atomic>

#include "app_type.h"
#include "benchmarker.h"
#include "post_yolov8.h"
#include "post_yolov8_face.h"
#include "post_yolov8_pose.h"

namespace app {
std::thread module_post_process_face(int npu_h, int npu_w, Queue& queue_in,
                                     Queue& queue_out, std::atomic<bool>& push_on) {
    auto func = [&, npu_h, npu_w]() {
        float face_conf_thres = 0.15;
        float face_iou_thres = 0.5;
        int face_nc = 1;  // number of classes
        int face_imh = npu_h;
        int face_imw = npu_w;

        mobilint::postface::YOLOv8FacePostProcessor post(
            face_nc, face_imh, face_imw, face_conf_thres, face_iou_thres, 1, false);
        Benchmarker bc;
        while (push_on) {
            InterThreadData data_in;
            auto qsc = queue_in.pop(data_in);
            if (qsc == Queue::StatusCode::CLOSED) {
                std::cout << "queue closed.\n";
                exit(1);
            }

            std::vector<std::array<float, 4>> boxes;
            std::vector<float> scores;
            std::vector<int> labels;

            std::vector<mobilint::NDArray<float>> result;

            for (size_t i = 0; i < data_in.npu_result.size(); ++i) {
                std::vector<int64_t> shape = {(signed long)data_in.npu_result[i].size()};
                result.push_back(mobilint::NDArray(data_in.npu_result[i].data(), shape));
            }

            bc.start();
            uint64_t ticket = post.enqueue(data_in.frame, result, boxes, scores, labels);
            post.receive(ticket);
            bc.end();

            queue_out.push({data_in.frame});
        }
    };

    return std::thread(func);
}

std::thread module_post_process_od(int npu_h, int npu_w, Queue& queue_in,
                                   Queue& queue_out, std::atomic<bool>& push_on) {
    auto func = [&, npu_h, npu_w]() {
        float od_conf_thres = 0.2;  // 0.5
        float od_iou_thres = 0.15;  // 0.45
        int od_nc = 80;             // number of classes
        int od_imh = npu_h;
        int od_imw = npu_w;

        mobilint::post::YOLOv8PostProcessor post(od_nc, od_imh, od_imw, od_conf_thres,
                                                 od_iou_thres, false);

        while (push_on) {
            InterThreadData data_in;
            auto qsc = queue_in.pop(data_in);
            if (qsc == Queue::StatusCode::CLOSED) {
                std::cout << "queue closed.\n";
                exit(1);
            }

            std::vector<std::array<float, 4>> boxes;
            std::vector<float> scores;
            std::vector<int> labels;
            std::vector<std::vector<float>> keypoints;

            uint64_t ticket = post.enqueue(data_in.frame, data_in.npu_result, boxes,
                                           scores, labels, keypoints);
            post.receive(ticket);

            queue_out.push({data_in.frame});
        }
    };

    return std::thread(func);
}

std::thread module_post_process_pose(int npu_h, int npu_w, Queue& queue_in,
                                     Queue& queue_out, std::atomic<bool>& push_on) {
    auto func = [&, npu_h, npu_w]() {
        float pose_conf_thres = 0.25;
        float pose_iou_thres = 0.65;
        int pose_nc = 1;  // number of classes
        int pose_imh = npu_h;
        int pose_imw = npu_w;

        mobilint::post::YOLOv8PosePostProcessor post(
            pose_nc, pose_imh, pose_imw, pose_conf_thres, pose_iou_thres, false);

        while (push_on) {
            InterThreadData data_in;
            auto qsc = queue_in.pop(data_in);
            if (qsc == Queue::StatusCode::CLOSED) break;

            std::vector<std::array<float, 4>> boxes;
            std::vector<float> scores;
            std::vector<int> labels;
            std::vector<std::vector<float>> keypoints;
            uint64_t ticket = post.enqueue(data_in.frame, data_in.npu_result, boxes,
                                           scores, labels, keypoints);

            post.receive(ticket);
            queue_out.push({data_in.frame});
        }
    };

    return std::thread(func);
}
}  // namespace app
