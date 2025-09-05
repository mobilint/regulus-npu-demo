#include <maccel/maccel.h>

#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "src/config/parser.h"
#include "src/model/model.h"
#include "src/postprocess/post_builder.h"
#include "src/preprocess/preprocess.h"

using namespace mobilint::post;

std::string get_filename(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr,
                "User must provide two arguments: 1) mxq model path 2) model config yaml "
                "file path 3) image path\n");
        exit(1);
    }
    std::string mxq_path = argv[1];
    std::string yaml_path = argv[2];
    std::string img_path = argv[3];
    std::string save_path =
        img_path.substr(0, img_path.find_last_of('.')) + "_result.jpg";

    std::string model_name = get_filename(mxq_path);

    NPUModel model(mxq_path);
    auto input_shape = model.get_input_shape();
    int imh = input_shape[0];
    int imw = input_shape[1];

    ModelInfo cfg;
    parse_yaml(yaml_path, cfg);
    auto post_processor = make_postprocessor(imh, imw, cfg);

    PreProcessor pre_processor;
    cv::Mat org_img = cv::imread(img_path);
    auto preprocessed = pre_processor(org_img, cfg);

    auto output = model(std::move(preprocessed));

    std::vector<std::array<float, 4>> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<std::vector<float>> extras;

    uint64_t ticket = post_processor->enqueue(output, boxes, scores, labels, extras);
    post_processor->receive(ticket);

    post_processor->plot_results(org_img, boxes, scores, labels);
    cv::imwrite(save_path, org_img);

    return 0;
}
