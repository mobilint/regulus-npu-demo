#include "model.h"

#include <maccel/type.h>

#include <filesystem>
#include <iostream>

using namespace mobilint;

NPUModel::NPUModel(std::string modelPath) { buildModel(modelPath); }

NPUModel::~NPUModel() { release(); }

std::vector<std::vector<float>> NPUModel::operator()(std::unique_ptr<float[]> image) {
    auto result = mModel->infer({image.get()}, mSc);
    return result;
}

void NPUModel::buildModel(std::string modelPath) {
    if (!filesystem::exists(modelPath)) {
        throw runtime_error("Error: model Does not exist: " + modelPath);
    }
    mModel = Model::create(modelPath, mSc);
    mModel->launch(*mAcc.get());
}

void NPUModel::release() {
    mModel.reset();
    mAcc.reset();
}

std::vector<int> NPUModel::get_input_shape() {
    auto buffer_info = mModel->getInputBufferInfo()[0];
    int target_h = buffer_info.original_height;
    int target_w = buffer_info.original_width;
    int target_ch = buffer_info.original_channel;
    return {target_h, target_w, target_h};
}