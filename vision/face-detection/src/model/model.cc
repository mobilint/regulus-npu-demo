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
