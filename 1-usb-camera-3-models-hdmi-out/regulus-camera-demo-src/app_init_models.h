#ifndef INIT_MODELS_H_
#define INIT_MODELS_H_

#include <memory>
#include <vector>

#include <maccel/maccel.h>
#include <maccel/type.h>

using mobilint::Accelerator;
using mobilint::BufferInfo;
using mobilint::Cluster;
using mobilint::Core;
using mobilint::Model;
using mobilint::ModelConfig;
using mobilint::StatusCode;

struct InitMaccelResult {
    std::unique_ptr<Accelerator> acc;
    ModelConfig mc;
    std::vector<std::unique_ptr<Model>> models;
};

InitMaccelResult init_3_models(std::string model_path1, std::string model_path2, std::string model_path3);

#endif
