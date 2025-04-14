#include <memory>
#include <vector>
#include <iostream>

#include "maccel/type.h"

#include "app_init_models.h"

InitMaccelResult init_3_models(std::string model_path1, std::string model_path2, std::string model_path3) {
    InitMaccelResult result;

    StatusCode sc;
    result.acc = Accelerator::create(sc);
    if (!sc) {
	std::cout << "Accelerator::create() failed.\n";
        exit(1);
    }

    result.mc.excludeAllCores();
    result.mc.include(Core::Core0);

    result.models.push_back(std::move(Model::create(model_path1, result.mc, sc)));
    result.models.push_back(std::move(Model::create(model_path2, result.mc, sc)));
    result.models.push_back(std::move(Model::create(model_path3, result.mc, sc)));

    for (const auto& model : result.models) {
	BufferInfo buffer_info = model->getInputBufferInfo()[0];
	const int h = buffer_info.original_height;
	const int w = buffer_info.original_width;
	const int c = buffer_info.original_channel;
	std::cout << "hwc : " << h << " " << w << " " << c << "\n";

	model->launch(*result.acc);
    }
    
    return result;
}
