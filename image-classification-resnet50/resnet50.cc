#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "maccel/maccel.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

namespace {

const char* RESNET50_MXQ_PATH = "resnet50.mxq";

constexpr int MAX_RATIO = 20;
constexpr float MEANS[] = {0.485, 0.456, 0.406};  // RGB
constexpr float STD[] = {0.229, 0.224, 0.225};    // RGB

using Example = std::pair<std::string, int>;

std::vector<Example> DATASET = {
    {"ILSVRC2012_val_00000001.JPEG", 65},
};

void resize_image(uint8_t* resized_img, int* resized_h, int* resized_w, uint8_t* img,
                  int h, int w, int target) {
    *resized_h = target;
    *resized_w = target;
    stbir_resize_uint8(img, w, h, 0, resized_img, *resized_w, *resized_h, 0, 3);
}

void crop_image(uint8_t* cropped_img, const uint8_t* img, int h, int w, int target_h,
                int target_w) {
    int sh = int((h - target_h) / 2.0);
    int sw = int((w - target_w) / 2.0);

    for (int y = 0; y < target_h; ++y) {
        for (int x = 0; x < target_w; ++x) {
            for (int c = 0; c < 3; ++c) {
                cropped_img[(target_w * y + x) * 3 + c] =
                    img[(w * (sh + y) + (sw + x)) * 3 + c];
            }
        }
    }
}

void rescale_image(float* rescaled_img, const uint8_t* img, int h, int w,
                   const float vs[3], const float std[3]) {
    for (int i = 0; i < h * w * 3; ++i) {
        rescaled_img[i] = ((float)img[i] / 255. - vs[i % 3]) / std[i % 3];
    }
}

template <typename T>
int argmax(const std::vector<T>& vs) {
    return std::max_element(vs.begin(), vs.end()) - vs.begin();
}

}  // namespace

int main() {
    mobilint::StatusCode sc;

    auto acc = mobilint::Accelerator::create(sc);
    if (!sc) {
        fprintf(stderr, "Failed to open device. (status code: %d)\n", int(sc));
        fprintf(stderr, "Hint: Please make sure that you have proper privilege.\n");
        fprintf(stderr, "Hint: Please make sure that the driver is correctly loaded.\n");
        exit(1);
    }

    auto model = mobilint::Model::create(RESNET50_MXQ_PATH, sc);
    if (!sc) {
        fprintf(stderr, "Failed to create a model. (status code: %d)\n", int(sc));
        exit(1);
    }
    sc = model->launch(*acc);
    if (!sc) {
        fprintf(stderr, "Failed to launch a model. (status code: %d)\n", int(sc));
    }

    auto buffer_info = model->getInputBufferInfo()[0];
    int target_h = buffer_info.original_height;
    int target_w = buffer_info.original_width;
    int target_ch = buffer_info.original_channel;

    auto resized_img = std::make_unique<uint8_t[]>(256 * 256 * 3 * MAX_RATIO);
    auto cropped_img = std::make_unique<uint8_t[]>(target_h * target_w * target_ch);
    auto input_img = std::make_unique<float[]>(target_h * target_w * target_ch);

    for (const auto& example : DATASET) {
        int h, w, resized_h, resized_w;
        uint8_t* img = stbi_load(example.first.c_str(), &w, &h, nullptr, 3);
        if (!img) {
            fprintf(stderr, "Failed to open an image file, %s.\n", example.first.c_str());
            exit(1);
        }

        resize_image(resized_img.get(), &resized_h, &resized_w, img, h, w, 256);
        crop_image(cropped_img.get(), resized_img.get(), resized_h, resized_w, target_h,
                   target_w);
        rescale_image(input_img.get(), cropped_img.get(), target_h, target_w, MEANS, STD);

        auto result = model->infer({input_img.get()}, sc);
        if (!sc) {
            fprintf(stderr, "Failed to infer an output. (error code: %d)\n",
                    static_cast<int>(sc));
            exit(1);
        }

        printf("label   : %d\n", example.second);
        printf("predict : %d\n", argmax(result[0]));

        STBI_FREE(img);
    }

    model->dispose();

    return 0;
}
