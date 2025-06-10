#ifndef INFER_H_
#define INFER_H_

#include "app_module_infer.h"

#include <atomic>

#include "app_type.h"
#include "maccel/model.h"
#include "tsqueue.h"

namespace app {
std::thread module_infer(mobilint::Model* model, Buffer& buffer_in, Queue& queue_out,
                         std::atomic<bool>& push_on,
                         std::vector<mobilint::Buffer>& repositioned_buffer) {
    auto func = [&, model]() {
        int64_t index = 0;
        while (push_on) {
            InterThreadData data_in;
            auto qsc = buffer_in.get(data_in, index);
            if (qsc == Buffer::StatusCode::CLOSED) {
                std::cout << "buffer closed.\n";
                exit(1);
            }

            std::vector<std::vector<float>> npu_result;
            mobilint::StatusCode sc =
                model->inferBufferToFloat(repositioned_buffer, npu_result);
            if (!sc) {
                std::cout << "infer failed.\n";
                exit(1);
            }

            queue_out.push({data_in.frame.clone(), {}, std::move(npu_result)});
        }
    };

    return std::thread(func);
}
}  // namespace app

#endif
