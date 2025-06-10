#ifndef BASE_PIPELINE_H
#define BASE_PIPELINE_H

#include <stdint.h>

#include <functional>

#include "glib.h"
#include "gst/gstelement.h"

namespace gst_wrapper {
class BasePipeline {
public:
    ~BasePipeline();

    GstElement* pipeline;

    void start();
    void stop();
    void startMessageLoop();

private:
    GstBus* bus;
    GstMessage* msg;
    gboolean terminate;
};

void process_data_from_appsink(GstElement* appsink,
                               std::function<void(uint8_t* data, unsigned long size)> fn);

void push_data_to_appsrc(GstElement* appsrc, uint8_t* data, unsigned long size);
}  // namespace gst_wrapper

#endif
