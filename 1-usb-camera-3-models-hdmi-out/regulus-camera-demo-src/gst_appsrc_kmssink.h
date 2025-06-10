#ifndef APPSRC_KMSSINK_PIPELINE_H
#define APPSRC_KMSSINK_PIPELINE_H

#include <cstdint>

#include "gst/gstelement.h"
#include "gst_base_pipeline.h"

namespace gst_wrapper {
class AppsrcKmssinkPipeline : public virtual BasePipeline {
public:
    AppsrcKmssinkPipeline() = delete;
    AppsrcKmssinkPipeline(const char* format_in, uint32_t width_in, uint32_t height_in,
                          uint32_t fps_in);

    GstElement* appsrc;

private:
    GstElement* kmssink;

    void create_elements(const char* format_in, uint32_t width_in, uint32_t height_in,
                         uint32_t fps_in);
    void link_elements(const char* format_out, uint32_t width_out, uint32_t height_out,
                       uint32_t fps_out);
};
}  // namespace gst_wrapper

#endif
