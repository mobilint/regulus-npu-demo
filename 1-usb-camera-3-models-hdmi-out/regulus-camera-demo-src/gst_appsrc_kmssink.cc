#include "gst_appsrc_kmssink.h"

#include <cstdint>

#include "gst/app/gstappsrc.h"
#include "gst/gstpipeline.h"
#include "gst/gstutils.h"
#include "gst/gstvalue.h"

namespace gst_wrapper {
AppsrcKmssinkPipeline::AppsrcKmssinkPipeline(const char* format_in, uint32_t width_in,
                                             uint32_t height_in, uint32_t fps_in) {
    create_elements(format_in, width_in, height_in, fps_in);
    link_elements(format_in, width_in, height_in, fps_in);
}

void AppsrcKmssinkPipeline::create_elements(const char* format_in, uint32_t width_in,
                                            uint32_t height_in, uint32_t fps_in) {
    /* create appsrc */
    appsrc = gst_element_factory_make("appsrc", "appsrc");
    if (!appsrc) {
        g_print("Failed to create element of type 'appsrc'\n");
        exit(-1);
    }

    g_object_set(appsrc, "max-buffers", 1, "block", TRUE, "emit-signals", FALSE, NULL);

    GstCaps* caps_appsrc_v4l2convert;
    caps_appsrc_v4l2convert = gst_caps_new_simple(
        "video/x-raw", "format", G_TYPE_STRING, format_in, "width", G_TYPE_INT, width_in,
        "height", G_TYPE_INT, height_in, "framerate", GST_TYPE_FRACTION, fps_in, 1, NULL);

    if (!caps_appsrc_v4l2convert) {
        g_print("Failed to create caps_appsrc_v4l2convert\n");
        exit(-1);
    }

    gst_app_src_set_caps(GST_APP_SRC_CAST(appsrc), caps_appsrc_v4l2convert);
    gst_caps_unref(caps_appsrc_v4l2convert);

    /* create kmssink */
    kmssink = gst_element_factory_make("kmssink", "kmssink");
    if (!kmssink) {
        g_print("Failed to create element 'kmssink'\n");
        exit(-1);
    }

    g_object_set(kmssink, "driver-name", "regulus", "force-modesetting", TRUE, "sync",
                 FALSE, NULL);

    /* create pipeline */
    pipeline = gst_pipeline_new("pipeline");

    /* add elements to the pipeline */
    gst_bin_add_many(GST_BIN(pipeline), appsrc, kmssink, NULL);
}

void AppsrcKmssinkPipeline::link_elements(const char* format_out, uint32_t width_out,
                                          uint32_t height_out, uint32_t fps_out) {
    /* link appsrc, kmssink */
    GstCaps* caps_appsrc_kmssink;
    caps_appsrc_kmssink =
        gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, format_out, "width",
                            G_TYPE_INT, width_out, "height", G_TYPE_INT, height_out,
                            "framerate", GST_TYPE_FRACTION, fps_out, 1, NULL);

    if (!caps_appsrc_kmssink) {
        g_print("Failed to create caps_appsrc_kmssink\n");
        exit(-1);
    }

    if (!gst_element_link_filtered(appsrc, kmssink, caps_appsrc_kmssink)) {
        g_warning("Failed to link appsrc and kmssink!");
        exit(-1);
    }

    gst_caps_unref(caps_appsrc_kmssink);
}
}  // namespace gst_wrapper
