#include <functional>
#include <stdint.h>
#include <iostream>

#include "glib.h"
#include "gst/gstutils.h"
#include "gst/gstelement.h"
#include "gst/app/gstappsrc.h"
#include "gst/app/gstappsink.h"

#include "gst_base_pipeline.h"

namespace gst_wrapper {
    BasePipeline::~BasePipeline() {
	/* Free resources */
	gst_object_unref(bus);
	gst_object_unref(GST_OBJECT(pipeline));
    }
    
    void BasePipeline::start() {
	/* Start pipeline */
	gst_element_set_state(pipeline, GST_STATE_PLAYING);
    }

    void BasePipeline::stop() {
	/* Stop pipeline */
	gst_element_set_state(pipeline, GST_STATE_NULL);
    }

    void BasePipeline::startMessageLoop() {
	/* Wait until error or EOS */
	bus = gst_element_get_bus (pipeline);
	while (!terminate) {
	    msg =
		gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE,
					    (GstMessageType) (GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

	    /* Parse message */
	    if (msg != NULL) {
		GError *err;
		gchar *debug_info;

		switch (GST_MESSAGE_TYPE (msg)) {
		case GST_MESSAGE_ERROR:
		    gst_message_parse_error (msg, &err, &debug_info);
		    g_printerr ("Error received from element %s: %s\n",
				GST_OBJECT_NAME (msg->src), err->message);
		    g_printerr ("Debugging information: %s\n",
				debug_info ? debug_info : "none");
		    g_clear_error (&err);
		    g_free (debug_info);
		    terminate = TRUE;
		    break;
		case GST_MESSAGE_EOS:
		    g_print ("End-Of-Stream reached.\n");
		    terminate = TRUE;
		    break;
		case GST_MESSAGE_STATE_CHANGED:
		    /* We are only interested in state-changed messages from the pipeline */
		    if (GST_MESSAGE_SRC (msg) == GST_OBJECT (pipeline)) {
			GstState old_state, new_state, pending_state;
			gst_message_parse_state_changed (msg, &old_state, &new_state, &pending_state);
			g_print ("Pipeline state changed from %s to %s:\n",
				 gst_element_state_get_name (old_state), gst_element_state_get_name (new_state));
		    }
		    break;
		default:
		    /* We should not reach here because we only asked for ERRORs and EOS */
		    g_printerr ("Unexpected message received.\n");
		    break;
		}
		gst_message_unref (msg);
	    }
	}
    }

    void process_data_from_appsink(GstElement *appsink, std::function<void(uint8_t *data, unsigned long size)> fn) {
	GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK_CAST(appsink));
	if (sample) {
	    GstBuffer *buffer = gst_sample_get_buffer(sample);
	    GstMapInfo info;
	    gst_buffer_map(buffer, &info, GST_MAP_READ);

	    /* Process data from appsink here */
	    fn(info.data, info.size);

	    /* Free resources */
	    gst_buffer_unmap(buffer, &info);
	    gst_sample_unref(sample);
	} else {
	    std::cout << "pull sample from appsink failed.\n";	    
	}
    }

    void push_data_to_appsrc(GstElement *appsrc, uint8_t *data, unsigned long size) {
	/* Allocate a buffer */
	GstBuffer* buffer;
	buffer = gst_buffer_new_allocate(NULL, size, NULL);

	GstMapInfo map;
	gst_buffer_map(buffer, &map, GST_MAP_WRITE);

	// Copy data to the buffer
	memcpy(map.data, data, size);
	gst_buffer_unmap(buffer, &map);

	gst_app_src_push_buffer(GST_APP_SRC_CAST(appsrc), buffer);
    }
}
