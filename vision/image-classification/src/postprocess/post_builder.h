#pragma once
#include <cctype>
#include <iostream>
#include <memory>
#include <string>

#include "postprocess.h"
#include "types.h"

#if __has_include("ssd_post.h")
#include "ssd_post.h"
#define HAS_SSD_POST 1
#else
#define HAS_SSD_POST 0
#endif

#if __has_include("yolo_anchor_post.h")
#include "yolo_anchor_post.h"
#define HAS_YOLO_ANCHOR_POST 1
#else
#define HAS_YOLO_ANCHOR_POST 0
#endif

#if __has_include("yolo_anchorless_post.h")
#include "yolo_anchorless_post.h"
#define HAS_YOLO_ANCHORLESS_POST 1
#else
#define HAS_YOLO_ANCHORLESS_POST 0
#endif

#if __has_include("yolo_anchor_seg_post.h")
#include "yolo_anchor_seg_post.h"
#define HAS_YOLO_ANCHOR_SEG_POST 1
#else
#define HAS_YOLO_ANCHOR_SEG_POST 0
#endif

#if __has_include("yolo_anchorless_seg_post.h")
#include "yolo_anchorless_seg_post.h"
#define HAS_YOLO_ANCHORLESS_SEG_POST 1
#else
#define HAS_YOLO_ANCHORLESS_SEG_POST 0
#endif

#if __has_include("yolo_anchorless_pose_post.h")
#include "yolo_anchorless_pose_post.h"
#define HAS_YOLO_ANCHORLESS_POSE_POST 1
#else
#define HAS_YOLO_ANCHORLESS_POSE_POST 0
#endif

#if __has_include("yolo_anchorless_face_post.h")
#include "yolo_anchorless_face_post.h"
#define HAS_YOLO_ANCHORLESS_FACE_POST 1
#else
#define HAS_YOLO_ANCHORLESS_FACE_POST 0
#endif

#if __has_include("cls_post.h")
#include "cls_post.h"
#define HAS_CLASSIFICATION_POST 1
#else
#define HAS_CLASSIFICATION_POST 0
#endif

using namespace mobilint::post;

inline bool has_anchors(const PostProcessInfo& p) {
    if (p.anchors.empty()) return false;
    for (const auto& layer : p.anchors) {
        if (!layer.empty()) return true;
    }
    return false;
}

inline std::unique_ptr<PostProcessor> make_postprocessor(int imh, int imw,
                                                         const ModelInfo& cfg) {
    const auto& post_info = cfg.m_postprocess;
    const bool anchors = has_anchors(post_info);

    auto to_lower = [](std::string s) {
        for (auto& ch : s) ch = (char)std::tolower((unsigned char)ch);
        return s;
    };
    const auto post_type = to_lower(post_info.type);

    if (post_info.task == Task::CLS) {
#if HAS_CLASSIFICATION_POST
        return std::make_unique<CLSPostProcessor>(imh, imw, cfg);
#else
        std::cerr << "[ERROR] Classification not supported (header missing)\n";
        return nullptr;
#endif
    }

    if (post_type == "ssd") {
#if HAS_SSD_POST
        return std::make_unique<SSDPostProcessor>(imh, imw, cfg);
#else
        std::cerr << "[ERROR] SSD not supported (header missing)\n";
        return nullptr;
#endif

    } else if (post_type == "yolo") {
        switch (post_info.task) {
        case Task::DET:
            if (anchors) {
#if HAS_YOLO_ANCHOR_POST
                return std::make_unique<YOLOAnchorPostProcessor>(imh, imw, cfg);
#else
                std::cerr << "[ERROR] DET + anchors not supported (header missing)\n";
                return nullptr;
#endif
            } else {
#if HAS_YOLO_ANCHORLESS_POST
                return std::make_unique<YOLOAnchorlessPostProcessor>(imh, imw, cfg);
#else
                std::cerr << "[ERROR] DET (anchorless) not supported (header missing)\n";
                return nullptr;
#endif
            }

        case Task::SEG:
            if (anchors) {
#if HAS_YOLO_ANCHOR_SEG_POST
                return std::make_unique<YOLOAnchorSegPostProcessor>(imh, imw, cfg);
#else
                std::cerr << "[ERROR] SEG + anchors not supported (header missing)\n";
                return nullptr;
#endif
            } else {
#if HAS_YOLO_ANCHORLESS_SEG_POST
                return std::make_unique<YOLOAnchorlessSegPostProcessor>(imh, imw, cfg);
#else
                std::cerr << "[ERROR] SEG (anchorless) not supported (header missing)\n";
                return nullptr;
#endif
            }

        case Task::FACE:
            if (anchors) {
                std::cerr << "[ERROR] FACE + anchors not implemented\n";
                return nullptr;
            } else {
#if HAS_YOLO_ANCHORLESS_FACE_POST
                return std::make_unique<YOLOAnchorlessFacePostProcessor>(imh, imw, cfg);
#else
                std::cerr << "[ERROR] FACE (anchorless) not supported (header missing)\n";
                return nullptr;
#endif
            }
        case Task::POSE:
            if (anchors) {
                std::cerr << "[ERROR] POSE + anchors not implemented\n";
                return nullptr;
            } else {
#if HAS_YOLO_ANCHORLESS_POSE_POST
                return std::make_unique<YOLOAnchorlessPosePostProcessor>(imh, imw, cfg);
#else
                std::cerr << "[ERROR] POSE (anchorless) not supported (header missing)\n";
                return nullptr;
#endif
            }

        default:
            std::cerr << "[ERROR] Task not supported\n";
            return nullptr;
        }
    } else {
        std::cerr << "[ERROR] Unknown post type\n";
        return nullptr;
    }
}
