#pragma once

#include <string>

struct YamlConfig {
    std::string camera_device = "";
    int camera_width;
    int camera_height;
    int camera_fps;
    int display_width;
    int display_height;
};

bool loadYamlConfig(const std::string& yaml_path, YamlConfig& cfg);
std::string findVideoDevice();