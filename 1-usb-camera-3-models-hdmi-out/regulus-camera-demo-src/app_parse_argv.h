#ifndef PARSE_ARGV_H_
#define PARSE_ARGV_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct ParseResult_dvec_w_h_f {
    std::vector<std::string> dev_paths;
    uint32_t camera_width;
    uint32_t camera_heigth;
    uint32_t camera_fps;
};

struct VisitCount {
    bool visited = false;
    int count = 0;
};

ParseResult_dvec_w_h_f parse_dvec1_w_h_f(int argc, char* argv[]);
ParseResult_dvec_w_h_f parse_dvec3_w_h_f(int argc, char* argv[]);
ParseResult_dvec_w_h_f parse_dvec4_w_h_f(int argc, char* argv[]);

#endif
