#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>

#include "app_parse_argv.h"

template <typename F>
static ParseResult_dvec_w_h_f parse_dvec_w_h_f(int argc, char* argv[], int dvec_num, F usage_info_fn) {
    std::vector<std::string> dev_paths;
    uint32_t camera_width = 640, camera_height = 480, camera_fps = 30;
    std::unordered_map<std::string, VisitCount> option_visit_counts {
	{ "-d", {} },
	{ "-w", {} },
	{ "-h", {} },
	{ "-f", {} }
    };

    enum class error_type {
	WRONG_USAGE
    };

    std::string state;
    
    try {
	for (int i = 1; i < argc; i++) {
	    std::string arg = argv[i];

	    if (option_visit_counts.contains(arg)) {
		if (option_visit_counts[arg].visited == true) {
		    // error: duplicated -option visit
		    throw error_type::WRONG_USAGE;
		} else {
		    option_visit_counts[arg].visited = true;
		    state = arg;
		}
	    } else {
		if (state == std::string("-d")) {
		    dev_paths.push_back(std::string("/dev/video") + std::string(argv[i]));
		    
		} else if (state == std::string("-w")) {
		    camera_width = atoi(argv[i]);
		} else if (state == std::string("-h")) {
		    camera_height = atoi(argv[i]);
		} else if (state == std::string("-f")) {
		    camera_fps = atoi(argv[i]);
		} else {
		    throw error_type::WRONG_USAGE;
		}
		option_visit_counts[state].count++;
	    }
	}

	if (option_visit_counts[std::string("-d")].count != dvec_num) {
	    throw error_type::WRONG_USAGE;
	}
    } catch (error_type e) {
	usage_info_fn();
	exit(1);
    }

    return { dev_paths, camera_width, camera_height, camera_fps };
}

static void printUsage_dvec1_w_h_f() {
    std::cout << "Usage: demo\n";
    std::cout << "    -d 4, USB camera device number \n";
    std::cout << "    -w 640, camera width\n";
    std::cout << "    -h 480, camera height\n";
    std::cout << "    -f 30, camera FPS\n";
}

ParseResult_dvec_w_h_f parse_dvec1_w_h_f(int argc, char* argv[]) {
    return parse_dvec_w_h_f(argc, argv, 1, printUsage_dvec1_w_h_f);
}

static void printUsage_dvec3_w_h_f() {
    std::cout << "Usage: demo\n";
    std::cout << "    -d 0 2 3, three USB camera device numbers \n";
    std::cout << "    -w 640, camera width\n";
    std::cout << "    -h 480, camera height\n";
    std::cout << "    -f 30, camera FPS\n";
}

ParseResult_dvec_w_h_f parse_dvec3_w_h_f(int argc, char* argv[]) {
    return parse_dvec_w_h_f(argc, argv, 3, printUsage_dvec3_w_h_f);
}

static void printUsage_dvec4_w_h_f() {
    std::cout << "Usage: demo\n";
    std::cout << "    -d 0 2 3 4, four USB camera device numbers \n";
    std::cout << "    -w 640, camera width\n";
    std::cout << "    -h 480, camera height\n";
    std::cout << "    -f 30, camera FPS\n";
}

ParseResult_dvec_w_h_f parse_dvec4_w_h_f(int argc, char* argv[]) {
    return parse_dvec_w_h_f(argc, argv, 4, printUsage_dvec4_w_h_f);
}
