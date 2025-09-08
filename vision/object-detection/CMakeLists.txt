cmake_minimum_required(VERSION 3.18)
project(vision_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_FLAGS "-O3 -march=armv8-a -ffast-math")
set(CMAKE_CXX_FLAGS "-O3 -march=armv8-a -ffast-math")

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(src)
add_executable(inference inference.cc)
add_executable(inference_yaml inference_yaml.cc)

target_link_libraries(inference
    PRIVATE visionlib
)

target_link_libraries(inference_yaml
    PRIVATE visionlib
)