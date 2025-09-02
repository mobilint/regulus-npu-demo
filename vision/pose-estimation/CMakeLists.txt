cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-O3 -march=armv8-a -ffast-math")
set(CMAKE_CXX_FLAGS "-O3 -march=armv8-a -ffast-math")

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(src)
add_executable(inference inference.cc)

target_include_directories(inference PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(inference PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(inference
    PRIVATE visionlib
    PRIVATE OpenMP::OpenMP_CXX
    PRIVATE ${OpenCV_LIBS}
)