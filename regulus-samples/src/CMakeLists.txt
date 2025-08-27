
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE VISION_SRC
    model/*.cc
    preprocess/*.cc
    postprocess/*.cc
)

add_library(visionlib STATIC ${VISION_SRC})

target_include_directories(visionlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/postprocess
)

target_link_libraries(visionlib
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC ${OpenCV_LIBS}
    maccel
)

target_include_directories(visionlib PUBLIC ${OpenCV_INCLUDE_DIRS})