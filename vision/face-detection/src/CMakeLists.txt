file(GLOB_RECURSE VISION_SRC CONFIGURE_DEPENDS
    model/*.cc
    config/*.cc
    preprocess/*.cc
    postprocess/*.cc
)

add_library(visionlib STATIC ${VISION_SRC})

target_include_directories(visionlib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/model
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${CMAKE_CURRENT_SOURCE_DIR}/preprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/postprocess
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(visionlib
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC ${OpenCV_LIBS}
    PUBLIC Threads::Threads
    PRIVATE maccel
)