cmake_minimum_required(VERSION 3.22.1)

project("myarm64rknn")

include_directories(${CMAKE_SOURCE_DIR}/include)

set(FFMPEG ${CMAKE_SOURCE_DIR}/ffmpeg)  # ffmpeg的路径
include_directories(${FFMPEG}/include)  # 导入ffmpeg的头文件
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -L${FFMPEG}/libs/${CMAKE_ANDROID_ARCH_ABI}")

add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        YoloDetector.cpp
        postprocess.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
        ${CMAKE_SOURCE_DIR}/libs/arm64-v8a/librknn_api.so

        EGL
        GLESv2
        android
        log)