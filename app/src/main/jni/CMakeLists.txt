project(nanodetncnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.8.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20230816-android/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(nanodetncnn SHARED nanodetncnn.cpp nanodet.cpp)

target_link_libraries(nanodetncnn ncnn ${OpenCV_LIBS} mediandk)
