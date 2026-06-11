cmake_minimum_required(VERSION 3.16)
project(drone_vision_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Eigen3 REQUIRED)

add_library(drone_vision_pipeline
    src/ImageTargetDetector.cpp
)

target_include_directories(drone_vision_pipeline
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(drone_vision_pipeline
    PUBLIC
        Eigen3::Eigen
        ${OpenCV_LIBS}
)

add_executable(run_detector examples/run_detector.cpp)
target_link_libraries(run_detector PRIVATE drone_vision_pipeline)
