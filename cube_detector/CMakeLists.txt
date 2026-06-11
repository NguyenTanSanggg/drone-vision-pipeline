cmake_minimum_required(VERSION 3.5)
project(cube_detector)

add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV 4.10.0 REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

add_executable(cube_detector
  CubeDetector.cpp
)

target_include_directories(cube_detector PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cube_detector
  ${OpenCV_LIBS}
)

ament_target_dependencies(cube_detector
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
)

target_compile_features(cube_detector PUBLIC c_std_99 cxx_std_17)

install(TARGETS
  cube_detector
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY cfg
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()