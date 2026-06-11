# UAV Vision Detection Pipeline

This repository contains the public computer vision module extracted from a UAV mission project.
The full UAV system was integrated with ROS/ROS2, PX4, and Gazebo, while this repository focuses only on the perception pipeline.

## Main Features

- Color-based target detection using HSV thresholding.
- Extra RGB-dominance masks for bright red, yellow, and blue targets.
- Morphological filtering and contour-based target selection.
- Center estimation and pixel error calculation.
- Optional lock gate for reducing target jumping between frames.
- Landing pad localization using gray-board segmentation and white-corner detection.
- Debug visualization with multiple processing stages.

## Project Structure

```text
.
├── CMakeLists.txt
├── README.md
├── config/
│   └── params_example.yaml
├── examples/
│   └── run_detector.cpp
├── include/
│   └── drone_vision_pipeline/
│       ├── ImageTargetDetector.hpp
│       └── VisionTypes.hpp
└── src/
    └── ImageTargetDetector.cpp
```

## Dependencies

- C++17
- OpenCV
- Eigen3
- CMake

On Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake libopencv-dev libeigen3-dev
```

## Build

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Run Example

Target color detection:

```bash
./run_detector path/to/input.jpg path/to/debug_output.jpg red
./run_detector path/to/input.jpg path/to/debug_output.jpg yellow
./run_detector path/to/input.jpg path/to/debug_output.jpg blue
```

Landing pad detection:

```bash
./run_detector path/to/landing_pad.jpg path/to/landing_debug.jpg landing
```

The program prints detection status, target center, pixel error, area, and output debug image path.

## Notes

- This repository does not include the full ROS/PX4 mission controller.
- The public version is intended to demonstrate the vision/perception module only.
- ROS topics, PX4 commands, mission state machine, and servo/landing controllers were intentionally removed to keep the repository clean and easy to review.

## Technologies

C++, Python-related UAV workflow, OpenCV, ROS/ROS2 integration in the original system, PX4, Gazebo, computer vision, contour detection, UAV perception.
