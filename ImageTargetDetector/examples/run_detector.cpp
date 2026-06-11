#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "drone_vision_pipeline/ImageTargetDetector.hpp"

namespace
{
drone_vision_pipeline::HsvRange hsvForColor(const std::string &color)
{
    using drone_vision_pipeline::HsvRange;

    if (color == "red")
    {
        return HsvRange{{170, 70, 90}, {16, 255, 255}};
    }
    if (color == "yellow")
    {
        return HsvRange{{14, 20, 120}, {38, 220, 255}};
    }
    if (color == "blue")
    {
        return HsvRange{{85, 45, 80}, {115, 255, 255}};
    }

    std::cerr << "Unknown color: " << color << ". Fallback to red.\n";
    return HsvRange{{170, 70, 90}, {16, 255, 255}};
}

void printUsage(const char *program)
{
    std::cerr << "Usage:\n"
              << "  " << program << " <input_image> <output_debug_image> <red|yellow|blue|landing>\n\n"
              << "Examples:\n"
              << "  " << program << " sample.jpg debug.jpg red\n"
              << "  " << program << " landing_pad.jpg landing_debug.jpg landing\n";
}
} // namespace

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        printUsage(argv[0]);
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];
    const std::string mode = argv[3];

    cv::Mat image = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Cannot read input image: " << inputPath << "\n";
        return 2;
    }

    using namespace drone_vision_pipeline;

    ImageTargetDetector detector;
    ImageDetectorParams detectorParams;
    detector.configure(detectorParams);

    CameraIntrinsics intrinsics;
    intrinsics.valid = false;
    intrinsics.width = image.cols;
    intrinsics.height = image.rows;

    cv::Mat debugImage;
    ImageTargetDetection detection;

    if (mode == "landing")
    {
        VisionLandingParams landingParams;
        detection = detector.detectLandingWhiteCorners(
            image,
            landingParams,
            intrinsics,
            0.0f,
            &debugImage);
    }
    else
    {
        ImageTargetLockInput lockInput;
        HsvRange range = hsvForColor(mode);
        detection = detector.detect(
            image,
            range,
            intrinsics,
            0.0f,
            lockInput,
            &debugImage);
    }

    if (!debugImage.empty())
    {
        cv::imwrite(outputPath, debugImage);
    }

    std::cout << "valid=" << (detection.valid ? "true" : "false")
              << " center_px=(" << detection.centerPx.x() << ", " << detection.centerPx.y() << ")"
              << " error_px=(" << detection.errorPx.x() << ", " << detection.errorPx.y() << ")"
              << " area_px=" << detection.areaPx
              << " radius_px=" << detection.radiusPx
              << " debug=" << outputPath << "\n";

    return detection.valid ? 0 : 3;
}
