#pragma once

#include <array>

#include <Eigen/Core>

namespace drone_vision_pipeline
{

// OpenCV HSV range: H [0..179], S [0..255], V [0..255].
// If min.H > max.H, the detector treats the hue range as wrapped.
struct HsvRange
{
    std::array<int, 3> min{0, 0, 0};
    std::array<int, 3> max{179, 255, 255};
};

struct CameraIntrinsics
{
    bool valid{false};
    int width{0};
    int height{0};
    float fx{0.0f};
    float fy{0.0f};
    float cx{0.0f};
    float cy{0.0f};
};

struct ImageDetectorParams
{
    int minAreaPx{250};
    int maxAreaPx{250000};
    float minRadiusPx{7.0f};
    float maxRadiusPx{260.0f};
    float minCircularity{0.55f};
    float minFillRatio{0.45f};
    int morphKernelSize{5};

    // Extra RGB-dominance masks for bright / low-saturation target colors.
    bool redLightRgbEnable{true};
    int redLightSMax{145};
    int redLightVMin{145};
    int redRgbMin{120};
    int redRgbMargin{12};

    bool yellowLightRgbEnable{true};
    int yellowLightSMax{150};
    int yellowLightVMin{145};
    int yellowRgbMin{120};
    int yellowRgbMargin{10};

    bool blueLightRgbEnable{true};
    int blueLightSMax{165};
    int blueLightVMin{120};
    int blueRgbMin{100};
    int blueRgbMargin{10};
};

struct VisionLandingParams
{
    bool enable{true};

    // Real landing pad size in meters. Used only when camera intrinsics and range are available.
    float landingPadSizeM{1.50f};

    // Landing detector based on gray board + white corners.
    int darkThreshold{130};
    int whiteThreshold{185};

    int boardMinAreaPx{800};
    int boardMaxAreaPx{260000};
    float boardMinAspectRatio{0.65f};
    float boardMaxAspectRatio{1.55f};
    float boardMinFillRatio{0.45f};
    float boardMaxFillRatio{1.05f};
    float boardExpectedSideMinRatio{0.30f};
    float boardExpectedSideMaxRatio{2.50f};
    bool boardOnlyValid{false};

    int cornerMinAreaPx{8};
    int cornerMaxAreaPx{9000};
    float cornerMinDistanceRatio{0.25f};

    int morphKernelSize{5};

    // These fields are kept for documentation/config compatibility with the original project.
    float imageTimeoutSec{0.50f};
    float descendGateM{0.25f};
    float descendStableSec{0.35f};
    float vzMps{0.30f};
    bool usePx4LandCommandFinal{false};
    float px4LandCommandAltitudeM{0.20f};
    bool debugEnable{true};
};

struct ImageTargetLockInput
{
    bool valid{false};
    bool useLockGate{true};
    bool usePredictionScore{true};
    Eigen::Vector2f predictedPixel{0.0f, 0.0f};
    Eigen::Vector2f releasePixel{0.0f, 0.0f};
    Eigen::Vector2f nadirPixel{0.0f, 0.0f};
    float lockGatePx{90.0f};
};

struct ImageTargetDetection
{
    bool valid{false};

    Eigen::Vector2f centerPx{0.0f, 0.0f};
    Eigen::Vector2f imageCenterPx{0.0f, 0.0f};
    Eigen::Vector2f errorPx{0.0f, 0.0f};

    Eigen::Vector2f errorNorm{0.0f, 0.0f};
    bool cameraInfoValid{false};
    Eigen::Vector2f opticalRayNorm{0.0f, 0.0f};

    bool opticalPositionValid{false};
    Eigen::Vector3f targetOpticalM{0.0f, 0.0f, 0.0f};

    bool metricValid{false};
    Eigen::Vector2f targetBodyXYM{0.0f, 0.0f};
    float rangeDownM{0.0f};

    float areaPx{0.0f};
    float radiusPx{0.0f};
    float lockDistancePx{0.0f};
    bool selectedByLock{false};
};

} // namespace drone_vision_pipeline
