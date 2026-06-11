#pragma once

#include <opencv2/core.hpp>

#include "drone_vision_pipeline/VisionTypes.hpp"

namespace drone_vision_pipeline
{
class ImageTargetDetector
{
public:
    void configure(const ImageDetectorParams &params);

    // Mo ta:
    //     Detect vong tron/khoi theo HSV va CameraInfo.
    // Input:
    //     bgrImage: anh BGR tu camera.
    //     hsvRange: nguong HSV min/max cua diem F hien tai.
    //     intrinsics: fx/fy/cx/cy lay tu /camera_down/camera_info.
    //     projectionRangeDownM: do cao/chieu sau uoc luong theo huong Down, don vi met.
    //     lockInput: diem pixel du doan tu FutureTargetPredictor de lock target.
    // Logic:
    //     - Threshold HSV, morphology, find contour.
    //     - Neu co lockInput.valid thi uu tien contour gan predictedPixel.
    //     - Neu useLockGate=true thi loai contour ngoai gate de tranh nhay target.
    // Output:
    //     ImageTargetDetection va anh debug gom raw/hsv/mask/morph/contour/output.
    ImageTargetDetection detect(
        const cv::Mat &bgrImage,
        const HsvRange &hsvRange,
        const CameraIntrinsics &intrinsics,
        float projectionRangeDownM,
        const ImageTargetLockInput &lockInput,
        cv::Mat *debugImage = nullptr) const;

    /**
     * Mo ta:
     *     Detect bai dap bang board xam + 4 goc trang, khong dung chu/tam H.
     *
     * Input:
     *     bgrImage: anh BGR tu camera.
     *     landingParams: tham so threshold/hinh hoc/bo loc board/corner.
     *     intrinsics: fx/fy/cx/cy lay tu CameraInfo.
     *     projectionRangeDownM: do cao/chieu sau uoc luong theo huong Down, don vi met.
     *
     * Logic:
     *     - Chuyen BGR -> Gray + CLAHE.
     *     - Tach board xam de gioi han ROI va lay hinh hoc tam vuong.
     *     - Tach cac blob trang o 4 goc.
     *     - Neu du 4 goc: tinh tam tu 4 goc.
     *     - Neu mat 1-2 goc: suy ra tam bang board xam va cac goc con lai.
     *     - Khong detect chu H, khong lay tam tu H.
     *
     * Output:
     *     ImageTargetDetection va debug image neu debugImage != nullptr.
     */
    ImageTargetDetection detectLandingWhiteCorners(
        const cv::Mat &bgrImage,
        const VisionLandingParams &landingParams,
        const CameraIntrinsics &intrinsics,
        float projectionRangeDownM,
        cv::Mat *debugImage = nullptr) const;

private:
    ImageDetectorParams params_{};

    cv::Mat buildMask(const cv::Mat &bgrImage, const cv::Mat &hsvImage, const HsvRange &hsvRange) const;
    cv::Mat buildHsvMask(const cv::Mat &hsvImage, const HsvRange &hsvRange) const;
    cv::Mat buildLightRedRgbMask(const cv::Mat &bgrImage, const cv::Mat &hsvImage) const;
    cv::Mat buildLightYellowRgbMask(const cv::Mat &bgrImage, const cv::Mat &hsvImage) const;
    cv::Mat buildLightBlueRgbMask(const cv::Mat &bgrImage, const cv::Mat &hsvImage) const;
    bool isRedHueRange(const HsvRange &hsvRange) const;
    bool isYellowHueRange(const HsvRange &hsvRange) const;
    bool isBlueHueRange(const HsvRange &hsvRange) const;
    bool isValidRange(const HsvRange &hsvRange) const;
    bool isValidIntrinsics(const CameraIntrinsics &intrinsics) const;
    cv::Mat buildShapeMaskGray(const cv::Mat &bgrImage, cv::Mat *enhancedGray = nullptr) const;
};
} // namespace drone_vision_pipeline
