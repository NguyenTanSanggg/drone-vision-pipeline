#include "drone_vision_pipeline/ImageTargetDetector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace
{
constexpr int kDebugPanelWidth = 320;
constexpr int kDebugPanelHeight = 240;
constexpr int kDebugPanelHeaderHeight = 24;
constexpr float kPredictionDistanceWeight = 20.0f;
constexpr float kCenterDistanceWeight = 0.10f;
constexpr float kLockedAreaWeight = 0.02f;

cv::Mat convertToBgr(const cv::Mat &image)
{
    if (image.empty())
    {
        return cv::Mat::zeros(kDebugPanelHeight, kDebugPanelWidth, CV_8UC3);
    }

    if (image.channels() == 1)
    {
        cv::Mat bgrImage;
        cv::cvtColor(image, bgrImage, cv::COLOR_GRAY2BGR);
        return bgrImage;
    }

    if (image.channels() == 3)
    {
        return image.clone();
    }

    cv::Mat normalized;
    image.convertTo(normalized, CV_8U);

    cv::Mat bgrImage;
    cv::cvtColor(normalized, bgrImage, cv::COLOR_GRAY2BGR);
    return bgrImage;
}

void drawPanelFrame(cv::Mat &panel, const std::string &title)
{
    if (panel.empty())
    {
        return;
    }

    cv::rectangle(
        panel,
        cv::Rect(0, 0, panel.cols, std::min(kDebugPanelHeaderHeight, panel.rows)),
        cv::Scalar(35, 35, 35),
        cv::FILLED);

    cv::rectangle(panel, cv::Rect(0, 0, panel.cols, panel.rows), cv::Scalar(220, 220, 220), 2);

    cv::putText(
        panel,
        title,
        cv::Point(8, 17),
        cv::FONT_HERSHEY_SIMPLEX,
        0.48,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA);
}

cv::Mat makePanel(const cv::Mat &image, const std::string &title)
{
    cv::Mat bgrImage = convertToBgr(image);
    cv::Mat panel;
    cv::resize(bgrImage, panel, cv::Size(kDebugPanelWidth, kDebugPanelHeight), 0.0, 0.0, cv::INTER_AREA);
    drawPanelFrame(panel, title);
    return panel;
}

void putDebugLine(cv::Mat &panel, int lineIndex, const std::string &text)
{
    const int y = kDebugPanelHeaderHeight + 20 + lineIndex * 22;
    if (y >= panel.rows - 4)
    {
        return;
    }

    cv::putText(
        panel,
        text,
        cv::Point(10, y),
        cv::FONT_HERSHEY_SIMPLEX,
        0.48,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA);
}

cv::Mat makeTextPanel(const std::vector<std::string> &lines, const std::string &title)
{
    cv::Mat panel = cv::Mat::zeros(kDebugPanelHeight, kDebugPanelWidth, CV_8UC3);
    panel.setTo(cv::Scalar(15, 15, 15));
    drawPanelFrame(panel, title);

    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        putDebugLine(panel, static_cast<int>(i), lines[i]);
    }

    return panel;
}

std::string floatText(float value, int precision = 3)
{
    if (!std::isfinite(value))
    {
        return "nan";
    }

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    return ss.str();
}

struct CircleCandidateMetrics
{
    bool accepted{false};
    bool partial{false};
    cv::Point2f center{0.0f, 0.0f};
    float radius{0.0f};
    double area{0.0};
    double perimeter{0.0};
    double circularity{0.0};
    double fillRatio{0.0};
    double hullCircularity{0.0};
    double hullFillRatio{0.0};
    double angularCoverageDeg{0.0};
};

CircleCandidateMetrics evaluateCircleCandidate(
    const std::vector<cv::Point> &contour,
    const drone_vision_pipeline::ImageDetectorParams &params)
{
    CircleCandidateMetrics metrics{};

    if (contour.empty())
    {
        return metrics;
    }

    metrics.area = cv::contourArea(contour);
    if (metrics.area < static_cast<double>(params.minAreaPx) || metrics.area > static_cast<double>(params.maxAreaPx))
    {
        return metrics;
    }

    metrics.perimeter = cv::arcLength(contour, true);
    if (metrics.perimeter <= 1e-6)
    {
        return metrics;
    }

    cv::minEnclosingCircle(contour, metrics.center, metrics.radius);
    if (!std::isfinite(metrics.radius) || metrics.radius <= 1.0f)
    {
        return metrics;
    }

    if (metrics.radius < params.minRadiusPx || metrics.radius > params.maxRadiusPx)
    {
        return metrics;
    }

    metrics.circularity = 4.0 * M_PI * metrics.area / (metrics.perimeter * metrics.perimeter);

    const double circleArea = M_PI * static_cast<double>(metrics.radius) * static_cast<double>(metrics.radius);
    metrics.fillRatio = metrics.area / std::max(1.0, circleArea);

    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    const double hullArea = cv::contourArea(hull);
    const double hullPerimeter = cv::arcLength(hull, true);
    if (hullArea > 1.0 && hullPerimeter > 1e-6)
    {
        metrics.hullCircularity = 4.0 * M_PI * hullArea / (hullPerimeter * hullPerimeter);
        metrics.hullFillRatio = hullArea / std::max(1.0, circleArea);
    }

    constexpr int kAngleBinCount = 36;
    std::array<bool, kAngleBinCount> angleBins{};
    for (const auto &point : contour)
    {
        const double dx = static_cast<double>(point.x) - static_cast<double>(metrics.center.x);
        const double dy = static_cast<double>(point.y) - static_cast<double>(metrics.center.y);
        const double distance = std::sqrt(dx * dx + dy * dy);
        if (distance < 0.35 * static_cast<double>(metrics.radius))
        {
            continue;
        }

        double angle = std::atan2(dy, dx);
        if (angle < 0.0)
        {
            angle += 2.0 * M_PI;
        }

        int bin = static_cast<int>(std::floor(angle / (2.0 * M_PI) * static_cast<double>(kAngleBinCount)));
        bin = std::clamp(bin, 0, kAngleBinCount - 1);
        angleBins[static_cast<std::size_t>(bin)] = true;
    }

    const int coveredBins = static_cast<int>(std::count(angleBins.begin(), angleBins.end(), true));
    metrics.angularCoverageDeg = static_cast<double>(coveredBins) * 360.0 / static_cast<double>(kAngleBinCount);

    const bool fullCircle =
        metrics.circularity >= static_cast<double>(params.minCircularity) &&
        metrics.fillRatio >= static_cast<double>(params.minFillRatio);

    // Khi hinh tron bi che/cat mot phan, contour raw thuong thanh dang C nen circularity/fill raw giam.
    // Dung convex hull + do phu goc de chap nhan vong tron bi khuat, khong can them param YAML.
    const double partialMinFill = std::max(0.18, static_cast<double>(params.minFillRatio) * 0.45);
    const double partialMinCircularity = std::max(0.18, static_cast<double>(params.minCircularity) * 0.40);
    const double partialMinHullCircularity = std::max(0.48, static_cast<double>(params.minCircularity) * 0.75);
    const double partialMinHullFill = std::max(0.30, static_cast<double>(params.minFillRatio) * 0.65);

    const bool partialCircle =
        metrics.fillRatio >= partialMinFill &&
        metrics.circularity >= partialMinCircularity &&
        metrics.hullCircularity >= partialMinHullCircularity &&
        metrics.hullFillRatio >= partialMinHullFill &&
        metrics.angularCoverageDeg >= 110.0;

    metrics.partial = !fullCircle && partialCircle;
    metrics.accepted = fullCircle || partialCircle;
    return metrics;
}

cv::Mat buildDebugCanvas(const std::vector<cv::Mat> &panels)
{
    const cv::Mat emptyPanel = makeTextPanel({"empty"}, "DEBUG");
    std::vector<cv::Mat> normalizedPanels;
    normalizedPanels.reserve(6U);

    for (const auto &panel : panels)
    {
        normalizedPanels.push_back(panel.empty() ? emptyPanel.clone() : panel);
    }

    while (normalizedPanels.size() < 6U)
    {
        normalizedPanels.push_back(emptyPanel.clone());
    }

    cv::Mat row1;
    cv::Mat row2;
    cv::Mat canvas;
    cv::hconcat(std::vector<cv::Mat>{normalizedPanels[0], normalizedPanels[1], normalizedPanels[2]}, row1);
    cv::hconcat(std::vector<cv::Mat>{normalizedPanels[3], normalizedPanels[4], normalizedPanels[5]}, row2);
    cv::vconcat(row1, row2, canvas);
    return canvas;
}

cv::Point2f toPoint2f(const Eigen::Vector2f &value)
{
    return cv::Point2f(value.x(), value.y());
}

bool isFiniteVector2(const Eigen::Vector2f &value)
{
    return std::isfinite(value.x()) && std::isfinite(value.y());
}


struct LandingBoardCandidate
{
    bool valid{false};
    cv::RotatedRect rect{};
    std::vector<cv::Point> contour{};
    float areaPx{0.0f};
    float fillRatio{0.0f};
    float aspectRatio{0.0f};
    float observedSidePx{0.0f};
    float expectedSideRatio{1.0f};
    float score{-std::numeric_limits<float>::infinity()};
};

struct LandingCornerSet
{
    int count{0};
    std::array<bool, 4> found{{false, false, false, false}};
    std::array<cv::Point2f, 4> centers{};
    cv::Point2f center{0.0f, 0.0f};
};


int makeOddKernelSize(int value)
{
    int kernelSize = std::max(1, value);
    if ((kernelSize % 2) == 0)
    {
        ++kernelSize;
    }
    return kernelSize;
}

std::array<cv::Point2f, 4> rotatedRectVertices(const cv::RotatedRect &rect)
{
    std::array<cv::Point2f, 4> vertices{};
    rect.points(vertices.data());
    return vertices;
}

float rotatedRectAspectRatio(const cv::RotatedRect &rect)
{
    const float width = std::max(rect.size.width, 1e-3f);
    const float height = std::max(rect.size.height, 1e-3f);
    return std::max(width, height) / std::min(width, height);
}

float rotatedRectArea(const cv::RotatedRect &rect)
{
    return std::max(rect.size.width, 0.0f) * std::max(rect.size.height, 0.0f);
}

float rotatedRectMeanSide(const cv::RotatedRect &rect)
{
    return 0.5f * (std::max(rect.size.width, 0.0f) + std::max(rect.size.height, 0.0f));
}

float pointDistance(const cv::Point2f &a, const cv::Point2f &b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool pointInsideRotatedRect(const cv::RotatedRect &rect, const cv::Point2f &point)
{
    const std::array<cv::Point2f, 4> vertices = rotatedRectVertices(rect);
    std::vector<cv::Point2f> polygon(vertices.begin(), vertices.end());
    return cv::pointPolygonTest(polygon, point, false) >= 0.0;
}

void drawRotatedRect(cv::Mat &image, const cv::RotatedRect &rect, const cv::Scalar &color, int thickness)
{
    if (image.empty())
    {
        return;
    }

    const std::array<cv::Point2f, 4> vertices = rotatedRectVertices(rect);
    for (int i = 0; i < 4; ++i)
    {
        cv::line(image, vertices[static_cast<std::size_t>(i)], vertices[static_cast<std::size_t>((i + 1) % 4)], color, thickness, cv::LINE_AA);
    }
}

cv::Point2f contourCenter(const std::vector<cv::Point> &contour, const cv::Rect &fallbackBox)
{
    const cv::Moments moments = cv::moments(contour);
    if (std::abs(moments.m00) > 1e-6)
    {
        return cv::Point2f(
            static_cast<float>(moments.m10 / moments.m00),
            static_cast<float>(moments.m01 / moments.m00));
    }

    return cv::Point2f(
        static_cast<float>(fallbackBox.x) + static_cast<float>(fallbackBox.width) * 0.5f,
        static_cast<float>(fallbackBox.y) + static_cast<float>(fallbackBox.height) * 0.5f);
}

LandingBoardCandidate findBestLandingBoard(
    const cv::Mat &darkMask,
    const drone_vision_pipeline::VisionLandingParams &params,
    const drone_vision_pipeline::CameraIntrinsics &intrinsics,
    bool intrinsicsValid,
    float projectionRangeDownM,
    const cv::Point2f &imageCenter,
    cv::Mat &debugImage)
{
    LandingBoardCandidate best{};

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat contourMask = darkMask.clone();
    cv::findContours(contourMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto &contour : contours)
    {
        const double areaD = cv::contourArea(contour);
        if (areaD < static_cast<double>(params.boardMinAreaPx) ||
            areaD > static_cast<double>(params.boardMaxAreaPx))
        {
            continue;
        }

        const cv::RotatedRect rect = cv::minAreaRect(contour);
        if (rect.size.width < 4.0f || rect.size.height < 4.0f)
        {
            continue;
        }

        const float area = static_cast<float>(areaD);
        const float rectArea = std::max(rotatedRectArea(rect), 1.0f);
        const float fillRatio = area / rectArea;
        const float aspectRatio = rotatedRectAspectRatio(rect);
        const float observedSidePx = rotatedRectMeanSide(rect);

        if (aspectRatio < params.boardMinAspectRatio || aspectRatio > params.boardMaxAspectRatio)
        {
            continue;
        }

        if (fillRatio < params.boardMinFillRatio || fillRatio > params.boardMaxFillRatio)
        {
            continue;
        }

        float expectedSideRatio = 1.0f;
        float expectedPenalty = 0.0f;
        if (intrinsicsValid &&
            std::isfinite(projectionRangeDownM) &&
            projectionRangeDownM > 0.05f &&
            params.landingPadSizeM > 0.05f)
        {
            const float expectedSideX = intrinsics.fx * params.landingPadSizeM / projectionRangeDownM;
            const float expectedSideY = intrinsics.fy * params.landingPadSizeM / projectionRangeDownM;
            const float expectedSidePx = 0.5f * (expectedSideX + expectedSideY);

            if (std::isfinite(expectedSidePx) && expectedSidePx > 5.0f)
            {
                expectedSideRatio = observedSidePx / expectedSidePx;
                if (expectedSideRatio < params.boardExpectedSideMinRatio ||
                    expectedSideRatio > params.boardExpectedSideMaxRatio)
                {
                    continue;
                }

                expectedPenalty = std::abs(expectedSideRatio - 1.0f) * 1500.0f;
            }
        }

        const float centerDistance = pointDistance(rect.center, imageCenter);
        const float score =
            area +
            3500.0f * fillRatio -
            1800.0f * std::abs(aspectRatio - 1.0f) -
            0.25f * centerDistance -
            expectedPenalty;

        drawRotatedRect(debugImage, rect, cv::Scalar(120, 120, 255), 1);

        if (!best.valid || score > best.score)
        {
            best.valid = true;
            best.rect = rect;
            best.contour = contour;
            best.areaPx = area;
            best.fillRatio = fillRatio;
            best.aspectRatio = aspectRatio;
            best.observedSidePx = observedSidePx;
            best.expectedSideRatio = expectedSideRatio;
            best.score = score;
        }
    }

    if (best.valid)
    {
        drawRotatedRect(debugImage, best.rect, cv::Scalar(0, 255, 0), 2);
        cv::circle(debugImage, best.rect.center, 4, cv::Scalar(0, 255, 0), -1);
    }

    return best;
}

LandingCornerSet detectWhiteLandingCorners(
    const cv::Mat &brightMask,
    const LandingBoardCandidate &board,
    const drone_vision_pipeline::VisionLandingParams &params,
    cv::Mat &debugImage)
{
    LandingCornerSet cornerSet{};

    if (!board.valid)
    {
        return cornerSet;
    }

    const std::array<cv::Point2f, 4> boardVertices = rotatedRectVertices(board.rect);
    const float minDistanceFromCenter = std::max(board.observedSidePx * params.cornerMinDistanceRatio, 1.0f);

    std::array<float, 4> bestVertexDistance{{
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()}};

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat contourMask = brightMask.clone();
    cv::findContours(contourMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto &contour : contours)
    {
        const double areaD = cv::contourArea(contour);
        if (areaD < static_cast<double>(params.cornerMinAreaPx) ||
            areaD > static_cast<double>(params.cornerMaxAreaPx))
        {
            continue;
        }

        const cv::Rect box = cv::boundingRect(contour);
        if (box.width < 2 || box.height < 2)
        {
            continue;
        }

        const cv::Point2f center = contourCenter(contour, box);
        if (!pointInsideRotatedRect(board.rect, center))
        {
            continue;
        }

        if (pointDistance(center, board.rect.center) < minDistanceFromCenter)
        {
            continue;
        }

        int nearestIndex = 0;
        float nearestDistance = std::numeric_limits<float>::infinity();
        for (int i = 0; i < 4; ++i)
        {
            const float distance = pointDistance(center, boardVertices[static_cast<std::size_t>(i)]);
            if (distance < nearestDistance)
            {
                nearestDistance = distance;
                nearestIndex = i;
            }
        }

        const std::size_t slot = static_cast<std::size_t>(nearestIndex);
        if (!cornerSet.found[slot] || nearestDistance < bestVertexDistance[slot])
        {
            cornerSet.found[slot] = true;
            cornerSet.centers[slot] = center;
            bestVertexDistance[slot] = nearestDistance;
        }
    }

    cv::Point2f sum(0.0f, 0.0f);
    for (int i = 0; i < 4; ++i)
    {
        const std::size_t slot = static_cast<std::size_t>(i);
        if (!cornerSet.found[slot])
        {
            continue;
        }

        ++cornerSet.count;
        sum += cornerSet.centers[slot];
        cv::circle(debugImage, cornerSet.centers[slot], 5, cv::Scalar(255, 255, 0), -1);
        cv::line(debugImage, boardVertices[slot], cornerSet.centers[slot], cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
    }

    if (cornerSet.count > 0)
    {
        cornerSet.center = sum * (1.0f / static_cast<float>(cornerSet.count));
        cv::circle(debugImage, cornerSet.center, 4, cv::Scalar(255, 0, 255), -1);
    }

    return cornerSet;
}

cv::Point2f averageDetectedLandingCorners(const LandingCornerSet &cornerSet)
{
    cv::Point2f sum(0.0f, 0.0f);
    int usedCount = 0;

    for (int i = 0; i < 4; ++i)
    {
        const std::size_t slot = static_cast<std::size_t>(i);
        if (!cornerSet.found[slot])
        {
            continue;
        }

        sum += cornerSet.centers[slot];
        ++usedCount;
    }

    if (usedCount <= 0)
    {
        return cornerSet.center;
    }

    return sum * (1.0f / static_cast<float>(usedCount));
}

bool areOppositeLandingCornerSlots(int firstIndex, int secondIndex)
{
    return ((firstIndex + 2) % 4) == secondIndex ||
           ((secondIndex + 2) % 4) == firstIndex;
}

bool tryEstimateCenterFromOppositeCorners(
    const LandingCornerSet &cornerSet,
    cv::Point2f &estimatedCenter)
{
    for (int i = 0; i < 2; ++i)
    {
        const int oppositeIndex = i + 2;
        const std::size_t firstSlot = static_cast<std::size_t>(i);
        const std::size_t oppositeSlot = static_cast<std::size_t>(oppositeIndex);

        if (cornerSet.found[firstSlot] && cornerSet.found[oppositeSlot])
        {
            estimatedCenter = (cornerSet.centers[firstSlot] + cornerSet.centers[oppositeSlot]) * 0.5f;
            return true;
        }
    }

    return false;
}

cv::Point2f estimateCenterFromAdjacentCorners(
    const cv::Point2f &firstCorner,
    const cv::Point2f &secondCorner,
    const cv::Point2f &boardCenter)
{
    const cv::Point2f edge = secondCorner - firstCorner;
    const float edgeLength = std::max(pointDistance(firstCorner, secondCorner), 1e-3f);
    const cv::Point2f sideMidpoint = (firstCorner + secondCorner) * 0.5f;

    const cv::Point2f normalA(-edge.y / edgeLength, edge.x / edgeLength);
    const cv::Point2f normalB(edge.y / edgeLength, -edge.x / edgeLength);

    // Hai goc canh nhau tao thanh mot canh cua hinh vuong 4 goc.
    // Tam nam cach trung diem canh mot nua chieu dai canh, ve phia ben trong board.
    const cv::Point2f candidateA = sideMidpoint + normalA * (edgeLength * 0.5f);
    const cv::Point2f candidateB = sideMidpoint + normalB * (edgeLength * 0.5f);

    if (pointDistance(candidateA, boardCenter) <= pointDistance(candidateB, boardCenter))
    {
        return candidateA;
    }

    return candidateB;
}

cv::Point2f estimateLandingCenterFromCornersAndBoard(
    const LandingCornerSet &cornerSet,
    const LandingBoardCandidate &board)
{
    if (!board.valid)
    {
        return cornerSet.center;
    }

    cv::Point2f estimatedCenter(0.0f, 0.0f);

    if (cornerSet.count >= 4)
    {
        return averageDetectedLandingCorners(cornerSet);
    }

    if (cornerSet.count == 3)
    {
        // Mat 1 goc: trong 3 goc con lai luon co it nhat mot cap goc doi dien.
        // Lay trung diem cap doi dien se ra tam chinh xac hon viec tron voi dinh board xam.
        if (tryEstimateCenterFromOppositeCorners(cornerSet, estimatedCenter))
        {
            return estimatedCenter;
        }

        return averageDetectedLandingCorners(cornerSet);
    }

    if (cornerSet.count == 2)
    {
        std::vector<int> indices;
        indices.reserve(2U);

        for (int i = 0; i < 4; ++i)
        {
            const std::size_t slot = static_cast<std::size_t>(i);
            if (cornerSet.found[slot])
            {
                indices.push_back(i);
            }
        }

        if (indices.size() != 2U)
        {
            return board.rect.center;
        }

        const int firstIndex = indices[0];
        const int secondIndex = indices[1];
        const cv::Point2f firstCorner = cornerSet.centers[static_cast<std::size_t>(firstIndex)];
        const cv::Point2f secondCorner = cornerSet.centers[static_cast<std::size_t>(secondIndex)];

        if (areOppositeLandingCornerSlots(firstIndex, secondIndex))
        {
            // Con lai la 2 goc doi dien: trung diem cua 2 goc doi dien la tam.
            return (firstCorner + secondCorner) * 0.5f;
        }

        // Con lai la 2 goc canh nhau:
        // tinh tam tu canh con lai thay vi lay tam board xam truc tiep.
        return estimateCenterFromAdjacentCorners(firstCorner, secondCorner, board.rect.center);
    }

    return board.rect.center;
}

} // namespace

namespace drone_vision_pipeline
{
void ImageTargetDetector::configure(const ImageDetectorParams &params)
{
    params_ = params;
    params_.minAreaPx = std::max(1, params_.minAreaPx);
    params_.maxAreaPx = std::max(params_.minAreaPx + 1, params_.maxAreaPx);
    params_.morphKernelSize = std::max(1, params_.morphKernelSize);

    if ((params_.morphKernelSize % 2) == 0)
    {
        ++params_.morphKernelSize;
    }

    if (!std::isfinite(params_.minRadiusPx) || params_.minRadiusPx < 0.0f)
    {
        params_.minRadiusPx = 0.0f;
    }

    if (!std::isfinite(params_.maxRadiusPx) || params_.maxRadiusPx < params_.minRadiusPx)
    {
        params_.maxRadiusPx = std::max(params_.minRadiusPx, 100000.0f);
    }

    params_.redLightSMax = std::clamp(params_.redLightSMax, 0, 255);
    params_.redLightVMin = std::clamp(params_.redLightVMin, 0, 255);
    params_.redRgbMin = std::clamp(params_.redRgbMin, 0, 255);
    params_.redRgbMargin = std::clamp(params_.redRgbMargin, 0, 255);

    params_.yellowLightSMax = std::clamp(params_.yellowLightSMax, 0, 255);
    params_.yellowLightVMin = std::clamp(params_.yellowLightVMin, 0, 255);
    params_.yellowRgbMin = std::clamp(params_.yellowRgbMin, 0, 255);
    params_.yellowRgbMargin = std::clamp(params_.yellowRgbMargin, 0, 255);

    params_.blueLightSMax = std::clamp(params_.blueLightSMax, 0, 255);
    params_.blueLightVMin = std::clamp(params_.blueLightVMin, 0, 255);
    params_.blueRgbMin = std::clamp(params_.blueRgbMin, 0, 255);
    params_.blueRgbMargin = std::clamp(params_.blueRgbMargin, 0, 255);
}

ImageTargetDetection ImageTargetDetector::detect(
    const cv::Mat &bgrImage,
    const HsvRange &hsvRange,
    const CameraIntrinsics &intrinsics,
    float projectionRangeDownM,
    const ImageTargetLockInput &lockInput,
    cv::Mat *debugImage) const
{
    try
    {
        ImageTargetDetection result{};

        if (bgrImage.empty() || !isValidRange(hsvRange))
        {
            return result;
        }

        cv::Mat hsvImage;
        cv::cvtColor(bgrImage, hsvImage, cv::COLOR_BGR2HSV);

        const cv::Mat rawMask = buildMask(bgrImage, hsvImage, hsvRange);
        cv::Mat morphMask = rawMask.clone();

        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(params_.morphKernelSize, params_.morphKernelSize));

        cv::morphologyEx(morphMask, morphMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(morphMask, morphMask, cv::MORPH_CLOSE, kernel);

        // Close them mot lan bang kernel lon hon de noi lai vong tron bi dut do loang sang,
        // bong trang o tam, hoac bi che mot phan nho. Khong tao param moi de giu YAML gon.
        const int robustCloseSize = std::max(params_.morphKernelSize, params_.morphKernelSize * 2 + 1);
        const cv::Mat robustCloseKernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(robustCloseSize, robustCloseSize));
        cv::morphologyEx(morphMask, morphMask, cv::MORPH_CLOSE, robustCloseKernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(morphMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        const bool intrinsicsValid = isValidIntrinsics(intrinsics);
        const cv::Point2f imageCenter(
            intrinsicsValid ? intrinsics.cx : static_cast<float>(bgrImage.cols) * 0.5f,
            intrinsicsValid ? intrinsics.cy : static_cast<float>(bgrImage.rows) * 0.5f);

        const bool lockValid = lockInput.valid && isFiniteVector2(lockInput.predictedPixel);
        const cv::Point2f lockPixel = toPoint2f(lockInput.predictedPixel);

        cv::Mat contourDebug = bgrImage.clone();
        cv::Mat finalDebug = bgrImage.clone();
        cv::drawMarker(contourDebug, imageCenter, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 24, 2);
        cv::drawMarker(finalDebug, imageCenter, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 24, 2);

        if (lockValid)
        {
            cv::drawMarker(contourDebug, lockPixel, cv::Scalar(0, 140, 255), cv::MARKER_TILTED_CROSS, 22, 2);
            cv::circle(contourDebug, lockPixel, static_cast<int>(std::round(lockInput.lockGatePx)), cv::Scalar(0, 140, 255), 2);
        }

        if (lockInput.valid && isFiniteVector2(lockInput.releasePixel))
        {
            cv::drawMarker(contourDebug, toPoint2f(lockInput.releasePixel), cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 22, 2);
        }

        if (lockInput.valid && isFiniteVector2(lockInput.nadirPixel))
        {
            cv::drawMarker(contourDebug, toPoint2f(lockInput.nadirPixel), cv::Scalar(255, 0, 180), cv::MARKER_DIAMOND, 20, 2);
        }

        double bestScore = -std::numeric_limits<double>::infinity();
        std::vector<cv::Point> bestContour;
        CircleCandidateMetrics bestMetrics{};
        float bestLockDistancePx = 0.0f;

        for (const auto &contour : contours)
        {
            cv::drawContours(contourDebug, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(160, 160, 160), 1);

            const CircleCandidateMetrics metrics = evaluateCircleCandidate(contour, params_);
            if (!metrics.accepted)
            {
                if (metrics.radius > 1.0f)
                {
                    cv::circle(contourDebug, metrics.center, static_cast<int>(std::round(metrics.radius)), cv::Scalar(0, 0, 255), 1);
                }
                continue;
            }

            double lockDist = 0.0;
            if (lockValid)
            {
                const double dx = static_cast<double>(metrics.center.x - lockPixel.x);
                const double dy = static_cast<double>(metrics.center.y - lockPixel.y);
                lockDist = std::sqrt(dx * dx + dy * dy);

                if (lockInput.useLockGate && lockDist > static_cast<double>(lockInput.lockGatePx))
                {
                    cv::circle(contourDebug, metrics.center, static_cast<int>(std::round(metrics.radius)), cv::Scalar(0, 0, 255), 2);
                    continue;
                }
            }

            cv::circle(
                contourDebug,
                metrics.center,
                static_cast<int>(std::round(metrics.radius)),
                metrics.partial ? cv::Scalar(0, 180, 255) : cv::Scalar(255, 0, 0),
                2);

            const double centerDx = static_cast<double>(metrics.center.x - imageCenter.x);
            const double centerDy = static_cast<double>(metrics.center.y - imageCenter.y);
            const double centerDist = std::sqrt(centerDx * centerDx + centerDy * centerDy);

            const double shapeCircularity = std::max(metrics.circularity, metrics.hullCircularity);
            const double shapeFillRatio = std::max(metrics.fillRatio, metrics.hullFillRatio);
            const double partialPenalty = metrics.partial ? 120.0 : 0.0;

            double score =
                metrics.area +
                200.0 * shapeCircularity +
                100.0 * shapeFillRatio -
                partialPenalty -
                kCenterDistanceWeight * centerDist;

            if (lockValid && lockInput.usePredictionScore)
            {
                // Khi da co lock, contour gan lock pixel phai duoc uu tien hon contour lon hon.
                // Metric hull/coverage giup van bam duoc hinh tron khi bi che mat mot phan.
                score =
                    kLockedAreaWeight * metrics.area +
                    200.0 * shapeCircularity +
                    100.0 * shapeFillRatio -
                    partialPenalty -
                    kPredictionDistanceWeight * lockDist -
                    0.05 * centerDist;
            }

            if (score > bestScore)
            {
                bestScore = score;
                bestContour = contour;
                bestMetrics = metrics;
                bestLockDistancePx = static_cast<float>(lockDist);
            }
        }

        bool hasBestContour = !bestContour.empty();
        cv::Point2f center(0.0f, 0.0f);
        float radius = 0.0f;

        if (hasBestContour)
        {
            center = bestMetrics.center;
            radius = bestMetrics.radius;

            result.valid = true;
            result.centerPx = Eigen::Vector2f(center.x, center.y);
            result.imageCenterPx = Eigen::Vector2f(imageCenter.x, imageCenter.y);
            result.errorPx = Eigen::Vector2f(center.x - imageCenter.x, center.y - imageCenter.y);
            result.areaPx = static_cast<float>(bestMetrics.area);
            result.radiusPx = radius;
            result.cameraInfoValid = intrinsicsValid;
            result.lockDistancePx = bestLockDistancePx;
            result.selectedByLock = lockValid;

            if (intrinsicsValid)
            {
                result.opticalRayNorm = Eigen::Vector2f(
                    result.errorPx.x() / intrinsics.fx,
                    result.errorPx.y() / intrinsics.fy);
                result.errorNorm = result.opticalRayNorm;
            }
            else
            {
                result.errorNorm = Eigen::Vector2f(
                    result.errorPx.x() / std::max(1.0f, static_cast<float>(bgrImage.cols) * 0.5f),
                    result.errorPx.y() / std::max(1.0f, static_cast<float>(bgrImage.rows) * 0.5f));
                result.opticalRayNorm = result.errorNorm;
            }

            if (intrinsicsValid && std::isfinite(projectionRangeDownM) && projectionRangeDownM > 0.0f)
            {
                result.metricValid = true;
                result.opticalPositionValid = true;
                result.rangeDownM = projectionRangeDownM;

                const float cameraXM = result.opticalRayNorm.x() * projectionRangeDownM;
                const float cameraYM = result.opticalRayNorm.y() * projectionRangeDownM;

                result.targetOpticalM = Eigen::Vector3f(cameraXM, cameraYM, projectionRangeDownM);
                result.targetBodyXYM = Eigen::Vector2f(-cameraYM, cameraXM);
            }

            cv::drawContours(contourDebug, std::vector<std::vector<cv::Point>>{bestContour}, -1, cv::Scalar(0, 255, 0), 3);
            cv::circle(finalDebug, center, static_cast<int>(std::round(radius)), cv::Scalar(0, 255, 0), 2);
            cv::circle(finalDebug, center, 4, cv::Scalar(0, 255, 255), -1);
            cv::line(finalDebug, imageCenter, center, cv::Scalar(0, 255, 255), 2);
        }

        if (lockValid)
        {
            cv::drawMarker(finalDebug, lockPixel, cv::Scalar(0, 140, 255), cv::MARKER_TILTED_CROSS, 24, 2);
            cv::circle(finalDebug, lockPixel, static_cast<int>(std::round(lockInput.lockGatePx)), cv::Scalar(0, 140, 255), 2);
        }

        if (lockInput.valid && isFiniteVector2(lockInput.releasePixel))
        {
            cv::drawMarker(finalDebug, toPoint2f(lockInput.releasePixel), cv::Scalar(255, 0, 255), cv::MARKER_DIAMOND, 24, 2);
        }

        if (lockInput.valid && isFiniteVector2(lockInput.nadirPixel))
        {
            cv::drawMarker(finalDebug, toPoint2f(lockInput.nadirPixel), cv::Scalar(255, 0, 180), cv::MARKER_DIAMOND, 20, 2);
        }

        if (debugImage != nullptr)
        {
            cv::Mat hsvBgr;
            cv::cvtColor(hsvImage, hsvBgr, cv::COLOR_HSV2BGR);

            std::vector<std::string> metricLines;
            metricLines.push_back(std::string("valid: ") + (result.valid ? "true" : "false"));
            metricLines.push_back("contours: " + std::to_string(contours.size()));
            metricLines.push_back("best_area_px: " + floatText(result.areaPx, 1));
            metricLines.push_back("radius_px: " + floatText(result.radiusPx, 1));
            metricLines.push_back("radius_gate_px: " + floatText(params_.minRadiusPx, 1) + ".." + floatText(params_.maxRadiusPx, 1));
            metricLines.push_back(std::string("partial_circle: ") + (bestMetrics.partial ? "true" : "false"));
            metricLines.push_back("circ/fill: " + floatText(static_cast<float>(bestMetrics.circularity), 2) + ", " + floatText(static_cast<float>(bestMetrics.fillRatio), 2));
            metricLines.push_back("hull/ang: " + floatText(static_cast<float>(bestMetrics.hullCircularity), 2) + ", " + floatText(static_cast<float>(bestMetrics.angularCoverageDeg), 0));
            metricLines.push_back("err_px: " + floatText(result.errorPx.x(), 1) + ", " + floatText(result.errorPx.y(), 1));
            metricLines.push_back("body_xy_m: " + floatText(result.targetBodyXYM.x(), 3) + ", " + floatText(result.targetBodyXYM.y(), 3));
            metricLines.push_back("pred_px: " + floatText(lockInput.predictedPixel.x(), 1) + ", " + floatText(lockInput.predictedPixel.y(), 1));
            metricLines.push_back("rel_px: " + floatText(lockInput.releasePixel.x(), 1) + ", " + floatText(lockInput.releasePixel.y(), 1));
            metricLines.push_back("lock_dist_px: " + floatText(result.lockDistancePx, 1));
            metricLines.push_back(std::string("camera_info: ") + (intrinsicsValid ? "true" : "false"));

            *debugImage = buildDebugCanvas(
                {
                    makePanel(bgrImage, "01 RAW BGR"),
                    makePanel(hsvBgr, "02 BGR->HSV"),
                    makePanel(rawMask, "03 HSV MASK RAW"),
                    makePanel(morphMask, "04 MORPH OPEN/CLOSE"),
                    makePanel(contourDebug, "05 CONTOUR + LOCK"),
                    hasBestContour ? makePanel(finalDebug, "06 TARGET + FUTURE") : makeTextPanel(metricLines, "06 TARGET + FUTURE")});

            cv::Mat metricPanel = makeTextPanel(metricLines, "METRIC DEBUG");
            const cv::Rect metricRoi(kDebugPanelWidth * 2, kDebugPanelHeight, kDebugPanelWidth, kDebugPanelHeight);
            metricPanel.copyTo((*debugImage)(metricRoi));
        }

        return result;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(std::string("ImageTargetDetector::detect failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("ImageTargetDetector::detect failed: unknown exception");
    }
}

ImageTargetDetection ImageTargetDetector::detectLandingWhiteCorners(
    const cv::Mat &bgrImage,
    const VisionLandingParams &landingParams,
    const CameraIntrinsics &intrinsics,
    float projectionRangeDownM,
    cv::Mat *debugImage) const
{
    try
    {
        ImageTargetDetection result{};

        if (bgrImage.empty())
        {
            return result;
        }

        cv::Mat grayRaw;
        cv::cvtColor(bgrImage, grayRaw, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(grayRaw, grayRaw, cv::Size(3, 3), 0.0);

        cv::Mat grayImage;
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(grayRaw, grayImage);

        const int kernelSize = makeOddKernelSize(landingParams.morphKernelSize);
        const cv::Mat boardKernel = cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(kernelSize, kernelSize));

        // Board xam dung lam ROI va lam hinh hoc du phong khi mat 1-2 goc trang.
        cv::Mat darkBoardMask;
        cv::threshold(
            grayImage,
            darkBoardMask,
            std::clamp(landingParams.darkThreshold, 0, 255),
            255,
            cv::THRESH_BINARY_INV);
        cv::morphologyEx(darkBoardMask, darkBoardMask, cv::MORPH_CLOSE, boardKernel);
        cv::morphologyEx(darkBoardMask, darkBoardMask, cv::MORPH_OPEN, boardKernel);

        cv::Mat brightMask;
        cv::threshold(
            grayImage,
            brightMask,
            std::clamp(landingParams.whiteThreshold, 0, 255),
            255,
            cv::THRESH_BINARY);
        cv::morphologyEx(brightMask, brightMask, cv::MORPH_OPEN, boardKernel);
        cv::morphologyEx(brightMask, brightMask, cv::MORPH_CLOSE, boardKernel);

        const bool intrinsicsValid = isValidIntrinsics(intrinsics);
        const cv::Point2f imageCenter(
            intrinsicsValid ? intrinsics.cx : static_cast<float>(bgrImage.cols) * 0.5f,
            intrinsicsValid ? intrinsics.cy : static_cast<float>(bgrImage.rows) * 0.5f);

        cv::Mat candidateDebug = bgrImage.clone();
        cv::Mat finalDebug = bgrImage.clone();
        cv::drawMarker(candidateDebug, imageCenter, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 24, 2);
        cv::drawMarker(finalDebug, imageCenter, cv::Scalar(255, 255, 255), cv::MARKER_CROSS, 24, 2);

        const LandingBoardCandidate board = findBestLandingBoard(
            darkBoardMask,
            landingParams,
            intrinsics,
            intrinsicsValid,
            projectionRangeDownM,
            imageCenter,
            candidateDebug);

        const LandingCornerSet cornerSet = detectWhiteLandingCorners(
            brightMask,
            board,
            landingParams,
            candidateDebug);


        bool centerValid = false;
        cv::Point2f landingCenter(0.0f, 0.0f);
        std::string centerSource = "none";
        float confidence = 0.0f;

        if (board.valid)
        {
            // Khong dung param corner_min_count_for_center nua.
            // Detector tu quyet dinh theo so goc thuc te:
            //   4 goc: tam = trung binh 4 goc.
            //   3 goc: tim cap goc doi dien roi lay trung diem.
            //   2 goc: neu doi dien thi lay trung diem; neu ke nhau thi noi suy tu canh va huong ve tam board.
            //   0/1 goc: khong du thong tin hinh hoc tu goc trang, chi cho phep fallback neu board_only_valid=true.
            if (cornerSet.count >= 2)
            {
                centerValid = true;
                landingCenter = estimateLandingCenterFromCornersAndBoard(cornerSet, board);
                centerSource = cornerSet.count >= 4 ? "corners4" : "auto_partial_corners";
                confidence = std::min(0.55f + 0.10f * static_cast<float>(std::min(cornerSet.count, 4)), 0.95f);
            }
            else if (landingParams.boardOnlyValid)
            {
                centerValid = true;
                landingCenter = board.rect.center;
                centerSource = "board_only";
                confidence = 0.40f;
            }
        }

        float rangeForProjectionM = projectionRangeDownM;
        if ((!std::isfinite(rangeForProjectionM) || rangeForProjectionM <= 0.0f) &&
            intrinsicsValid &&
            board.valid &&
            board.observedSidePx > 5.0f &&
            landingParams.landingPadSizeM > 0.05f)
        {
            const float focalMean = 0.5f * (intrinsics.fx + intrinsics.fy);
            rangeForProjectionM = focalMean * landingParams.landingPadSizeM / board.observedSidePx;
        }

        if (centerValid)
        {
            result.valid = true;
            result.centerPx = Eigen::Vector2f(landingCenter.x, landingCenter.y);
            result.imageCenterPx = Eigen::Vector2f(imageCenter.x, imageCenter.y);
            result.errorPx = Eigen::Vector2f(landingCenter.x - imageCenter.x, landingCenter.y - imageCenter.y);
            result.areaPx = board.areaPx;
            result.radiusPx = board.observedSidePx * 0.5f;
            result.cameraInfoValid = intrinsicsValid;
            result.lockDistancePx = 0.0f;
            result.selectedByLock = false;

            if (intrinsicsValid)
            {
                result.opticalRayNorm = Eigen::Vector2f(
                    result.errorPx.x() / intrinsics.fx,
                    result.errorPx.y() / intrinsics.fy);
                result.errorNorm = result.opticalRayNorm;
            }
            else
            {
                result.errorNorm = Eigen::Vector2f(
                    result.errorPx.x() / std::max(1.0f, static_cast<float>(bgrImage.cols) * 0.5f),
                    result.errorPx.y() / std::max(1.0f, static_cast<float>(bgrImage.rows) * 0.5f));
                result.opticalRayNorm = result.errorNorm;
            }

            if (intrinsicsValid && std::isfinite(rangeForProjectionM) && rangeForProjectionM > 0.0f)
            {
                result.metricValid = true;
                result.opticalPositionValid = true;
                result.rangeDownM = rangeForProjectionM;

                const float cameraXM = result.opticalRayNorm.x() * rangeForProjectionM;
                const float cameraYM = result.opticalRayNorm.y() * rangeForProjectionM;

                result.targetOpticalM = Eigen::Vector3f(cameraXM, cameraYM, rangeForProjectionM);
                result.targetBodyXYM = Eigen::Vector2f(-cameraYM, cameraXM);
            }
        }

        if (board.valid)
        {
            drawRotatedRect(finalDebug, board.rect, cv::Scalar(0, 255, 0), 2);
        }

        for (int i = 0; i < 4; ++i)
        {
            const std::size_t slot = static_cast<std::size_t>(i);
            if (cornerSet.found[slot])
            {
                cv::circle(finalDebug, cornerSet.centers[slot], 6, cv::Scalar(255, 255, 0), -1);
            }
        }

        if (centerValid)
        {
            cv::circle(finalDebug, landingCenter, 7, cv::Scalar(0, 255, 255), -1);
            cv::line(finalDebug, imageCenter, landingCenter, cv::Scalar(0, 255, 255), 2);
            cv::putText(finalDebug, centerSource, landingCenter + cv::Point2f(8.0f, -8.0f), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
        }

        if (debugImage != nullptr)
        {
            std::vector<std::string> metricLines;
            metricLines.push_back(std::string("valid: ") + (result.valid ? "true" : "false"));
            metricLines.push_back("source: " + centerSource);
            metricLines.push_back("conf: " + floatText(confidence, 2));
            metricLines.push_back("pad_m: " + floatText(landingParams.landingPadSizeM, 2));
            metricLines.push_back("board: " + std::string(board.valid ? "true" : "false") + " area=" + floatText(board.areaPx, 0));
            metricLines.push_back("board_side_px: " + floatText(board.observedSidePx, 1));
            metricLines.push_back("side_ratio: " + floatText(board.expectedSideRatio, 2));
            metricLines.push_back("range_m: " + floatText(rangeForProjectionM, 2));
            metricLines.push_back("corners: " + std::to_string(cornerSet.count) + "/auto");
            metricLines.push_back("dark/white: " + std::to_string(std::clamp(landingParams.darkThreshold, 0, 255)) + "/" +
                std::to_string(std::clamp(landingParams.whiteThreshold, 0, 255)));
            metricLines.push_back("err_px: " + floatText(result.errorPx.x(), 1) + ", " + floatText(result.errorPx.y(), 1));
            metricLines.push_back("body_xy_m: " + floatText(result.targetBodyXYM.x(), 3) + ", " + floatText(result.targetBodyXYM.y(), 3));
            metricLines.push_back(std::string("camera_info: ") + (intrinsicsValid ? "true" : "false"));

            cv::Mat brightCornerDebug = convertToBgr(brightMask);
            cv::Mat darkBoardDebug = convertToBgr(darkBoardMask);
            if (board.valid)
            {
                drawRotatedRect(brightCornerDebug, board.rect, cv::Scalar(0, 255, 0), 2);
                drawRotatedRect(darkBoardDebug, board.rect, cv::Scalar(0, 255, 0), 2);
            }

            for (int i = 0; i < 4; ++i)
            {
                const std::size_t slot = static_cast<std::size_t>(i);
                if (cornerSet.found[slot])
                {
                    cv::circle(brightCornerDebug, cornerSet.centers[slot], 5, cv::Scalar(255, 255, 0), -1);
                }
            }

            cv::Mat metricPanel = makeTextPanel(metricLines, "CORNER LAND DEBUG");
            *debugImage = buildDebugCanvas({
                makePanel(bgrImage, "01 INPUT BGR"),
                makePanel(grayImage, "02 CLAHE GRAY"),
                makePanel(darkBoardDebug, "03 BOARD ROI"),
                makePanel(brightCornerDebug, "04 WHITE CORNERS"),
                makePanel(candidateDebug, "05 BOARD+CORNERS"),
                makePanel(finalDebug, "06 FINAL")});

            const cv::Rect metricRoi(kDebugPanelWidth * 2, kDebugPanelHeight, kDebugPanelWidth, kDebugPanelHeight);
            if (metricRoi.x + metricRoi.width <= debugImage->cols && metricRoi.y + metricRoi.height <= debugImage->rows)
            {
                metricPanel.copyTo((*debugImage)(metricRoi));
            }
        }

        return result;
    }
    catch (const std::exception &exception)
    {
        throw std::runtime_error(std::string("ImageTargetDetector::detectLandingWhiteCorners failed: ") + exception.what());
    }
    catch (...)
    {
        throw std::runtime_error("ImageTargetDetector::detectLandingWhiteCorners failed: unknown exception");
    }
}


cv::Mat ImageTargetDetector::buildMask(
    const cv::Mat &bgrImage,
    const cv::Mat &hsvImage,
    const HsvRange &hsvRange) const
{
    cv::Mat mask = buildHsvMask(hsvImage, hsvRange);

    // Tang cuong RGB theo tung nhom mau.
    // HSV chinh van giu S cao de bat mau thuong + mau toi.
    // RGB dominance chi them vung chay sang/gan trang, co dieu kien kenh mau troi hon
    // de tranh an nham nen trang/xam.
    if (params_.redLightRgbEnable && isRedHueRange(hsvRange))
    {
        const cv::Mat lightRedMask = buildLightRedRgbMask(bgrImage, hsvImage);
        cv::bitwise_or(mask, lightRedMask, mask);
    }

    if (params_.yellowLightRgbEnable && isYellowHueRange(hsvRange))
    {
        const cv::Mat lightYellowMask = buildLightYellowRgbMask(bgrImage, hsvImage);
        cv::bitwise_or(mask, lightYellowMask, mask);
    }

    if (params_.blueLightRgbEnable && isBlueHueRange(hsvRange))
    {
        const cv::Mat lightBlueMask = buildLightBlueRgbMask(bgrImage, hsvImage);
        cv::bitwise_or(mask, lightBlueMask, mask);
    }

    return mask;
}

cv::Mat ImageTargetDetector::buildHsvMask(const cv::Mat &hsvImage, const HsvRange &hsvRange) const
{
    cv::Mat mask;

    const int hMin = std::clamp(hsvRange.min[0], 0, 179);
    const int sMin = std::clamp(hsvRange.min[1], 0, 255);
    const int vMin = std::clamp(hsvRange.min[2], 0, 255);
    const int hMax = std::clamp(hsvRange.max[0], 0, 179);
    const int sMax = std::clamp(hsvRange.max[1], 0, 255);
    const int vMax = std::clamp(hsvRange.max[2], 0, 255);

    if (hMin <= hMax)
    {
        cv::inRange(hsvImage, cv::Scalar(hMin, sMin, vMin), cv::Scalar(hMax, sMax, vMax), mask);
        return mask;
    }

    cv::Mat lowMask;
    cv::Mat highMask;
    cv::inRange(hsvImage, cv::Scalar(0, sMin, vMin), cv::Scalar(hMax, sMax, vMax), lowMask);
    cv::inRange(hsvImage, cv::Scalar(hMin, sMin, vMin), cv::Scalar(179, sMax, vMax), highMask);
    cv::bitwise_or(lowMask, highMask, mask);
    return mask;
}

cv::Mat ImageTargetDetector::buildLightRedRgbMask(
    const cv::Mat &bgrImage,
    const cv::Mat &hsvImage) const
{
    if (bgrImage.empty() || hsvImage.empty())
    {
        return cv::Mat();
    }

    std::vector<cv::Mat> hsvChannels;
    std::vector<cv::Mat> bgrChannels;
    cv::split(hsvImage, hsvChannels);
    cv::split(bgrImage, bgrChannels);

    if (hsvChannels.size() < 3U || bgrChannels.size() < 3U)
    {
        return cv::Mat::zeros(hsvImage.rows, hsvImage.cols, CV_8UC1);
    }

    const int sMax = std::clamp(params_.redLightSMax, 0, 255);
    const int vMin = std::clamp(params_.redLightVMin, 0, 255);
    const int redMin = std::clamp(params_.redRgbMin, 0, 255);
    const int margin = std::clamp(params_.redRgbMargin, 0, 255);

    cv::Mat lowSaturationMask;
    cv::Mat brightMask;
    cv::Mat redMinMask;
    cv::compare(hsvChannels[1], sMax, lowSaturationMask, cv::CMP_LE);
    cv::compare(hsvChannels[2], vMin, brightMask, cv::CMP_GE);
    cv::compare(bgrChannels[2], redMin, redMinMask, cv::CMP_GE);

    // RGB dominance: R phai lon hon G/B mot bien nho.
    // Nen trang/xam thuong co R≈G≈B nen se bi loai, con vung do chay sang van con R nhinh hon.
    cv::Mat red16;
    cv::Mat green16;
    cv::Mat blue16;
    bgrChannels[2].convertTo(red16, CV_16S);
    bgrChannels[1].convertTo(green16, CV_16S);
    bgrChannels[0].convertTo(blue16, CV_16S);

    cv::Mat redMinusGreen = red16 - green16;
    cv::Mat redMinusBlue = red16 - blue16;
    cv::Mat redDominatesGreen;
    cv::Mat redDominatesBlue;
    cv::compare(redMinusGreen, cv::Scalar(margin), redDominatesGreen, cv::CMP_GE);
    cv::compare(redMinusBlue, cv::Scalar(margin), redDominatesBlue, cv::CMP_GE);

    cv::Mat redDominanceMask;
    cv::bitwise_and(redDominatesGreen, redDominatesBlue, redDominanceMask);
    cv::bitwise_and(redDominanceMask, redMinMask, redDominanceMask);

    // Hue gan do duoc uu tien, nhung khong bat buoc tuyet doi vi vung chay sang co S thap
    // lam Hue kem on dinh. RGB dominance moi la dieu kien chinh.
    cv::Mat hueRedLow;
    cv::Mat hueRedHigh;
    cv::Mat hueRedMask;
    cv::inRange(hsvImage, cv::Scalar(0, 0, vMin), cv::Scalar(15, 255, 255), hueRedLow);
    cv::inRange(hsvImage, cv::Scalar(165, 0, vMin), cv::Scalar(179, 255, 255), hueRedHigh);
    cv::bitwise_or(hueRedLow, hueRedHigh, hueRedMask);

    cv::Mat lightRedMask;
    cv::bitwise_and(lowSaturationMask, brightMask, lightRedMask);
    cv::bitwise_and(lightRedMask, redDominanceMask, lightRedMask);

    cv::Mat hueSupportedMask;
    cv::bitwise_and(hueRedMask, redDominanceMask, hueSupportedMask);
    cv::bitwise_and(hueSupportedMask, brightMask, hueSupportedMask);

    cv::bitwise_or(lightRedMask, hueSupportedMask, lightRedMask);
    return lightRedMask;
}


cv::Mat ImageTargetDetector::buildLightYellowRgbMask(
    const cv::Mat &bgrImage,
    const cv::Mat &hsvImage) const
{
    if (bgrImage.empty() || hsvImage.empty())
    {
        return cv::Mat();
    }

    std::vector<cv::Mat> hsvChannels;
    std::vector<cv::Mat> bgrChannels;
    cv::split(hsvImage, hsvChannels);
    cv::split(bgrImage, bgrChannels);

    if (hsvChannels.size() < 3U || bgrChannels.size() < 3U)
    {
        return cv::Mat::zeros(hsvImage.rows, hsvImage.cols, CV_8UC1);
    }

    const int sMax = std::clamp(params_.yellowLightSMax, 0, 255);
    const int vMin = std::clamp(params_.yellowLightVMin, 0, 255);
    const int yellowMin = std::clamp(params_.yellowRgbMin, 0, 255);
    const int margin = std::clamp(params_.yellowRgbMargin, 0, 255);

    cv::Mat lowSaturationMask;
    cv::Mat brightMask;
    cv::Mat redMinMask;
    cv::Mat greenMinMask;
    cv::compare(hsvChannels[1], sMax, lowSaturationMask, cv::CMP_LE);
    cv::compare(hsvChannels[2], vMin, brightMask, cv::CMP_GE);
    cv::compare(bgrChannels[2], yellowMin, redMinMask, cv::CMP_GE);
    cv::compare(bgrChannels[1], yellowMin, greenMinMask, cv::CMP_GE);

    // Yellow dominance: R va G deu cao, dong thoi B thap hon R/G mot bien nho.
    // Nen trang/xam co R≈G≈B nen se bi loai; vang chay sang van con B thap hon.
    cv::Mat red16;
    cv::Mat green16;
    cv::Mat blue16;
    bgrChannels[2].convertTo(red16, CV_16S);
    bgrChannels[1].convertTo(green16, CV_16S);
    bgrChannels[0].convertTo(blue16, CV_16S);

    cv::Mat redMinusBlue = red16 - blue16;
    cv::Mat greenMinusBlue = green16 - blue16;
    cv::Mat redDominatesBlue;
    cv::Mat greenDominatesBlue;
    cv::compare(redMinusBlue, cv::Scalar(margin), redDominatesBlue, cv::CMP_GE);
    cv::compare(greenMinusBlue, cv::Scalar(margin), greenDominatesBlue, cv::CMP_GE);

    cv::Mat yellowDominanceMask;
    cv::bitwise_and(redDominatesBlue, greenDominatesBlue, yellowDominanceMask);
    cv::bitwise_and(yellowDominanceMask, redMinMask, yellowDominanceMask);
    cv::bitwise_and(yellowDominanceMask, greenMinMask, yellowDominanceMask);

    cv::Mat hueYellowMask;
    cv::inRange(hsvImage, cv::Scalar(15, 0, vMin), cv::Scalar(45, 255, 255), hueYellowMask);

    cv::Mat lightYellowMask;
    cv::bitwise_and(lowSaturationMask, brightMask, lightYellowMask);
    cv::bitwise_and(lightYellowMask, yellowDominanceMask, lightYellowMask);

    cv::Mat hueSupportedMask;
    cv::bitwise_and(hueYellowMask, yellowDominanceMask, hueSupportedMask);
    cv::bitwise_and(hueSupportedMask, brightMask, hueSupportedMask);

    cv::bitwise_or(lightYellowMask, hueSupportedMask, lightYellowMask);
    return lightYellowMask;
}

cv::Mat ImageTargetDetector::buildLightBlueRgbMask(
    const cv::Mat &bgrImage,
    const cv::Mat &hsvImage) const
{
    if (bgrImage.empty() || hsvImage.empty())
    {
        return cv::Mat();
    }

    std::vector<cv::Mat> hsvChannels;
    std::vector<cv::Mat> bgrChannels;
    cv::split(hsvImage, hsvChannels);
    cv::split(bgrImage, bgrChannels);

    if (hsvChannels.size() < 3U || bgrChannels.size() < 3U)
    {
        return cv::Mat::zeros(hsvImage.rows, hsvImage.cols, CV_8UC1);
    }

    const int sMax = std::clamp(params_.blueLightSMax, 0, 255);
    const int vMin = std::clamp(params_.blueLightVMin, 0, 255);
    const int blueMin = std::clamp(params_.blueRgbMin, 0, 255);
    const int margin = std::clamp(params_.blueRgbMargin, 0, 255);

    cv::Mat lowSaturationMask;
    cv::Mat brightMask;
    cv::Mat blueMinMask;
    cv::compare(hsvChannels[1], sMax, lowSaturationMask, cv::CMP_LE);
    cv::compare(hsvChannels[2], vMin, brightMask, cv::CMP_GE);
    cv::compare(bgrChannels[0], blueMin, blueMinMask, cv::CMP_GE);

    // Blue dominance: B phai lon hon R/G mot bien nho.
    // Nen trang/xam co B≈G≈R nen se bi loai; xanh chay sang van con B nhinh hon.
    cv::Mat red16;
    cv::Mat green16;
    cv::Mat blue16;
    bgrChannels[2].convertTo(red16, CV_16S);
    bgrChannels[1].convertTo(green16, CV_16S);
    bgrChannels[0].convertTo(blue16, CV_16S);

    cv::Mat blueMinusRed = blue16 - red16;
    cv::Mat blueMinusGreen = blue16 - green16;
    cv::Mat blueDominatesRed;
    cv::Mat blueDominatesGreen;
    cv::compare(blueMinusRed, cv::Scalar(margin), blueDominatesRed, cv::CMP_GE);
    cv::compare(blueMinusGreen, cv::Scalar(margin), blueDominatesGreen, cv::CMP_GE);

    cv::Mat blueDominanceMask;
    cv::bitwise_and(blueDominatesRed, blueDominatesGreen, blueDominanceMask);
    cv::bitwise_and(blueDominanceMask, blueMinMask, blueDominanceMask);

    cv::Mat hueBlueMask;
    cv::inRange(hsvImage, cv::Scalar(85, 0, vMin), cv::Scalar(140, 255, 255), hueBlueMask);

    cv::Mat lightBlueMask;
    cv::bitwise_and(lowSaturationMask, brightMask, lightBlueMask);
    cv::bitwise_and(lightBlueMask, blueDominanceMask, lightBlueMask);

    cv::Mat hueSupportedMask;
    cv::bitwise_and(hueBlueMask, blueDominanceMask, hueSupportedMask);
    cv::bitwise_and(hueSupportedMask, brightMask, hueSupportedMask);

    cv::bitwise_or(lightBlueMask, hueSupportedMask, lightBlueMask);
    return lightBlueMask;
}

bool ImageTargetDetector::isRedHueRange(const HsvRange &hsvRange) const
{
    const int hMin = std::clamp(hsvRange.min[0], 0, 179);
    const int hMax = std::clamp(hsvRange.max[0], 0, 179);

    return hMin > hMax || hMin >= 155 || hMax <= 20;
}

bool ImageTargetDetector::isYellowHueRange(const HsvRange &hsvRange) const
{
    const int hMin = std::clamp(hsvRange.min[0], 0, 179);
    const int hMax = std::clamp(hsvRange.max[0], 0, 179);

    if (hMin > hMax)
    {
        return false;
    }

    return hMin <= 45 && hMax >= 15;
}

bool ImageTargetDetector::isBlueHueRange(const HsvRange &hsvRange) const
{
    const int hMin = std::clamp(hsvRange.min[0], 0, 179);
    const int hMax = std::clamp(hsvRange.max[0], 0, 179);

    if (hMin > hMax)
    {
        return false;
    }

    return hMin <= 140 && hMax >= 85;
}

bool ImageTargetDetector::isValidRange(const HsvRange &hsvRange) const
{
    for (int i = 0; i < 3; ++i)
    {
        if (hsvRange.min[i] < 0 || hsvRange.max[i] < 0)
        {
            return false;
        }
    }

    return true;
}

bool ImageTargetDetector::isValidIntrinsics(const CameraIntrinsics &intrinsics) const
{
    return intrinsics.valid &&
        intrinsics.width > 0 &&
        intrinsics.height > 0 &&
        std::isfinite(intrinsics.fx) &&
        std::isfinite(intrinsics.fy) &&
        std::isfinite(intrinsics.cx) &&
        std::isfinite(intrinsics.cy) &&
        intrinsics.fx > 1.0f &&
        intrinsics.fy > 1.0f;
}
} // namespace drone_vision_pipeline
