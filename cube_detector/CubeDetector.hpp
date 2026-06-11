#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

class CubeDetectorNode : public rclcpp::Node
{
public:
	CubeDetectorNode();

private:
	// ───────────────────────── Inner types ──────────────────────────
	struct VictimModel {
		bool       valid         = false;
		bool       hasCircle     = false;
		float      confidence    = 0.0f;
		cv::Rect   boxBbox;
		cv::Point2f boxCenter;
		cv::Point2f circleCenter;
		float      circleRadius  = 0.0f;
	};

	struct KalmanState {
		double x = 0.0, y = 0.0, z = 0.0;
		double cov_x = 1.0, cov_y = 1.0, cov_z = 1.0;

		void reset() { x = y = z = 0.0; cov_x = cov_y = cov_z = 1.0; }
	};

	struct LockState {
		bool        locked           = false;
		cv::Point2f center           = {0.f, 0.f};
		float       visible_size     = 0.f;
		int         missed           = 0;

		void reset() { locked = false; center = {0.f, 0.f}; visible_size = 0.f; missed = 0; }
	};

	struct LastSeenState {
		bool            has_pose      = false;
		double          x = 0.0, y = 0.0, z = 0.0;
		bool            timer_started = false;
		rclcpp::Time    lost_start    = rclcpp::Time(0, 0, RCL_ROS_TIME);

		void reset() {
			has_pose      = false;
			x = y = z     = 0.0;
			timer_started = false;
			lost_start    = rclcpp::Time(0, 0, RCL_ROS_TIME);
		}
	};

	// ─────────────────────────── Params ─────────────────────────────
	struct Params {
		// Box HSV
		int    h_min = 5,   s_min = 80,  v_min = 50;
		int    h_max = 30,  s_max = 255, v_max = 255;

		// Box area
		int    min_box_area = 500;
		int    max_box_area = 200000;

		// Circle HSV
		int    circle_h_min = 0,   circle_s_min = 0,   circle_v_min = 0;
		int    circle_h_max = 180, circle_s_max = 50,  circle_v_max = 200;

		// Circle geometry
		double circle_min_radius         = 15.0;
		double circle_max_radius         = 80.0;
		double circle_position_tolerance = 0.15;

		// Physical
		double box_width_m = 0.45;

		// Lock
		int    max_lock_missed      = 30;
		double lock_max_dist_px     = 120.0;
		double lock_min_size_ratio  = 0.5;

		// Timing
		double last_seen_hold_timeout_s = 0.5;

		// Kalman
		double kalman_q = 0.01;
		double kalman_r = 0.5;

		// Scoring weights
		double score_dist_weight        = 3.0;
		double score_size_ratio_weight  = 120.0;
		double score_area_weight        = 0.02;
		double score_fill_weight        = 30.0;
		double score_circularity_weight = 40.0;
		double score_center_weight      = 0.5;
	} _p;

	// ───────────────────────── ROS I/O ──────────────────────────────
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr      _image_sub;
	rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camera_info_sub;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr         _image_pub;
	rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr _target_pose_pub;
	rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr             _target_valid_pub;
	rclcpp::Publisher<std_msgs::msg::String>::SharedPtr           _perf_pub;
	// ─────────────────────── Camera state ───────────────────────────
	bool      _has_camera_info = false;
	cv::Mat   _camera_matrix;
	cv::Mat   _dist_coeffs;

	// ─────────────────── Detection / tracking state ─────────────────
	LockState     _lock;
	LastSeenState _last_seen;
	KalmanState   _kalman;

	bool   _has_last_pose = false;
	double _last_x = 0.0, _last_y = 0.0, _last_z = 0.0;

	// ──────────────── Cached / pre-computed resources ───────────────
	cv::Mat _morph_kernel;

	using Clock = std::chrono::steady_clock;
	Clock::time_point _perf_last_pub = Clock::now();
	uint64_t _perf_frame_count = 0;
	double _perf_total_ms = 0.0;

	// Detailed timing accumulators
	uint64_t _perf_detail_count = 0;
	double _perf_detect_ms = 0.0;
	double _perf_pose_ms = 0.0;
	double _perf_kalman_ms = 0.0;
	double _perf_publish_ms = 0.0;
	double _perf_annotate_ms = 0.0;
	std::string _perf_overlay_text;

	// ─────────────────────── Private methods ────────────────────────
	void loadParameters();

	// Callbacks
	void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
	void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);

	// Detection pipeline
	VictimModel detectVictimModel(const cv::Mat &frame);
	bool detectBoxRegion(const cv::Mat &hsv, cv::Rect &boxBbox, cv::Point2f &boxCenter);
	bool detectCircularHandle(const cv::Mat &frame, const cv::Rect &boxRegion,
	                          cv::Point2f &circleCenter, float &radius);
	bool validateVictimGeometry(const VictimModel &model);
	bool isLockedCandidateValid(const cv::Rect &bbox, const cv::Point2f &center) const;
	double scoreLockedCandidate(const cv::Point2f &center, const cv::Rect &bbox, double area) const;
	double scoreUnlockedCandidate(
		double area, double fill_ratio, double circularity,
		const cv::Point2f &center, const cv::Point2f &img_center) const;

	// Pose estimation
	void estimatePose(const VictimModel &target, int image_width, int image_height,
	                  double &x, double &y, double &z) const;

	// Kalman
	void kalmanUpdate(double measured_x, double measured_y, double measured_z);
	void kalmanPredict();

	// Publishing helpers
	geometry_msgs::msg::PoseStamped buildPoseMsg(
		const std_msgs::msg::Header &header, double x, double y, double z) const;
	void publishResults(const VictimModel &victim, const std_msgs::msg::Header &header,
	                    bool valid, double x, double y, double z);

	// State management
	void resetLockState();

	// Annotation
	void annotateImage(cv_bridge::CvImagePtr image, const VictimModel &target,
	                   double x, double y, double z,
	                   const std::string &perf_text) const;
};
