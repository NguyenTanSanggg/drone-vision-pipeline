#include "CubeDetector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

#include <sensor_msgs/image_encodings.hpp>

// ═══════════════════════════════════════════════════════════════════
//  Constructor
// ═══════════════════════════════════════════════════════════════════
CubeDetectorNode::CubeDetectorNode()
	: Node("cube_detector_node")
{
	loadParameters();

	// Pre-compute morphology kernel once
	_morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	auto qos = rclcpp::QoS(1).best_effort();

	_image_sub = create_subscription<sensor_msgs::msg::Image>(
		"/camera/image", qos,
		std::bind(&CubeDetectorNode::image_callback, this, std::placeholders::_1));

	_camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
		"/camera/camera_info", qos,
		std::bind(&CubeDetectorNode::camera_info_callback, this, std::placeholders::_1));

	_image_pub         = create_publisher<sensor_msgs::msg::Image>("/cube_detector/debug_image", qos);
	_target_pose_pub   = create_publisher<geometry_msgs::msg::PoseStamped>("/target_pose", qos);
	_target_valid_pub  = create_publisher<std_msgs::msg::Bool>("/target_valid", qos);
	_perf_pub          = create_publisher<std_msgs::msg::String>("/cube_detector/performance", 10);

	RCLCPP_INFO(get_logger(), "===== CubeDetectorNode started =====");
	RCLCPP_INFO(get_logger(), "box_width_m       : %.3f", _p.box_width_m);
	RCLCPP_INFO(get_logger(), "min_box_area      : %d",   _p.min_box_area);
	RCLCPP_INFO(get_logger(), "circle_min_radius : %.1f", _p.circle_min_radius);
	RCLCPP_INFO(get_logger(), "circle_max_radius : %.1f", _p.circle_max_radius);
	RCLCPP_INFO(get_logger(), "max_lock_missed   : %d",   _p.max_lock_missed);
	RCLCPP_INFO(get_logger(), "kalman q=%.4f r=%.4f",     _p.kalman_q, _p.kalman_r);
}

// ═══════════════════════════════════════════════════════════════════
//  Parameter loading  (declare + get in one call)
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::loadParameters()
{
	// Box HSV
	_p.h_min = declare_parameter<int>("h_min", _p.h_min);
	_p.s_min = declare_parameter<int>("s_min", _p.s_min);
	_p.v_min = declare_parameter<int>("v_min", _p.v_min);
	_p.h_max = declare_parameter<int>("h_max", _p.h_max);
	_p.s_max = declare_parameter<int>("s_max", _p.s_max);
	_p.v_max = declare_parameter<int>("v_max", _p.v_max);

	// Box area
	_p.min_box_area = declare_parameter<int>("min_box_area", _p.min_box_area);
	_p.max_box_area = declare_parameter<int>("max_box_area", _p.max_box_area);

	// Circle HSV
	_p.circle_h_min = declare_parameter<int>("circle_h_min", _p.circle_h_min);
	_p.circle_s_min = declare_parameter<int>("circle_s_min", _p.circle_s_min);
	_p.circle_v_min = declare_parameter<int>("circle_v_min", _p.circle_v_min);
	_p.circle_h_max = declare_parameter<int>("circle_h_max", _p.circle_h_max);
	_p.circle_s_max = declare_parameter<int>("circle_s_max", _p.circle_s_max);
	_p.circle_v_max = declare_parameter<int>("circle_v_max", _p.circle_v_max);

	// Circle geometry
	_p.circle_min_radius         = declare_parameter<double>("circle_min_radius",         _p.circle_min_radius);
	_p.circle_max_radius         = declare_parameter<double>("circle_max_radius",         _p.circle_max_radius);
	_p.circle_position_tolerance = declare_parameter<double>("circle_position_tolerance", _p.circle_position_tolerance);

	// Physical / lock
	_p.box_width_m             = declare_parameter<double>("box_width_m",             _p.box_width_m);
	_p.max_lock_missed         = declare_parameter<int>   ("max_lock_missed",         _p.max_lock_missed);
	_p.lock_max_dist_px        = declare_parameter<double>("lock_max_dist_px",        _p.lock_max_dist_px);
	_p.lock_min_size_ratio     = declare_parameter<double>("lock_min_size_ratio",     _p.lock_min_size_ratio);
	_p.last_seen_hold_timeout_s= declare_parameter<double>("last_seen_hold_timeout_s",_p.last_seen_hold_timeout_s);

	// Kalman
	_p.kalman_q = declare_parameter<double>("kalman_q", _p.kalman_q);
	_p.kalman_r = declare_parameter<double>("kalman_r", _p.kalman_r);

	// Scoring weights
	_p.score_dist_weight        = declare_parameter<double>("score_dist_weight",        _p.score_dist_weight);
	_p.score_size_ratio_weight  = declare_parameter<double>("score_size_ratio_weight",  _p.score_size_ratio_weight);
	_p.score_area_weight        = declare_parameter<double>("score_area_weight",        _p.score_area_weight);
	_p.score_fill_weight        = declare_parameter<double>("score_fill_weight",        _p.score_fill_weight);
	_p.score_circularity_weight = declare_parameter<double>("score_circularity_weight", _p.score_circularity_weight);
	_p.score_center_weight      = declare_parameter<double>("score_center_weight",      _p.score_center_weight);
}

// ═══════════════════════════════════════════════════════════════════
//  Camera info callback
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
	if (_has_camera_info) return;

	_camera_matrix = cv::Mat(3, 3, CV_64F, const_cast<double *>(msg->k.data())).clone();

	_dist_coeffs = msg->d.empty()
		? cv::Mat::zeros(5, 1, CV_64F)
		: cv::Mat(static_cast<int>(msg->d.size()), 1, CV_64F,
		          const_cast<double *>(msg->d.data())).clone();

	_has_camera_info = true;
	RCLCPP_INFO(get_logger(), "Camera intrinsics received: fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
		msg->k[0], msg->k[4], msg->k[2], msg->k[5]);
}

// ═══════════════════════════════════════════════════════════════════
//  Image callback  (main loop, kept thin — delegates to helpers)
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
	if (!_has_camera_info) {
		RCLCPP_WARN_ONCE(get_logger(), "Waiting for camera info...");
		return;
	}

	cv_bridge::CvImagePtr cv_ptr;
	try {
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	} catch (const cv_bridge::Exception &e) {
		RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
		return;
	}

	const auto perf_start = std::chrono::steady_clock::now();
	const rclcpp::Time now_ts =
		(msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
			? now()
			: rclcpp::Time(msg->header.stamp);

	double pose_ms = 0.0;
	double kalman_ms = 0.0;

	// ── Detection ──────────────────────────────────────────────────
	const auto start_detect = std::chrono::steady_clock::now();
	VictimModel victim = detectVictimModel(cv_ptr->image);
	const auto end_detect = std::chrono::steady_clock::now();
	const double detect_ms =
		std::chrono::duration_cast<std::chrono::microseconds>(end_detect - start_detect).count() * 1e-3;

	bool   publish_valid = false;
	double pub_x = 0.0, pub_y = 0.0, pub_z = 0.0;

	if (victim.valid) {
		// Update lock state
		_lock.center       = victim.boxCenter;
		_lock.visible_size = static_cast<float>(std::max(victim.boxBbox.width, victim.boxBbox.height));
		_lock.missed       = 0;
		_lock.locked       = true;

		// Raw pose → Kalman update
		double raw_x, raw_y, raw_z;
		const auto start_pose = std::chrono::steady_clock::now();
		estimatePose(victim, cv_ptr->image.cols, cv_ptr->image.rows, raw_x, raw_y, raw_z);
		const auto end_pose = std::chrono::steady_clock::now();
		pose_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_pose - start_pose).count() * 1e-3;

		const auto start_kalman = std::chrono::steady_clock::now();
		kalmanUpdate(raw_x, raw_y, raw_z);
		const auto end_kalman = std::chrono::steady_clock::now();
		kalman_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_kalman - start_kalman).count() * 1e-3;

		pub_x = _kalman.x;
		pub_y = _kalman.y;
		pub_z = _kalman.z;

		// Persist filtered pose
		_last_x = pub_x; _last_y = pub_y; _last_z = pub_z;
		_has_last_pose = true;

		// Reset last-seen window
		_last_seen.has_pose      = true;
		_last_seen.x             = pub_x;
		_last_seen.y             = pub_y;
		_last_seen.z             = pub_z;
		_last_seen.timer_started = false;
		_last_seen.lost_start    = rclcpp::Time(0, 0, RCL_ROS_TIME);

		publish_valid = true;

		RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
			"[BOX VALID] box=(%d,%d,%d,%d) raw=(%.3f,%.3f,%.3f) filt=(%.3f,%.3f,%.3f) m",
			victim.boxBbox.x, victim.boxBbox.y, victim.boxBbox.width, victim.boxBbox.height,
			raw_x, raw_y, raw_z, pub_x, pub_y, pub_z);
	}
	else {
		// Predict Kalman even without a measurement (keeps state smooth)
		const auto start_kalman = std::chrono::steady_clock::now();
		kalmanPredict();
		const auto end_kalman = std::chrono::steady_clock::now();
		kalman_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_kalman - start_kalman).count() * 1e-3;

		_lock.missed++;

		// Start lost timer on first miss
		if (!_last_seen.timer_started) {
			_last_seen.timer_started = true;
			_last_seen.lost_start    = now_ts;
		}

		const double lost_s = (now_ts - _last_seen.lost_start).seconds();
		const bool   can_hold = _last_seen.has_pose && (lost_s <= _p.last_seen_hold_timeout_s);

		if (can_hold) {
			pub_x = _last_seen.x;
			pub_y = _last_seen.y;
			pub_z = _last_seen.z;

			_last_x = pub_x; _last_y = pub_y; _last_z = pub_z;
			_has_last_pose = true;

			publish_valid = true;

			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
				"[LAST_SEEN HOLD] lost=%.2fs pose=(%.3f,%.3f,%.3f) m",
				lost_s, pub_x, pub_y, pub_z);
		}
		else {
			if (_lock.missed > _p.max_lock_missed) {
				resetLockState();
			}

			RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
				"[SEARCHING] area=%d box=(%d,%d,%d,%d) missed=%d locked=%d",
				victim.boxBbox.area(),
				victim.boxBbox.x, victim.boxBbox.y, victim.boxBbox.width, victim.boxBbox.height,
				_lock.missed, static_cast<int>(_lock.locked));
		}
	}

	// ── Publish ────────────────────────────────────────────────────
	const auto start_publish = std::chrono::steady_clock::now();
	publishResults(victim, msg->header, publish_valid, pub_x, pub_y, pub_z);
	const auto end_publish = std::chrono::steady_clock::now();
	const double publish_ms =
		std::chrono::duration_cast<std::chrono::microseconds>(end_publish - start_publish).count() * 1e-3;

	// ── Debug image ────────────────────────────────────────────────
	const auto start_annotate = std::chrono::steady_clock::now();
	annotateImage(cv_ptr, victim,
	              _has_last_pose ? _last_x : 0.0,
	              _has_last_pose ? _last_y : 0.0,
	              _has_last_pose ? _last_z : 0.0,
	              _perf_overlay_text);
	const auto end_annotate = std::chrono::steady_clock::now();
	const double annotate_ms =
		std::chrono::duration_cast<std::chrono::microseconds>(end_annotate - start_annotate).count() * 1e-3;

	_image_pub->publish(*cv_ptr->toImageMsg());

	const auto perf_end = std::chrono::steady_clock::now();
	const double frame_ms =
		std::chrono::duration_cast<std::chrono::microseconds>(perf_end - perf_start).count() * 1e-3;
	_perf_total_ms += frame_ms;
	_perf_frame_count += 1;

	_perf_detect_ms += detect_ms;
	_perf_pose_ms += (victim.valid ? pose_ms : 0.0);
	_perf_kalman_ms += kalman_ms;
	_perf_publish_ms += publish_ms;
	_perf_annotate_ms += annotate_ms;
	_perf_detail_count += 1;

	if (perf_end - _perf_last_pub >= std::chrono::seconds(1)) {
		const double wall_s =
			std::chrono::duration<double>(perf_end - _perf_last_pub).count();
		const double out_fps = (wall_s > 1e-6 && _perf_frame_count > 0)
			? static_cast<double>(_perf_frame_count) / wall_s
			: 0.0;
		const double avg_ms = _perf_frame_count > 0
			? _perf_total_ms / static_cast<double>(_perf_frame_count)
			: 0.0;
		const double cap_fps = avg_ms > 1e-6 ? 1000.0 / avg_ms : 0.0;

		std::ostringstream fps_text;
		fps_text << "FPS=" << std::fixed << std::setprecision(1) << out_fps
		         << " cap=" << std::setprecision(0) << cap_fps;
		std_msgs::msg::String perf_msg;
		perf_msg.data = fps_text.str();
		_perf_pub->publish(perf_msg);
		_perf_overlay_text = perf_msg.data;

		_perf_last_pub = perf_end;
		_perf_frame_count = 0;
		_perf_total_ms = 0.0;
		_perf_detect_ms = 0.0;
		_perf_pose_ms = 0.0;
		_perf_kalman_ms = 0.0;
		_perf_publish_ms = 0.0;
		_perf_annotate_ms = 0.0;
		_perf_detail_count = 0;
	}
}

// ═══════════════════════════════════════════════════════════════════
//  Detection pipeline
// ═══════════════════════════════════════════════════════════════════
CubeDetectorNode::VictimModel CubeDetectorNode::detectVictimModel(const cv::Mat &frame)
{
	VictimModel result;

	cv::Mat hsv;
	cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

	if (!detectBoxRegion(hsv, result.boxBbox, result.boxCenter)) {
		return result;  // result.valid == false
	}

	result.valid      = true;
	result.confidence = 0.85f;
	return result;
}

bool CubeDetectorNode::detectBoxRegion(const cv::Mat &hsv, cv::Rect &boxBbox, cv::Point2f &boxCenter)
{
	// ── Color mask ─────────────────────────────────────────────────
	cv::Mat mask;
	cv::inRange(hsv,
		cv::Scalar(_p.h_min, _p.s_min, _p.v_min),
		cv::Scalar(_p.h_max, _p.s_max, _p.v_max),
		mask);

	cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  _morph_kernel);
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, _morph_kernel);

	// Image center used for proximity scoring
	const cv::Point2f img_center(hsv.cols * 0.5f, hsv.rows * 0.5f);

	// ── Contour extraction ─────────────────────────────────────────
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	bool   found     = false;
	double bestScore = -1e18;

	for (const auto &contour : contours) {
		const double area = cv::contourArea(contour);
		if (area < _p.min_box_area || area > _p.max_box_area) continue;

		const double perimeter = cv::arcLength(contour, true);
		if (perimeter < 1e-6) continue;

		const cv::Rect  bbox = cv::boundingRect(contour);
		if (bbox.width <= 0 || bbox.height <= 0) continue;

		const cv::Point2f center(
			bbox.x + bbox.width  * 0.5f,
			bbox.y + bbox.height * 0.5f);

		const double aspect      = static_cast<double>(bbox.width) / bbox.height;
		const double bbox_area   = static_cast<double>(bbox.width  * bbox.height);
		const double fill_ratio  = area / std::max(1.0, bbox_area);
		const double circularity = 4.0 * M_PI * area / (perimeter * perimeter);

		// Shape filters
		if (circularity > 0.82)             continue;
		if (aspect < 0.60 || aspect > 2.20) continue;
		if (fill_ratio < 0.45)              continue;

		// Lock gate + scoring
		double score = 0.0;
		if (_lock.locked) {
			if (!isLockedCandidateValid(bbox, center)) continue;
			score = scoreLockedCandidate(center, bbox, area);
		} else {
			score = scoreUnlockedCandidate(area, fill_ratio, circularity, center, img_center);
		}

		if (!found || score > bestScore) {
			bestScore = score;
			boxBbox   = bbox;
			boxCenter = center;
			found     = true;
		}
	}

	return found;
}

bool CubeDetectorNode::isLockedCandidateValid(const cv::Rect &bbox, const cv::Point2f &center) const
{
	const double dx   = center.x - _lock.center.x;
	const double dy   = center.y - _lock.center.y;
	const double dist = std::sqrt(dx * dx + dy * dy);

	if (dist > _p.lock_max_dist_px) return false;

	const float vis = static_cast<float>(std::max(bbox.width, bbox.height));
	if (_lock.visible_size > 1e-6f && vis < _lock.visible_size * _p.lock_min_size_ratio) {
		return false;
	}

	return true;
}

double CubeDetectorNode::scoreLockedCandidate(
	const cv::Point2f &center, const cv::Rect &bbox, double area) const
{
	const double dx         = center.x - _lock.center.x;
	const double dy         = center.y - _lock.center.y;
	const double dist       = std::sqrt(dx * dx + dy * dy);
	const double vis        = static_cast<double>(std::max(bbox.width, bbox.height));
	const double size_ratio = vis / std::max(1.0f, _lock.visible_size);

	return -_p.score_dist_weight       * dist
	       -_p.score_size_ratio_weight * std::abs(1.0 - size_ratio)
	       +_p.score_area_weight       * area;
}

double CubeDetectorNode::scoreUnlockedCandidate(
	double area, double fill_ratio, double circularity,
	const cv::Point2f &center, const cv::Point2f &img_center) const
{
	const double dx   = center.x - img_center.x;
	const double dy   = center.y - img_center.y;
	const double dist = std::sqrt(dx * dx + dy * dy);

	return  _p.score_area_weight        * area
	       +_p.score_fill_weight        * fill_ratio
	       -_p.score_circularity_weight * circularity
	       -_p.score_center_weight      * dist;
}

bool CubeDetectorNode::detectCircularHandle(
	const cv::Mat &frame, const cv::Rect &boxRegion,
	cv::Point2f &circleCenter, float &radius)
{
	cv::Rect searchRegion = {
		boxRegion.x,
		std::max(0, boxRegion.y - boxRegion.height),
		boxRegion.width,
		boxRegion.height * 2
	};
	if (searchRegion.area() <= 0) return false;

	cv::Mat hsv;
	cv::cvtColor(frame(searchRegion), hsv, cv::COLOR_BGR2HSV);

	cv::Mat mask;
	cv::inRange(hsv,
		cv::Scalar(_p.circle_h_min, _p.circle_s_min, _p.circle_v_min),
		cv::Scalar(_p.circle_h_max, _p.circle_s_max, _p.circle_v_max),
		mask);

	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT,
		1, mask.rows / 2, 100, 20,
		static_cast<int>(_p.circle_min_radius),
		static_cast<int>(_p.circle_max_radius));

	if (circles.empty()) return false;

	circleCenter = { circles[0][0] + searchRegion.x, circles[0][1] + searchRegion.y };
	radius       = circles[0][2];
	return true;
}

bool CubeDetectorNode::validateVictimGeometry(const VictimModel &model)
{
	if (!model.hasCircle) return false;

	const float boxCenterX = model.boxBbox.x + model.boxBbox.width * 0.5f;

	return (model.circleCenter.y <= model.boxBbox.y) &&
	       (std::abs(model.circleCenter.x - boxCenterX) <= model.boxBbox.width * 0.4f);
}

// ═══════════════════════════════════════════════════════════════════
//  Pose estimation
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::estimatePose(
	const VictimModel &target, int /*image_width*/, int /*image_height*/,
	double &x, double &y, double &z) const
{
	const double fx = _camera_matrix.at<double>(0, 0);
	const double fy = _camera_matrix.at<double>(1, 1);
	const double cx = _camera_matrix.at<double>(0, 2);
	const double cy = _camera_matrix.at<double>(1, 2);

	const double pixel_width = target.boxBbox.width;
	if (pixel_width < 1.0) { x = y = z = 0.0; return; }

	z = (fx * _p.box_width_m) / pixel_width;
	x = (target.boxCenter.x - cx) * z / fx;
	y = (target.boxCenter.y - cy) * z / fy;
}

// ═══════════════════════════════════════════════════════════════════
//  Kalman filter  (1-D per axis, stationary model)
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::kalmanPredict()
{
	// State is assumed constant between frames; only covariance grows
	_kalman.cov_x += _p.kalman_q;
	_kalman.cov_y += _p.kalman_q;
	_kalman.cov_z += _p.kalman_q;
}

void CubeDetectorNode::kalmanUpdate(double meas_x, double meas_y, double meas_z)
{
	// Predict step
	const double pred_cov_x = _kalman.cov_x + _p.kalman_q;
	const double pred_cov_y = _kalman.cov_y + _p.kalman_q;
	const double pred_cov_z = _kalman.cov_z + _p.kalman_q;

	// Kalman gain
	const double Kx = pred_cov_x / (pred_cov_x + _p.kalman_r);
	const double Ky = pred_cov_y / (pred_cov_y + _p.kalman_r);
	const double Kz = pred_cov_z / (pred_cov_z + _p.kalman_r);

	// State update
	_kalman.x += Kx * (meas_x - _kalman.x);
	_kalman.y += Ky * (meas_y - _kalman.y);
	_kalman.z += Kz * (meas_z - _kalman.z);

	// Covariance update
	_kalman.cov_x = (1.0 - Kx) * pred_cov_x;
	_kalman.cov_y = (1.0 - Ky) * pred_cov_y;
	_kalman.cov_z = (1.0 - Kz) * pred_cov_z;
}

// ═══════════════════════════════════════════════════════════════════
//  Publishing helpers
// ═══════════════════════════════════════════════════════════════════
geometry_msgs::msg::PoseStamped CubeDetectorNode::buildPoseMsg(
	const std_msgs::msg::Header &header, double x, double y, double z) const
{
	geometry_msgs::msg::PoseStamped msg;
	msg.header             = header;
	msg.header.frame_id    = "camera_frame";
	msg.pose.position.x    = x;
	msg.pose.position.y    = y;
	msg.pose.position.z    = z;
	msg.pose.orientation.w = 1.0;
	return msg;
}

void CubeDetectorNode::publishResults(
	const VictimModel &/*victim*/, const std_msgs::msg::Header &header,
	bool valid, double x, double y, double z)
{
	std_msgs::msg::Bool valid_msg;
	valid_msg.data = valid;
	_target_valid_pub->publish(valid_msg);

	if (valid) {
		_target_pose_pub->publish(buildPoseMsg(header, x, y, z));
	}
}

// ═══════════════════════════════════════════════════════════════════
//  State management
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::resetLockState()
{
	_lock.reset();
	_last_seen.reset();
	_kalman.reset();

	_has_last_pose = false;
	_last_x = _last_y = _last_z = 0.0;
}

// ═══════════════════════════════════════════════════════════════════
//  Debug annotation
// ═══════════════════════════════════════════════════════════════════
void CubeDetectorNode::annotateImage(
	cv_bridge::CvImagePtr image, const VictimModel &target,
	double x, double y, double z,
	const std::string &perf_text) const
{
	cv::Mat &img = image->image;

	// Box overlay
	if (target.boxBbox.area() > 0) {
		cv::rectangle(img, target.boxBbox, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
		cv::circle(img, cv::Point(target.boxCenter), 5, cv::Scalar(0, 255, 0), -1, cv::LINE_AA);
	}

	// Info panel
	cv::rectangle(img, cv::Point(8, 8), cv::Point(290, 225), cv::Scalar(40, 40, 40), -1);

	const auto putRow = [&](const std::string &text, int row, cv::Scalar color) {
		cv::putText(img, text, cv::Point(15, 30 + row * 35),
			cv::FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv::LINE_AA);
	};

	// Status row
	const std::string status = std::string("Box: ") + (target.valid ? "LOCKED" : "SEARCHING");
	cv::putText(img, status, cv::Point(15, 30),
		cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

	const cv::Scalar pose_color = _has_last_pose ? cv::Scalar(255, 255, 0) : cv::Scalar(180, 180, 180);

	auto fmtAxis = [&](char axis, double val) -> std::string {
		if (!_has_last_pose) return std::string(1, axis) + ": N/A";
		std::ostringstream ss;
		ss << axis << ": " << std::fixed << std::setprecision(3) << val << " m";
		return ss.str();
	};

	putRow(fmtAxis('X', x), 1, pose_color);
	putRow(fmtAxis('Y', y), 2, pose_color);
	putRow(fmtAxis('Z', z), 3, pose_color);

	std::ostringstream conf;
	conf << "Conf: " << std::fixed << std::setprecision(2) << target.confidence;
	putRow(conf.str(), 4, cv::Scalar(0, 255, 255));

	if (!perf_text.empty()) {
		cv::putText(img, perf_text, cv::Point(15, 30 + 5 * 35),
			cv::FONT_HERSHEY_SIMPLEX, 0.70, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
	}
}

// ═══════════════════════════════════════════════════════════════════
//  Entry point
// ═══════════════════════════════════════════════════════════════════
int main(int argc, char *argv[])
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<CubeDetectorNode>());
	rclcpp::shutdown();
	return 0;
}