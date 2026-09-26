// Independent, outer-boundary ArUco raster fixture. No ground truth enters detection.
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kWidth = 800;
constexpr int kHeight = 600;
constexpr int kSupersample = 4;
constexpr double kMarkerSize = 40.0;
const cv::Matx33d kK(850, 0, 399.5, 0, 850, 299.5, 0, 0, 1);
const std::array<cv::Point2d, 4> kOrigins = {
    cv::Point2d(0, 0), cv::Point2d(260, 0),
    cv::Point2d(0, 260), cv::Point2d(260, 260)};

struct PoseSpec {
  cv::Vec3d rvec;
  cv::Vec3d tvec;
  double blur_sigma;
};

struct Error {
  bool success = false;
  int detected = 0;
  double corner_mae_px = std::numeric_limits<double>::quiet_NaN();
  double corner_p95_px = std::numeric_limits<double>::quiet_NaN();
  double camera_center_error_mm = std::numeric_limits<double>::quiet_NaN();
  double rotation_error_deg = std::numeric_limits<double>::quiet_NaN();
};

std::array<cv::Point3f, 4> object_corners(int marker) {
  const auto p = kOrigins.at(marker);
  return {cv::Point3f(p.x, p.y, 0),
          cv::Point3f(p.x + kMarkerSize, p.y, 0),
          cv::Point3f(p.x + kMarkerSize, p.y + kMarkerSize, 0),
          cv::Point3f(p.x, p.y + kMarkerSize, 0)};
}

cv::Mat render(const PoseSpec& pose, const std::array<cv::Mat, 4>& markers) {
  cv::Mat R;
  cv::Rodrigues(pose.rvec, R);
  cv::Matx33d plane_to_camera;
  for (int row = 0; row < 3; ++row) {
    plane_to_camera(row, 0) = R.at<double>(row, 0);
    plane_to_camera(row, 1) = R.at<double>(row, 1);
    plane_to_camera(row, 2) = pose.tvec[row];
  }
  const cv::Matx33d H = kK * plane_to_camera;
  const cv::Matx33d inverse = H.inv();
  cv::Mat hi(kHeight * kSupersample, kWidth * kSupersample, CV_8UC1);
  for (int sy = 0; sy < hi.rows; ++sy) {
    const double y = (sy + 0.5) / kSupersample - 0.5;
    auto* pixel = hi.ptr<unsigned char>(sy);
    for (int sx = 0; sx < hi.cols; ++sx) {
      // Output pixel centres are at integer coordinates. Physical edges are
      // at half-integer boundaries; each high-resolution sample is integrated.
      const double x = (sx + 0.5) / kSupersample - 0.5;
      const cv::Vec3d q = inverse * cv::Vec3d(x, y, 1);
      const double world_x = q[0] / q[2];
      const double world_y = q[1] / q[2];
      double intensity = 230;
      for (size_t marker = 0; marker < kOrigins.size(); ++marker) {
        const auto p = kOrigins[marker];
        if (world_x >= p.x && world_x < p.x + kMarkerSize &&
            world_y >= p.y && world_y < p.y + kMarkerSize) {
          const cv::Mat& texture = markers[marker];
          const int tx = std::min(texture.cols - 1,
                                  static_cast<int>((world_x - p.x) / kMarkerSize * texture.cols));
          const int ty = std::min(texture.rows - 1,
                                  static_cast<int>((world_y - p.y) / kMarkerSize * texture.rows));
          intensity = texture.at<unsigned char>(ty, tx) ? 245 : 20;
          break;
        }
      }
      const double light = 0.86 + 0.20 * x / kWidth + 0.06 * y / kHeight;
      pixel[sx] = cv::saturate_cast<unsigned char>(intensity * light);
    }
  }
  cv::Mat image;
  cv::resize(hi, image, cv::Size(kWidth, kHeight), 0, 0, cv::INTER_AREA);
  if (pose.blur_sigma > 0) cv::GaussianBlur(image, image, cv::Size(), pose.blur_sigma);
  return image;
}

Error measure(const cv::Mat& image, const PoseSpec& truth,
              const cv::aruco::DetectorParameters& parameters) {
  cv::aruco::ArucoDetector detector(
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50), parameters);
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<int> ids;
  detector.detectMarkers(image, corners, ids);
  Error result;
  result.detected = static_cast<int>(ids.size());
  if (ids.size() != kOrigins.size()) return result;
  std::vector<cv::Point3f> objects;
  std::vector<cv::Point2f> pixels;
  std::vector<double> corner_errors;
  const cv::Mat K(kK), D = cv::Mat::zeros(1, 5, CV_64F);
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 1 || ids[i] > 4 || corners[i].size() != 4) return result;
    const auto object = object_corners(ids[i] - 1);
    std::vector<cv::Point2f> projected;
    cv::projectPoints(std::vector<cv::Point3f>(object.begin(), object.end()),
                      truth.rvec, truth.tvec, K, D, projected);
    for (int j = 0; j < 4; ++j) {
      objects.push_back(object[j]);
      pixels.push_back(corners[i][j]);
      corner_errors.push_back(cv::norm(corners[i][j] - projected[j]));
    }
  }
  std::vector<cv::Mat> rvecs, tvecs;
  if (cv::solvePnPGeneric(objects, pixels, K, D, rvecs, tvecs, false,
                          cv::SOLVEPNP_IPPE) < 1) return result;
  int best = -1;
  double best_rms = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < rvecs.size(); ++i) {
    std::vector<cv::Point2f> projected;
    cv::projectPoints(objects, rvecs[i], tvecs[i], K, D, projected);
    double sum = 0;
    for (size_t j = 0; j < pixels.size(); ++j)
      sum += cv::norm(projected[j] - pixels[j]) * cv::norm(projected[j] - pixels[j]);
    if (sum < best_rms) { best_rms = sum; best = static_cast<int>(i); }
  }
  cv::Mat rvec = rvecs[best].clone(), tvec = tvecs[best].clone();
  cv::solvePnPRefineLM(objects, pixels, K, D, rvec, tvec);
  cv::Mat estimated_R, true_R;
  cv::Rodrigues(rvec, estimated_R);
  cv::Rodrigues(truth.rvec, true_R);
  cv::Mat true_t(truth.tvec);
  cv::Mat center = -estimated_R.t() * tvec;
  cv::Mat true_center = -true_R.t() * true_t;
  cv::Mat delta_R = estimated_R * true_R.t();
  const double trace = delta_R.at<double>(0, 0) + delta_R.at<double>(1, 1) + delta_R.at<double>(2, 2);
  std::sort(corner_errors.begin(), corner_errors.end());
  result.success = true;
  result.corner_mae_px = std::accumulate(corner_errors.begin(), corner_errors.end(), 0.0) / corner_errors.size();
  result.corner_p95_px = corner_errors[static_cast<size_t>(std::ceil(0.95 * corner_errors.size())) - 1];
  result.camera_center_error_mm = cv::norm(center - true_center);
  result.rotation_error_deg = std::acos(std::clamp((trace - 1) / 2, -1.0, 1.0)) * 180 / CV_PI;
  return result;
}

void write_error(std::ostream& output, const Error& error) {
  output << "{\"detected\":" << error.detected << ",\"success\":"
         << (error.success ? "true" : "false");
  if (error.success)
    output << ",\"corner_mae_px\":" << error.corner_mae_px
           << ",\"corner_p95_px\":" << error.corner_p95_px
           << ",\"camera_center_error_mm\":" << error.camera_center_error_mm
           << ",\"rotation_error_deg\":" << error.rotation_error_deg;
  output << '}';
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: pose_fixture OUTPUT.json\n";
    return 2;
  }
  const std::vector<PoseSpec> views = {
      {{0.02, -0.03, 0.01}, {-147.3, -152.1, 640.0}, 0.0},
      {{0.06, 0.04, -0.02}, {-139.7, -146.4, 660.0}, 0.5},
      {{-0.08, 0.02, 0.04}, {-157.2, -140.8, 690.0}, 0.8},
      {{0.03, -0.09, -0.05}, {-148.6, -159.3, 610.0}, 1.0},
      {{-0.05, 0.08, 0.03}, {-141.2, -153.7, 670.0}, 0.7},
      {{0.10, -0.02, -0.03}, {-154.8, -145.1, 720.0}, 0.4},
      {{-0.04, -0.06, 0.07}, {-144.4, -150.6, 635.0}, 1.1},
      {{0.07, 0.07, -0.04}, {-151.9, -143.2, 700.0}, 0.6},
  };
  auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  std::array<cv::Mat, 4> markers;
  for (int i = 0; i < 4; ++i)
    cv::aruco::generateImageMarker(dictionary, i + 1, 600, markers[i], 1);
  if (std::filesystem::exists(argv[1])) throw std::runtime_error("output already exists");
  for (size_t i = 0; i < views.size(); ++i)
    if (std::filesystem::exists(std::string(argv[1]) + ".view" + std::to_string(i) + ".png"))
      throw std::runtime_error("rendered image target already exists");
  std::ofstream output(argv[1], std::ios::out);
  if (!output) throw std::runtime_error("cannot open output");
  output << std::setprecision(10)
         << "{\"fixture\":\"outer-boundary-4x-area-aruco\","
         << "\"image_size\":[800,600],\"supersampling\":4,"
         << "\"units\":\"mm\",\"views\":[";
  cv::aruco::DetectorParameters defaults;
  cv::aruco::DetectorParameters subpix;
  subpix.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
  bool all_success = true;
  for (size_t i = 0; i < views.size(); ++i) {
    cv::Mat image = render(views[i], markers);
    if (!cv::imwrite(std::string(argv[1]) + ".view" + std::to_string(i) + ".png", image))
      throw std::runtime_error("cannot write rendered fixture image");
    const Error baseline = measure(image, views[i], defaults);
    const Error refined = measure(image, views[i], subpix);
    all_success = all_success && baseline.success && refined.success;
    if (i) output << ',';
    output << "{\"index\":" << i << ",\"rvec\":[" << views[i].rvec[0] << ','
           << views[i].rvec[1] << ',' << views[i].rvec[2] << "],\"tvec\":["
           << views[i].tvec[0] << ',' << views[i].tvec[1] << ',' << views[i].tvec[2]
           << "],\"blur_sigma\":" << views[i].blur_sigma << ",\"default\":";
    write_error(output, baseline);
    output << ",\"subpix\":";
    write_error(output, refined);
    output << '}';
  }
  output << "]}\n";
  std::cout << argv[1] << '\n';
  return all_success ? 0 : 3;
}
