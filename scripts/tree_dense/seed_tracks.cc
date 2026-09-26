// Test-only measured ORB correspondences triangulated with supplied COLMAP poses.
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct View {
  int id, width, height;
  std::string path;
  cv::Matx33d k, r;
  cv::Vec3d t, center;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

int main(int argc, char** argv) {
  if (argc != 3) { std::cerr << "usage: tree_dense_seed views.tsv seeds.txt\n"; return 2; }
  std::ifstream in(argv[1]);
  if (!in) { std::cerr << "cannot open views input\n"; return 2; }
  std::vector<View> views;
  std::string line;
  auto orb = cv::ORB::create(3000, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream row(line);
    View v;
    double fx, fy, cx, cy;
    row >> v.id >> v.width >> v.height >> v.path >> fx >> fy >> cx >> cy;
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) row >> v.r(r,c);
    for (int r = 0; r < 3; ++r) row >> v.t[r];
    if (!row || v.width<=0 || v.height<=0 || v.width>800 || v.height>800 ||
        !std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) ||
        !std::isfinite(cy) || fx<=0 || fy<=0) {
      std::cerr << "invalid view row\n"; return 2;
    }
    for (double x : v.r.val) if (!std::isfinite(x)) return 2;
    for (double x : v.t.val) if (!std::isfinite(x)) return 2;
    v.k = cv::Matx33d(fx,0,cx, 0,fy,cy, 0,0,1);
    v.center = -(v.r.t() * v.t);
    cv::Mat gray = cv::imread(v.path, cv::IMREAD_GRAYSCALE);
    if (gray.empty() || gray.cols != v.width || gray.rows != v.height) {
      std::cerr << "image read or size failed: " << v.path << "\n"; return 2;
    }
    orb->detectAndCompute(gray, cv::noArray(), v.keypoints, v.descriptors);
    std::cerr << "view " << v.id << " ORB " << v.keypoints.size() << "\n";
    views.push_back(std::move(v));
  }
  if (views.size() < 2 || views.size() > 12) { std::cerr << "view count outside 2..12\n"; return 2; }
  std::ofstream out(argv[2]);
  if (!out) return 2;
  out << std::setprecision(12);
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  size_t all = 0;
  for (size_t i = 0; i < views.size(); ++i) for (size_t j = i + 1; j < views.size(); ++j) {
    auto const& a = views[i]; auto const& b = views[j];
    if (a.descriptors.empty() || b.descriptors.empty()) continue;
    std::vector<std::vector<cv::DMatch>> ab, ba;
    matcher.knnMatch(a.descriptors, b.descriptors, ab, 2);
    matcher.knnMatch(b.descriptors, a.descriptors, ba, 2);
    std::vector<int> reverse(b.keypoints.size(), -1);
    for (size_t k = 0; k < ba.size(); ++k)
      if (ba[k].size() == 2 && ba[k][0].distance < .8f * ba[k][1].distance)
        reverse[k] = ba[k][0].trainIdx;
    cv::Matx34d pa, pb;
    for (int r=0; r<3; ++r) {
      for (int c=0; c<3; ++c) { pa(r,c) = a.r(r,c); pb(r,c) = b.r(r,c); }
      pa(r,3)=a.t[r]; pb(r,3)=b.t[r];
    }
    auto const ka = a.k * pa, kb = b.k * pb;
    size_t kept = 0;
    for (size_t k=0; k<ab.size() && kept<500 && all<10000; ++k) {
      if (ab[k].size()!=2 || ab[k][0].distance >= .8f*ab[k][1].distance) continue;
      auto const& m=ab[k][0];
      if (reverse[m.trainIdx] != static_cast<int>(k)) continue;
      auto const p=a.keypoints[k].pt, q=b.keypoints[m.trainIdx].pt;
      cv::Mat points4;
      cv::triangulatePoints(cv::Mat(ka), cv::Mat(kb),
        std::vector<cv::Point2d>{{p.x,p.y}}, std::vector<cv::Point2d>{{q.x,q.y}}, points4);
      if (points4.type()!=CV_64FC1) { std::cerr << "unexpected triangulation type\n"; return 2; }
      double w=points4.at<double>(3,0);
      if (!std::isfinite(w) || std::abs(w)<1e-12) continue;
      cv::Vec3d x(points4.at<double>(0,0)/w, points4.at<double>(1,0)/w,
                  points4.at<double>(2,0)/w);
      if (!std::isfinite(x[0]) || !std::isfinite(x[1]) || !std::isfinite(x[2])) continue;
      auto xa=a.r*x+a.t, xb=b.r*x+b.t;
      if (xa[2]<=0 || xb[2]<=0) continue;
      double ea=std::hypot(a.k(0,0)*xa[0]/xa[2]+a.k(0,2)-p.x,
                           a.k(1,1)*xa[1]/xa[2]+a.k(1,2)-p.y);
      double eb=std::hypot(b.k(0,0)*xb[0]/xb[2]+b.k(0,2)-q.x,
                           b.k(1,1)*xb[1]/xb[2]+b.k(1,2)-q.y);
      if (!std::isfinite(ea) || !std::isfinite(eb) || ea>2 || eb>2) continue;
      auto ra=x-a.center, rb=x-b.center;
      double cosine=ra.dot(rb)/(cv::norm(ra)*cv::norm(rb));
      if (!std::isfinite(cosine) || cosine>std::cos(1.0*CV_PI/180.0)) continue;
      out << x[0] << ' ' << x[1] << ' ' << x[2] << ' '
          << i << ' ' << p.x << ' ' << p.y << ' '
          << j << ' ' << q.x << ' ' << q.y << ' ' << ea << ' ' << eb << '\n';
      ++kept; ++all;
    }
    std::cerr << "pair " << i << '-' << j << " seeds " << kept << "\n";
  }
  std::cerr << "total seeds " << all << "\n";
  return all ? 0 : 3;
}
