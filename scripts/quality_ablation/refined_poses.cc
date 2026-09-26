// Fixture-specific CORNER_REFINE_SUBPIX experiment. Uses the production IPPE+LM solver.
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  cv::Mat K = (cv::Mat_<double>(3,3) << 1500,0,800, 0,1500,600, 0,0,1);
  cv::Mat D = cv::Mat::zeros(1,4,CV_64F);
  std::map<int, std::vector<cv::Point3f>> layout;
  layout[1]={{0,0,0},{40,0,0},{40,40,0},{0,40,0}};
  layout[2]={{260,0,0},{300,0,0},{300,40,0},{260,40,0}};
  layout[3]={{0,260,0},{40,260,0},{40,300,0},{0,300,0}};
  layout[4]={{260,260,0},{300,260,0},{300,300,0},{260,300,0}};
  cv::aruco::DetectorParameters params;
  params.cornerRefinementMethod=cv::aruco::CORNER_REFINE_SUBPIX;
  cv::aruco::ArucoDetector detector(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),params);
  std::cout << std::setprecision(15) << "{\"poses\":[";
  for (int vi=1; vi<argc; ++vi) {
    cv::Mat gray=cv::imread(argv[vi],cv::IMREAD_GRAYSCALE | cv::IMREAD_IGNORE_ORIENTATION);
    if (gray.cols!=1600 || gray.rows!=1200) return 3;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector.detectMarkers(gray,corners,ids);
    std::vector<cv::Point3f> objects;
    std::vector<cv::Point2f> pixels;
    for (size_t i=0;i<ids.size();++i) if (layout.count(ids[i])) {
      objects.insert(objects.end(),layout[ids[i]].begin(),layout[ids[i]].end());
      pixels.insert(pixels.end(),corners[i].begin(),corners[i].end());
    }
    if (objects.size()<4) return 4;
    std::vector<cv::Mat> rv,tv;
    if (cv::solvePnPGeneric(objects,pixels,K,D,rv,tv,false,cv::SOLVEPNP_IPPE)<2) return 5;
    auto residual=[&](const cv::Mat& r,const cv::Mat& t) {
      std::vector<cv::Point2f> projected;
      cv::projectPoints(objects,r,t,K,D,projected);
      double sum=0;
      for(size_t k=0;k<pixels.size();++k) {double e=cv::norm(pixels[k]-projected[k]);sum+=e*e;}
      return std::sqrt(sum/pixels.size());
    };
    int best=residual(rv[0],tv[0])<=residual(rv[1],tv[1])?0:1;
    if (std::abs(residual(rv[0],tv[0])-residual(rv[1],tv[1]))<.25) return 6;
    cv::Mat r=rv[best].clone(),t=tv[best].clone(),R;
    cv::solvePnPRefineLM(objects,pixels,K,D,r,t);
    cv::Rodrigues(r,R);
    if(vi>1) std::cout << ',';
    std::cout << "{\"imageId\":\"view-" << vi << "\",\"rotation\":[";
    for(int k=0;k<9;++k) {if(k)std::cout<<',';std::cout<<R.at<double>(k/3,k%3);}
    std::cout << "],\"translationMm\":[";
    for(int k=0;k<3;++k) {if(k)std::cout<<',';std::cout<<t.at<double>(k);}
    std::cout << "],\"markerRmsPx\":" << residual(r,t) << ",\"markerCount\":" << ids.size() << '}';
  }
  std::cout << "]}\n";
}
