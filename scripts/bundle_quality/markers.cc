// Extract the default-detector board observations used by the production pose path.
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iomanip>
#include <iostream>
#include <map>

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  std::map<int,std::array<cv::Point3f,4>> board;
  board[1]={cv::Point3f(0,0,0),{40,0,0},{40,40,0},{0,40,0}};
  board[2]={cv::Point3f(260,0,0),{300,0,0},{300,40,0},{260,40,0}};
  board[3]={cv::Point3f(0,260,0),{40,260,0},{40,300,0},{0,300,0}};
  board[4]={cv::Point3f(260,260,0),{300,260,0},{300,300,0},{260,300,0}};
  cv::aruco::ArucoDetector detector(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50));
  std::cout << std::setprecision(15);
  for(int vi=1;vi<argc;++vi) {
    cv::Mat gray=cv::imread(argv[vi],cv::IMREAD_GRAYSCALE | cv::IMREAD_IGNORE_ORIENTATION);
    if(gray.cols!=1600 || gray.rows!=1200) return 3;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector.detectMarkers(gray,corners,ids);
    int used=0;
    for(size_t i=0;i<ids.size();++i) {
      auto found=board.find(ids[i]);
      if(found==board.end() || corners[i].size()!=4) continue;
      ++used;
      for(int k=0;k<4;++k) {
        const auto& q=found->second[k];
        const auto& p=corners[i][k];
        std::cout << vi-1 << ' ' << ids[i] << ' ' << k << ' '
                  << q.x << ' ' << q.y << ' ' << q.z << ' ' << p.x << ' ' << p.y << '\n';
      }
    }
    if(used!=4) return 4;
  }
}
