#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
struct Camera {
  cv::Mat K, R, t, image;
};
struct Seed { cv::Point3d x; cv::Point2d a, b; int original_index; };

Camera camera(const cv::FileNode& view) {
  const auto c=view["camera"];
  const int w=(int)c["width"], h=(int)c["height"];
  const double fx=(double)c["fx"], fy=(double)c["fy"];
  const double cx=(double)c["cx"], cy=(double)c["cy"];
  Camera out;
  out.K=(cv::Mat_<double>(3,3)<<fx,0,cx,0,fy,cy,0,0,1);
  out.R=cv::Mat(3,3,CV_64F); out.t=cv::Mat(3,1,CV_64F);
  auto r=view["rotation"], t=view["translation"];
  for(int i=0;i<9;++i) out.R.at<double>(i/3,i%3)=(double)r[i];
  for(int i=0;i<3;++i) out.t.at<double>(i)=(double)t[i];
  out.image=cv::imread((std::string)view["output_image"],cv::IMREAD_GRAYSCALE);
  if(out.image.size()!=cv::Size(w,h)) throw std::runtime_error("scene image missing or dimensions differ");
  return out;
}
cv::Point2d rectified_point(const cv::Point2d& p,const Camera& c,const cv::Mat& R,const cv::Mat& P,bool vertical) {
  std::vector<cv::Point2d> q;
  cv::undistortPoints(std::vector<cv::Point2d>{p},q,c.K,cv::Mat(),R,P);
  return vertical ? cv::Point2d(q[0].y,q[0].x) : q[0];
}
cv::Point2d project(const cv::Point3d& x,const Camera& c,const cv::Mat& R,const cv::Mat& P,bool vertical) {
  cv::Mat X=(cv::Mat_<double>(3,1)<<x.x,x.y,x.z);
  cv::Mat Y=R*(c.R*X+c.t);
  // Each world point is already expressed in its own camera frame. P's fourth
  // column is for the common rectified first-camera frame, not this mapping.
  cv::Mat p=P(cv::Rect(0,0,3,3))*Y;
  cv::Point2d q(p.at<double>(0)/p.at<double>(2),p.at<double>(1)/p.at<double>(2));
  return vertical ? cv::Point2d(q.y,q.x) : q;
}
void pgm(const fs::path& path,const cv::Mat& image) {
  std::ofstream f(path,std::ios::binary);
  if(!f || image.type()!=CV_8UC1) throw std::runtime_error("cannot write grayscale PGM");
  f<<"P5\n"<<image.cols<<' '<<image.rows<<"\n255\n";
  for(int y=0;y<image.rows;++y) f.write(reinterpret_cast<const char*>(image.ptr(y)),image.cols);
  if(!f) throw std::runtime_error("PGM write failed");
}
cv::Mat read_pgm(const fs::path& path) {
  std::ifstream f(path,std::ios::binary); std::string magic; int w=0,h=0,maxval=0;
  f>>magic>>w>>h>>maxval;
  if(!f || magic!="P5" || w<=0 || h<=0 || maxval!=255 || w*h>8000000) throw std::runtime_error("invalid PGM");
  f.get(); cv::Mat image(h,w,CV_8UC1);
  f.read(reinterpret_cast<char*>(image.data),size_t(w)*h);
  if(!f) throw std::runtime_error("truncated PGM");
  return image;
}
void prepare(const fs::path& scene,const fs::path& out) {
  if(fs::exists(out)) throw std::runtime_error("output exists; use a fresh directory");
  cv::FileStorage f((scene/"conversion.json").string(),cv::FileStorage::READ);
  if(!f.isOpened()) throw std::runtime_error("cannot read conversion.json");
  auto views=f["views"];
  if(views.size()!=10) throw std::runtime_error("expected fixed ten-view scene");
  Camera a=camera(views[1]), b=camera(views[4]);
  cv::Mat R=b.R*a.R.t(), T=b.t-R*a.t;
  cv::Mat R1,R2,P1,P2,Q;
  cv::stereoRectify(a.K,cv::Mat(),b.K,cv::Mat(),a.image.size(),R,T,R1,R2,P1,P2,Q,cv::CALIB_ZERO_DISPARITY,0,a.image.size());
  bool vertical=std::abs(P2.at<double>(1,3))>std::abs(P2.at<double>(0,3));
  int axis=vertical?1:0;
  bool swap=P2.at<double>(axis,3)>0; // d=x_left-x_right=-f*T/Z must be positive.
  cv::Mat m1x,m1y,m2x,m2y,ra,rb;
  cv::initUndistortRectifyMap(a.K,cv::Mat(),R1,P1,a.image.size(),CV_32FC1,m1x,m1y);
  cv::initUndistortRectifyMap(b.K,cv::Mat(),R2,P2,b.image.size(),CV_32FC1,m2x,m2y);
  cv::remap(a.image,ra,m1x,m1y,cv::INTER_LINEAR);
  cv::remap(b.image,rb,m2x,m2y,cv::INTER_LINEAR);
  if(vertical) {cv::transpose(ra,ra);cv::transpose(rb,rb);}
  if(swap) std::swap(ra,rb);
  std::vector<Seed> seeds;
  std::ifstream input(scene/"measured-seeds.txt");
  if(!input) throw std::runtime_error("missing measured-seeds.txt");
  std::string line; int index=0;
  while(std::getline(input,line)) {
    std::istringstream s(line); Seed q; int i,j; double e1,e2;
    if(!(s>>q.x.x>>q.x.y>>q.x.z>>i>>q.a.x>>q.a.y>>j>>q.b.x>>q.b.y>>e1>>e2)) throw std::runtime_error("bad seed line");
    q.original_index=index++;
    if(i==1 && j==4) seeds.push_back(q);
    else if(i==4 && j==1) {std::swap(q.a,q.b);seeds.push_back(q);}
  }
  if(seeds.size()<50) throw std::runtime_error("too few pair seeds");
  fs::create_directories(out);
  pgm(out/"left.pgm",ra); pgm(out/"right.pgm",rb);
  std::ofstream rows(out/"correspondences.csv");
  rows<<"split,source_index,left_x,left_y,right_x,right_y,geometric_disparity,observed_disparity,epipolar_residual\n"<<std::setprecision(12);
  double min_d=1e30,max_d=-1e30; int train=0, held=0;
  for(size_t n=0;n<seeds.size();++n) {
    const auto& q=seeds[n];
    auto aa=rectified_point(q.a,a,R1,P1,vertical), bb=rectified_point(q.b,b,R2,P2,vertical);
    auto ga=project(q.x,a,R1,P1,vertical), gb=project(q.x,b,R2,P2,vertical);
    if(swap) {std::swap(aa,bb);std::swap(ga,gb);}
    double d=ga.x-gb.x;
    bool hold=(n%5==0); // fixed split before disparity-bound selection.
    if(!hold) {min_d=std::min(min_d,d);max_d=std::max(max_d,d);++train;} else ++held;
    rows<<(hold?"holdout":"train")<<','<<q.original_index<<','<<aa.x<<','<<aa.y<<','<<bb.x<<','<<bb.y<<','<<d<<','<<(aa.x-bb.x)<<','<<std::abs(aa.y-bb.y)<<'\n';
  }
  // ELAS starts at zero. The identical search interval is passed to SGBM.
  int ndisp=(int)std::ceil(max_d+8);
  ndisp=((std::max(16,ndisp)+15)/16)*16;
  if(min_d<0 || max_d>=ra.cols || ndisp>=ra.cols || ndisp>512) throw std::runtime_error("geometric disparity range outside supported image/search bounds");
  std::ofstream meta(out/"geometry.json");
  meta<<std::setprecision(12)<<"{\"axis\":\""<<(vertical?"vertical":"horizontal")<<"\",\"swapped\":"<<(swap?"true":"false")
      <<",\"width\":"<<ra.cols<<",\"height\":"<<ra.rows<<",\"pair_seeds\":"<<seeds.size()
      <<",\"training\":"<<train<<",\"heldout\":"<<held<<",\"geometric_min\":"<<min_d
      <<",\"geometric_max\":"<<max_d<<",\"ndisp\":"<<ndisp<<"}\n";
  std::cout<<"prepared "<<seeds.size()<<" pair seeds; axis="<<(vertical?"vertical":"horizontal")<<" swap="<<swap<<" ndisp="<<ndisp<<"\n";
}
void pfm(const fs::path& path,const cv::Mat& d) {
  std::ofstream f(path,std::ios::binary); if(!f) throw std::runtime_error("cannot write PFM");
  f<<"Pf\n"<<d.cols<<' '<<d.rows<<"\n-1.0\n";
  for(int y=d.rows-1;y>=0;--y) f.write(reinterpret_cast<const char*>(d.ptr<float>(y)),d.cols*sizeof(float));
}
void sgbm(const fs::path& dir,int ndisp) {
  auto l=read_pgm(dir/"left.pgm"),r=read_pgm(dir/"right.pgm");
  if(l.empty()||r.empty()||l.size()!=r.size()||ndisp<16||ndisp%16||ndisp>=l.cols) throw std::runtime_error("invalid matcher input");
  constexpr int block=5;
  auto matcher=cv::StereoSGBM::create(0,ndisp,block,8*block*block,32*block*block,1,31,10,100,2,cv::StereoSGBM::MODE_SGBM);
  cv::Mat raw,d;
  matcher->compute(l,r,raw); raw.convertTo(d,CV_32F,1.0/16);
  for(int y=0;y<d.rows;++y) for(int x=0;x<d.cols;++x) if(d.at<float>(y,x)<0 || d.at<float>(y,x)>=ndisp) d.at<float>(y,x)=INFINITY;
  pfm(dir/"sgbm.pfm",d);
  cv::Mat rf,lf,reverse_raw,reverse_d;
  cv::flip(r,rf,1); cv::flip(l,lf,1);
  matcher->compute(rf,lf,reverse_raw);
  reverse_raw.convertTo(reverse_d,CV_32F,1.0/16);
  cv::flip(reverse_d,reverse_d,1);
  for(int y=0;y<reverse_d.rows;++y) for(int x=0;x<reverse_d.cols;++x)
    if(reverse_d.at<float>(y,x)<0 || reverse_d.at<float>(y,x)>=ndisp) reverse_d.at<float>(y,x)=INFINITY;
  pfm(dir/"sgbm-right.pfm",reverse_d);
}
void selftest() {
  const double focal=500,baseline=0.2,depth=4;
  cv::Mat K=(cv::Mat_<double>(3,3)<<focal,0,320,0,focal,240,0,0,1),I=cv::Mat::eye(3,3,CV_64F),T=(cv::Mat_<double>(3,1)<<0,-baseline,0);
  cv::Mat R1,R2,P1,P2,Q;
  cv::stereoRectify(K,cv::Mat(),K,cv::Mat(),cv::Size(640,480),I,T,R1,R2,P1,P2,Q,cv::CALIB_ZERO_DISPARITY,0);
  if(std::abs(P2.at<double>(1,3))<=std::abs(P2.at<double>(0,3))) throw std::runtime_error("vertical axis check");
  Camera a{K,I,cv::Mat::zeros(3,1,CV_64F),{}},b{K,I,T,{}};
  auto p=project({0,0,depth},a,R1,P1,true),q=project({0,0,depth},b,R2,P2,true);
  if(std::abs(std::abs(p.x-q.x)-focal*baseline/depth)>1e-7 || std::abs(p.y-q.y)>1e-7 ||
     (p.x-q.x)*P2.at<double>(1,3)>=0) throw std::runtime_error("transposed project/sign check");
  T=(cv::Mat_<double>(3,1)<<baseline,0,0); b.t=T;
  cv::stereoRectify(K,cv::Mat(),K,cv::Mat(),cv::Size(640,480),I,T,R1,R2,P1,P2,Q,cv::CALIB_ZERO_DISPARITY,0);
  p=project({0,0,depth},a,R1,P1,false);q=project({0,0,depth},b,R2,P2,false);
  if(std::abs((q.x-p.x)-focal*baseline/depth)>1e-7 || std::abs(p.y-q.y)>1e-7 ||
     P2.at<double>(0,3)<=0) throw std::runtime_error("horizontal swap/project check");
  double source_c=2732,scale=768.0/5464;
  double corrected=source_c*scale-0.5;
  if(std::abs(corrected-383.5)>1e-10) throw std::runtime_error("COLMAP to OpenCV center/resize check");
  std::cout<<"geometry selftest passed\n";
}
int main(int argc,char** argv) {try {
  if(argc==2 && std::string(argv[1])=="selftest") selftest();
  else if(argc==4 && std::string(argv[1])=="prepare") prepare(argv[2],argv[3]);
  else if(argc==4 && std::string(argv[1])=="sgbm") sgbm(argv[2],std::stoi(argv[3]));
  else throw std::runtime_error("usage: tree_pair_oracle selftest | prepare SCENE OUTPUT | sgbm OUTPUT NDISP");
  return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}}
