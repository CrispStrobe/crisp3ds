// Test-only conflict-free multiview ORB tracks and anchored Ceres bundle refinement.
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct View {
  int id, width, height;
  std::string path;
  cv::Matx33d K,R;
  cv::Vec3d t,center;
  std::array<double,6> camera;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};
struct Edge {int a,b; float distance;};
struct Observation {int view,key; double u,v;};
struct Track {cv::Vec3d xyz; std::vector<Observation> obs; int held=-1;};
struct DSU {
  std::vector<int> parent,views;
  explicit DSU(int n):parent(n),views(n,0) {std::iota(parent.begin(),parent.end(),0);}
  int find(int i) {return parent[i]==i?i:parent[i]=find(parent[i]);}
  bool join(int i,int j) {
    i=find(i);j=find(j);if(i==j)return true;
    if(views[i]&views[j])return false;
    if(i>j)std::swap(i,j);parent[j]=i;views[i]|=views[j];return true;
  }
};
cv::Matx34d matrix(const View& v) {
  cv::Matx34d p;
  for(int r=0;r<3;++r){for(int c=0;c<3;++c)p(r,c)=v.R(r,c);p(r,3)=v.t[r];}
  return v.K*p;
}
bool project(const View& v,const cv::Vec3d& x,double& u,double& w,double& depth) {
  cv::Vec3d q=v.R*x+v.t;depth=q[2];
  if(!(depth>1e-9))return false;
  u=v.K(0,0)*q[0]/depth+v.K(0,2);w=v.K(1,1)*q[1]/depth+v.K(1,2);
  return std::isfinite(u)&&std::isfinite(w);
}
bool triangulate(const View& a,const View& b,const cv::Point2f& p,const cv::Point2f& q,cv::Vec3d& x) {
  cv::Mat points;
  cv::triangulatePoints(cv::Mat(matrix(a)),cv::Mat(matrix(b)),
      std::vector<cv::Point2d>{{p.x,p.y}},std::vector<cv::Point2d>{{q.x,q.y}},points);
  if(points.type()!=CV_64FC1)return false;
  double w=points.at<double>(3,0);
  if(!std::isfinite(w)||std::abs(w)<1e-12)return false;
  for(int k=0;k<3;++k)x[k]=points.at<double>(k,0)/w;
  if(!std::isfinite(x[0])||!std::isfinite(x[1])||!std::isfinite(x[2]))return false;
  double ua,va,za,ub,vb,zb;
  if(!project(a,x,ua,va,za)||!project(b,x,ub,vb,zb))return false;
  if(std::hypot(ua-p.x,va-p.y)>2||std::hypot(ub-q.x,vb-q.y)>2)return false;
  cv::Vec3d ra=x-a.center,rb=x-b.center;
  double cosine=ra.dot(rb)/(cv::norm(ra)*cv::norm(rb));
  return std::isfinite(cosine)&&cosine<=std::cos(CV_PI/180);
}
struct Residual {
  double fx,fy,cx,cy,u,v;
  template<class T> bool operator()(const T* camera,const T* point,T* r)const {
    T q[3];ceres::AngleAxisRotatePoint(camera,point,q);
    for(int k=0;k<3;++k)q[k]+=camera[k+3];
    r[0]=T(fx)*q[0]/q[2]+T(cx)-T(u);
    r[1]=T(fy)*q[1]/q[2]+T(cy)-T(v);
    return true;
  }
};
double rms(const std::vector<View>& views,const std::vector<Track>& tracks,bool held,int& count,int& behind) {
  double sum=0;count=0;behind=0;
  for(auto const& track:tracks)for(size_t k=0;k<track.obs.size();++k) {
    if((static_cast<int>(k)==track.held)!=held)continue;
    auto const& o=track.obs[k];auto const& v=views[o.view];
    double q[3];ceres::AngleAxisRotatePoint(v.camera.data(),track.xyz.val,q);
    for(int i=0;i<3;++i)q[i]+=v.camera[i+3];
    if(q[2]<=0){++behind;continue;}
    double du=v.K(0,0)*q[0]/q[2]+v.K(0,2)-o.u;
    double dv=v.K(1,1)*q[1]/q[2]+v.K(1,2)-o.v;
    sum+=du*du+dv*dv;++count;
  }
  return count?std::sqrt(sum/count):0;
}
int selftest() {
  DSU d(4);d.views={1,2,4,1};
  if(!d.join(0,1)||d.join(0,3)||!d.join(1,2))return 1;
  View a{},b{};a.K=b.K=cv::Matx33d(1000,0,320,0,1000,240,0,0,1);
  a.R=b.R=cv::Matx33d::eye();a.t={0,0,0};b.t={-1,0,0};
  a.center={0,0,0};b.center={1,0,0};cv::Vec3d x;
  if(!triangulate(a,b,{320,240},{220,240},x)||cv::norm(x-cv::Vec3d(0,0,10))>1e-6)return 2;
  if(triangulate(a,b,{320,240},{320,240},x))return 3;
  double test_camera[6]={.01,-.02,.005,-.4,.03,.02};
  double test_point[3]={.2,-.1,5};
  ceres::AutoDiffCostFunction<Residual,2,6,3> jacobian(
      new Residual{1000,1000,320,240,300,220});
  double const* blocks[]={test_camera,test_point};
  double residual[2],j_camera[12],j_point[6];double* jacobians[]={j_camera,j_point};
  if(!jacobian.Evaluate(blocks,residual,jacobians))return 4;
  for(int block=0;block<2;++block)for(int k=0;k<(block?3:6);++k) {
    double* parameter=block?test_point:test_camera;double old=parameter[k],plus[2],minus[2];
    parameter[k]=old+1e-6;jacobian.Evaluate(blocks,plus,nullptr);
    parameter[k]=old-1e-6;jacobian.Evaluate(blocks,minus,nullptr);parameter[k]=old;
    for(int row=0;row<2;++row) {
      double analytic=block?j_point[row*3+k]:j_camera[row*6+k];
      if(std::abs(analytic-(plus[row]-minus[row])/2e-6)>1e-5)return 5;
    }
  }
  // A perturbed third camera should improve while both gauge anchors stay exact.
  std::array<double,6> cam[3]={{{0,0,0,0,0,0}},{{0,0,0,-1,0,0}},{{0,0,0,-.35,.03,0}}};
  const auto fixed0=cam[0],fixed1=cam[1];
  std::array<std::array<double,3>,4> pts{{{{-.3,-.2,5}},{{.2,.3,6}},{{.5,-.1,4}},{{-.4,.4,7}}}};
  ceres::Problem problem;double before=0;
  for(auto& point:pts)for(int i=0;i<3;++i){
    double true_tx=i==0?0:(i==1?-1:-.4);
    double u=1000*(point[0]+true_tx)/point[2]+320;
    double v=1000*point[1]/point[2]+240;
    double r[2];Residual{1000,1000,320,240,u,v}(cam[i].data(),point.data(),r);
    before+=r[0]*r[0]+r[1]*r[1];
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Residual,2,6,3>(
        new Residual{1000,1000,320,240,u,v}),new ceres::HuberLoss(1),cam[i].data(),point.data());
  }
  problem.SetParameterBlockConstant(cam[0].data());problem.SetParameterBlockConstant(cam[1].data());
  ceres::Solver::Options options;options.linear_solver_type=ceres::DENSE_SCHUR;
  options.max_num_iterations=30;options.minimizer_progress_to_stdout=false;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  if(!summary.IsSolutionUsable()||cam[0]!=fixed0||cam[1]!=fixed1||
      std::abs(cam[2][3]+.4)>1e-4||before<=0)return 4;
  std::cout<<"conflict, triangulation, anchored BA tests passed\n";return 0;
}
int main(int argc,char** argv) {
  if(argc==2&&std::string(argv[1])=="--self-test")return selftest();
  if(argc!=3&&!(argc==4&&std::string(argv[3])=="--no-ba")){
    std::cerr<<"usage: tree_refine views.tsv output.txt [--no-ba]\n";return 2;}
  bool no_ba=argc==4;
  if(std::filesystem::exists(argv[2])){std::cerr<<"output exists\n";return 2;}
  std::ifstream in(argv[1]);if(!in)return 2;
  std::vector<View> views;std::string line;
  auto orb=cv::ORB::create(3000,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20);
  while(std::getline(in,line)) {
    if(line.empty())continue;std::istringstream row(line);View v{};double fx,fy,cx,cy;
    row>>v.id>>v.width>>v.height>>v.path>>fx>>fy>>cx>>cy;
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)row>>v.R(r,c);
    for(int k=0;k<3;++k)row>>v.t[k];
    if(!row||views.size()>=10||v.id!=static_cast<int>(views.size())||
       v.width!=768||v.height!=512||!std::isfinite(fx)||!std::isfinite(fy)||
       !std::isfinite(cx)||!std::isfinite(cy)||fx<=0||fy<=0)return 3;
    for(double x:v.R.val)if(!std::isfinite(x))return 3;
    for(double x:v.t.val)if(!std::isfinite(x))return 3;
    v.K=cv::Matx33d(fx,0,cx,0,fy,cy,0,0,1);v.center=-(v.R.t()*v.t);
    double colmajor[9];for(int r=0;r<3;++r)for(int c=0;c<3;++c)colmajor[c*3+r]=v.R(r,c);
    ceres::RotationMatrixToAngleAxis(colmajor,v.camera.data());
    for(int k=0;k<3;++k)v.camera[k+3]=v.t[k];
    cv::Mat gray=cv::imread(v.path,cv::IMREAD_GRAYSCALE);
    if(gray.empty()||gray.cols!=v.width||gray.rows!=v.height)return 4;
    orb->detectAndCompute(gray,cv::noArray(),v.keypoints,v.descriptors);
    views.push_back(std::move(v));
  }
  if(views.size()!=10)return 5;
  std::vector<int> offset(views.size()+1,0);
  for(size_t i=0;i<views.size();++i)offset[i+1]=offset[i]+views[i].keypoints.size();
  DSU d(offset.back());for(size_t i=0;i<views.size();++i)
    for(int k=offset[i];k<offset[i+1];++k)d.views[k]=1<<i;
  cv::BFMatcher matcher(cv::NORM_HAMMING);std::vector<Edge> edges;
  int pair_seeds=0;
  for(size_t i=0;i<views.size();++i)for(size_t j=i+1;j<views.size();++j) {
    auto const& a=views[i];auto const& b=views[j];
    if(a.descriptors.empty()||b.descriptors.empty())continue;
    std::vector<std::vector<cv::DMatch>> ab,ba;
    matcher.knnMatch(a.descriptors,b.descriptors,ab,2);
    matcher.knnMatch(b.descriptors,a.descriptors,ba,2);
    std::vector<int> reverse(b.keypoints.size(),-1);
    for(size_t k=0;k<ba.size();++k)if(ba[k].size()==2&&ba[k][0].distance<.8f*ba[k][1].distance)
      reverse[k]=ba[k][0].trainIdx;
    int kept=0;
    for(size_t k=0;k<ab.size()&&kept<500&&pair_seeds<10000;++k) {
      if(ab[k].size()!=2||ab[k][0].distance>=.8f*ab[k][1].distance)continue;
      auto m=ab[k][0];if(reverse[m.trainIdx]!=static_cast<int>(k))continue;
      cv::Vec3d x;
      if(!triangulate(a,b,a.keypoints[k].pt,b.keypoints[m.trainIdx].pt,x))continue;
      edges.push_back({offset[i]+static_cast<int>(k),offset[j]+m.trainIdx,m.distance});
      ++kept;++pair_seeds;
    }
  }
  std::sort(edges.begin(),edges.end(),[](auto const& a,auto const& b){
    if(a.distance!=b.distance)return a.distance<b.distance;
    return std::tie(a.a,a.b)<std::tie(b.a,b.b);
  });
  int conflicts=0;for(auto const& e:edges)if(!d.join(e.a,e.b))++conflicts;
  std::map<int,std::vector<Observation>> components;
  for(size_t i=0;i<views.size();++i)for(int k=0;k<static_cast<int>(views[i].keypoints.size());++k) {
    int global=offset[i]+k,root=d.find(global);
    if(d.views[root]&(d.views[root]-1)) {
      auto p=views[i].keypoints[k].pt;
      components[root].push_back({static_cast<int>(i),k,p.x,p.y});
    }
  }
  std::vector<Track> tracks;int lengths[11]={},rejected=0;
  for(auto& [root,obs]:components) {
    if(obs.size()<2)continue;
    // Reserve one observation from every 3+ view component. It can establish
    // correspondence, but cannot initialize, filter, or optimize geometry.
    int held=obs.size()>=3?static_cast<int>(obs.size()-1):-1;
    cv::Vec3d best;double best_err=1e99;
    for(int a=0;a<static_cast<int>(obs.size())-(held>=0);++a)
      for(int b=a+1;b<static_cast<int>(obs.size())-(held>=0);++b) {
      auto& oa=obs[a];auto& ob=obs[b];cv::Vec3d x;
      if(!triangulate(views[oa.view],views[ob.view],views[oa.view].keypoints[oa.key].pt,
          views[ob.view].keypoints[ob.key].pt,x))continue;
      double sum=0;bool valid=true;
      for(int k=0;k<static_cast<int>(obs.size());++k){
        if(k==held)continue;auto& o=obs[k];double u,v,z;
        if(!project(views[o.view],x,u,v,z)){valid=false;break;}
        double e=std::hypot(u-o.u,v-o.v);if(e>4){valid=false;break;}sum+=e*e;}
      if(valid&&sum<best_err){best_err=sum;best=x;}
    }
    if(best_err==1e99){++rejected;continue;}
    Track track{best,obs,held};
    lengths[obs.size()]++;tracks.push_back(std::move(track));
  }
  if(tracks.empty())return 6;
  int initial_train_n,initial_held_n,behind;
  double initial_train=rms(views,tracks,false,initial_train_n,behind);
  double initial_held=rms(views,tracks,true,initial_held_n,behind);
  // Every optimized camera must be tied through training tracks to an anchor.
  // Unsupported cameras remain at the supplied pose.
  std::vector<int> support(views.size(),0),connected(views.size(),0);
  std::vector<std::vector<int>> adjacency(views.size());
  for(auto const& track:tracks){std::vector<int> indices;
    for(size_t k=0;k<track.obs.size();++k)if(static_cast<int>(k)!=track.held){
      int view=track.obs[k].view;++support[view];indices.push_back(view);}
    for(int a:indices)for(int b:indices)if(a!=b)adjacency[a].push_back(b);
  }
  if(!support[0]||!support[1]){std::cerr<<"anchor lacks training tracks\n";return 7;}
  std::vector<int> queue{0};connected[0]=1;
  for(size_t i=0;i<queue.size();++i)for(int next:adjacency[queue[i]])
    if(!connected[next]){connected[next]=1;queue.push_back(next);}
  for(size_t i=0;i<views.size();++i)if(support[i]&&!connected[i]){
    std::cerr<<"training graph disconnected from anchors\n";return 7;}
  ceres::Problem problem;
  for(auto& track:tracks)for(size_t k=0;k<track.obs.size();++k) {
    if(static_cast<int>(k)==track.held)continue;
    auto& o=track.obs[k];auto& v=views[o.view];
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Residual,2,6,3>(
        new Residual{v.K(0,0),v.K(1,1),v.K(0,2),v.K(1,2),o.u,o.v}),
        new ceres::HuberLoss(1),v.camera.data(),track.xyz.val);
  }
  // Fix two entire world-to-camera poses: removes similarity gauge and retains source baseline.
  if(!problem.HasParameterBlock(views[0].camera.data())||
     !problem.HasParameterBlock(views[1].camera.data()))return 7;
  problem.SetParameterBlockConstant(views[0].camera.data());
  problem.SetParameterBlockConstant(views[1].camera.data());
  ceres::Solver::Options options;options.max_num_iterations=30;
  options.linear_solver_type=ceres::DENSE_SCHUR;options.num_threads=2;
  ceres::Solver::Summary summary;
  if(!no_ba){ceres::Solve(options,&problem,&summary);if(!summary.IsSolutionUsable())return 7;}
  int final_train_n,final_held_n,behind_train,behind_held;
  double final_train=rms(views,tracks,false,final_train_n,behind_train);
  double final_held=rms(views,tracks,true,final_held_n,behind_held);
  if(behind_train||behind_held||!std::isfinite(final_train)||!std::isfinite(final_held)||
     final_train_n!=initial_train_n||final_held_n!=initial_held_n)return 8;
  std::ofstream out(argv[2]);if(!out)return 9;out<<std::setprecision(17);
  out<<"SUMMARY "<<pair_seeds<<' '<<edges.size()<<' '<<conflicts<<' '<<rejected<<' '<<tracks.size()
     <<' '<<initial_train_n<<' '<<initial_held_n<<' '<<initial_train<<' '<<final_train
     <<' '<<initial_held<<' '<<final_held<<' '<<summary.iterations.size()
     <<' '<<std::count_if(support.begin(),support.end(),[](int n){return n>0;})
     <<' '<<std::count(connected.begin(),connected.end(),1)<<' '<<no_ba<<'\n';
  out<<"LENGTHS";for(int length=2;length<=10;++length)out<<' '<<lengths[length];out<<'\n';
  for(auto& v:views){double R[9];ceres::AngleAxisToRotationMatrix(v.camera.data(),R);
    out<<"CAMERA "<<v.id;for(int r=0;r<3;++r)for(int c=0;c<3;++c)out<<' '<<R[c*3+r];
    for(int k=0;k<3;++k)out<<' '<<v.camera[k+3];out<<'\n';}
  for(auto const& track:tracks){out<<"POINT "<<track.xyz[0]<<' '<<track.xyz[1]<<' '<<track.xyz[2]
    <<' '<<track.obs.size();for(auto const& o:track.obs)out<<' '<<o.view<<' '<<o.key<<' '<<o.u<<' '<<o.v;
    out<<'\n';}
  return 0;
}
