// Fixed-camera, test-only multiview correspondence diagnostic.
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

struct View {
  int id=0,width=0,height=0;
  std::string path;
  cv::Matx33d K,R;
  cv::Vec3d t,center;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};
struct Edge {int a,b;float distance;};
struct Obs {int view,key;cv::Point2f pixel;int global;};
struct Component {std::vector<Obs> obs;};
struct DSU {
  std::vector<int> parent,mask;
  explicit DSU(int n):parent(n),mask(n) {std::iota(parent.begin(),parent.end(),0);}
  int find(int i){return parent[i]==i?i:parent[i]=find(parent[i]);}
  bool join(int a,int b){a=find(a);b=find(b);if(a==b)return true;
    if(mask[a]&mask[b])return false;
    if(a>b)std::swap(a,b);parent[b]=a;mask[a]|=mask[b];return true;}
};
cv::Matx34d camera_matrix(const View& v){cv::Matx34d p;
  for(int r=0;r<3;++r){for(int c=0;c<3;++c)p(r,c)=v.R(r,c);p(r,3)=v.t[r];}
  return v.K*p;}
bool project(const View& v,const cv::Vec3d& x,cv::Point2d& p){
  const cv::Vec3d q=v.R*x+v.t;
  if(!(q[2]>1e-9))return false;
  p={v.K(0,0)*q[0]/q[2]+v.K(0,2),v.K(1,1)*q[1]/q[2]+v.K(1,2)};
  return std::isfinite(p.x)&&std::isfinite(p.y);
}
bool triangulate(const View& a,const View& b,cv::Point2f pa,cv::Point2f pb,cv::Vec3d& x){
  cv::Mat h;cv::triangulatePoints(cv::Mat(camera_matrix(a)),cv::Mat(camera_matrix(b)),
    std::vector<cv::Point2d>{{pa.x,pa.y}},std::vector<cv::Point2d>{{pb.x,pb.y}},h);
  if(h.type()!=CV_64FC1)return false;
  const double w=h.at<double>(3,0);if(!std::isfinite(w)||std::abs(w)<1e-12)return false;
  for(int k=0;k<3;++k)x[k]=h.at<double>(k,0)/w;
  cv::Point2d p,q;if(!project(a,x,p)||!project(b,x,q))return false;
  if(cv::norm(p-cv::Point2d(pa))>2||cv::norm(q-cv::Point2d(pb))>2)return false;
  cv::Vec3d ra=x-a.center,rb=x-b.center;
  double cosine=ra.dot(rb)/(cv::norm(ra)*cv::norm(rb));
  return std::isfinite(cosine)&&cosine<=std::cos(CV_PI/180);
}
double error(const View& v,const cv::Vec3d& x,cv::Point2f pixel){
  cv::Point2d p;if(!project(v,x,p))return 1e9;
  return cv::norm(p-cv::Point2d(pixel));
}
struct Fit {bool valid=false;cv::Vec3d x;double rms=0,max=0;};
Fit fit(const std::vector<View>& views,const std::vector<Obs>& obs,int n){
  Fit best;double best_sum=1e100;
  for(int i=0;i<n;++i)for(int j=i+1;j<n;++j){
    cv::Vec3d x;if(!triangulate(views[obs[i].view],views[obs[j].view],obs[i].pixel,obs[j].pixel,x))continue;
    double sum=0,maxerr=0;bool finite=true;
    for(int k=0;k<n;++k){double e=error(views[obs[k].view],x,obs[k].pixel);
      if(e>=1e9){finite=false;break;}sum+=e*e;maxerr=std::max(maxerr,e);}
    if(finite&&sum<best_sum){best_sum=sum;best={true,x,std::sqrt(sum/n),maxerr};}
  }
  return best;
}
bool cycle_supported(const std::vector<Obs>& obs,const std::set<std::pair<int,int>>& links){
  if(obs.size()<3)return false;
  std::vector<int> participation(obs.size());
  for(size_t i=0;i<obs.size();++i)for(size_t j=i+1;j<obs.size();++j)
    for(size_t k=j+1;k<obs.size();++k){
      auto has=[&](int a,int b){return links.count(std::minmax(obs[a].global,obs[b].global))!=0;};
      if(has(i,j)&&has(i,k)&&has(j,k)){++participation[i];++participation[j];++participation[k];}
    }
  return std::all_of(participation.begin(),participation.end(),[](int n){return n>0;});
}
double percentile(std::vector<double> v,double fraction){
  if(v.empty())return -1;std::sort(v.begin(),v.end());
  return v[std::min(v.size()-1,static_cast<size_t>(fraction*(v.size()-1)))];
}
int self_test(){
  View a,b,c;a.K=b.K=c.K=cv::Matx33d(1000,0,320,0,1000,240,0,0,1);
  a.R=b.R=c.R=cv::Matx33d::eye();a.t={0,0,0};b.t={-1,0,0};c.t={0,-1,0};
  a.center={0,0,0};b.center={1,0,0};c.center={0,1,0};
  std::vector<View> views{a,b,c};cv::Vec3d x{0,0,10};
  std::vector<Obs> good{{0,0,{320,240},0},{1,0,{220,240},1},{2,0,{320,140},2}};
  std::set<std::pair<int,int>> triangle{{0,1},{0,2},{1,2}},chain{{0,1},{1,2}};
  if(!cycle_supported(good,triangle)||cycle_supported(good,chain))return 1;
  Fit f=fit(views,good,3);if(!f.valid||f.max>1e-6||cv::norm(f.x-x)>1e-6)return 2;
  good[2].pixel={500,140};
  if(!cycle_supported(good,triangle)||fit(views,good,3).max<4)return 3;
  // Both chain links pass pair geometry yet refer to distinct world points.
  std::vector<Obs> false_chain{{0,0,{320,240},0},{1,0,{220,240},1},{2,0,{270,190},2}};
  cv::Vec3d x01,x12;
  if(!triangulate(views[0],views[1],false_chain[0].pixel,false_chain[1].pixel,x01)||
     !triangulate(views[1],views[2],false_chain[1].pixel,false_chain[2].pixel,x12)||
     cv::norm(x01-x12)<1||cycle_supported(false_chain,chain))return 6;
  cv::Point2d p;if(!project(views[0],x,p)||p!=cv::Point2d(320,240))return 4;
  // A COLMAP center of (320.5,240.5) maps to OpenCV (320,240).
  views[0].K(0,2)=320.5-0.5;views[0].K(1,2)=240.5-0.5;
  if(!project(views[0],x,p)||p!=cv::Point2d(320,240))return 5;
  std::cout<<"known-good triple, false chain, inconsistent third view, pixel convention passed\n";return 0;
}
int main(int argc,char** argv){
  if(argc==2&&std::string(argv[1])=="--self-test")return self_test();
  bool sift=argc==4&&std::string(argv[3])=="--sift";
  if(argc!=3&&!sift){std::cerr<<"usage: tree_tracks views.tsv report.json [--sift]\n";return 2;}
  if(std::filesystem::exists(argv[2])){std::cerr<<"report exists\n";return 2;}
  std::ifstream in(argv[1]);if(!in)return 2;
  std::vector<View> views;std::string line;
  cv::Ptr<cv::Feature2D> detector=sift?cv::Ptr<cv::Feature2D>(cv::SIFT::create(3000)):
    cv::Ptr<cv::Feature2D>(cv::ORB::create(3000,1.2f,8,31,0,2,cv::ORB::HARRIS_SCORE,31,20));
  while(std::getline(in,line)){
    if(line.empty())continue;std::istringstream row(line);View v;double fx,fy,cx,cy;
    row>>v.id>>v.width>>v.height>>v.path>>fx>>fy>>cx>>cy;
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)row>>v.R(r,c);
    for(int k=0;k<3;++k)row>>v.t[k];
    if(!row||views.size()>=10||v.id!=static_cast<int>(views.size())||v.width!=768||v.height!=512||
       !std::isfinite(fx)||!std::isfinite(fy)||!std::isfinite(cx)||!std::isfinite(cy)||fx<=0||fy<=0)return 3;
    for(double value:v.R.val)if(!std::isfinite(value))return 3;
    for(double value:v.t.val)if(!std::isfinite(value))return 3;
    cv::Matx33d check=v.R*v.R.t();
    if(cv::norm(cv::Mat(check),cv::Mat(cv::Matx33d::eye()),cv::NORM_INF)>1e-3||
       std::abs(cv::determinant(cv::Mat(v.R))-1)>1e-3)return 3;
    v.K=cv::Matx33d(fx,0,cx,0,fy,cy,0,0,1);v.center=-(v.R.t()*v.t);
    cv::Mat gray=cv::imread(v.path,cv::IMREAD_GRAYSCALE);
    if(gray.empty()||gray.cols!=v.width||gray.rows!=v.height)return 4;
    detector->detectAndCompute(gray,cv::noArray(),v.keypoints,v.descriptors);
    views.push_back(std::move(v));
  }
  if(views.size()!=10)return 5;
  std::vector<int> offset(11);for(int i=0;i<10;++i)offset[i+1]=offset[i]+views[i].keypoints.size();
  DSU d(offset.back());for(int i=0;i<10;++i)for(int k=offset[i];k<offset[i+1];++k)d.mask[k]=1<<i;
  cv::BFMatcher matcher(sift?cv::NORM_L2:cv::NORM_HAMMING);std::vector<Edge> edges;
  for(int i=0;i<10;++i)for(int j=i+1;j<10;++j){
    auto& a=views[i];auto& b=views[j];if(a.descriptors.empty()||b.descriptors.empty())continue;
    std::vector<std::vector<cv::DMatch>> ab,ba;
    matcher.knnMatch(a.descriptors,b.descriptors,ab,2);
    matcher.knnMatch(b.descriptors,a.descriptors,ba,2);
    std::vector<int> reverse(b.keypoints.size(),-1);
    for(size_t k=0;k<ba.size();++k)if(ba[k].size()==2&&ba[k][0].distance<.8f*ba[k][1].distance)
      reverse[k]=ba[k][0].trainIdx;
    int kept=0;
    for(size_t k=0;k<ab.size()&&kept<500&&edges.size()<10000;++k){
      if(ab[k].size()!=2||ab[k][0].distance>=.8f*ab[k][1].distance)continue;
      auto m=ab[k][0];if(reverse[m.trainIdx]!=static_cast<int>(k))continue;
      cv::Vec3d x;if(!triangulate(a,b,a.keypoints[k].pt,b.keypoints[m.trainIdx].pt,x))continue;
      edges.push_back({offset[i]+static_cast<int>(k),offset[j]+m.trainIdx,m.distance});++kept;
    }
  }
  std::sort(edges.begin(),edges.end(),[](const Edge& a,const Edge& b){
    if(a.distance!=b.distance)return a.distance<b.distance;
    return std::tie(a.a,a.b)<std::tie(b.a,b.b);});
  std::set<std::pair<int,int>> links;int conflicts=0;
  for(const Edge& e:edges){links.insert(std::minmax(e.a,e.b));if(!d.join(e.a,e.b))++conflicts;}
  std::map<int,Component> components;
  for(int i=0;i<10;++i)for(int k=0;k<static_cast<int>(views[i].keypoints.size());++k){
    int global=offset[i]+k,root=d.find(global);
    if(d.mask[root]&(d.mask[root]-1))components[root].obs.push_back({i,k,views[i].keypoints[k].pt,global});
  }
  std::map<std::string,int> counts;std::map<int,int> lengths,accepted_lengths;
  std::map<std::string,std::vector<double>> residuals;
  std::vector<double> all_errors,accepted_errors;
  int all_observations=0,accepted_observations=0;
  int invalid_reserved=0,invalid_leave_one_out=0;
  std::vector<std::string> details;
  for(auto& [root,component]:components){auto& obs=component.obs;
    std::sort(obs.begin(),obs.end(),[](const Obs& a,const Obs& b){return a.view<b.view;});
    if(obs.size()<2)continue;++lengths[obs.size()];
    if(obs.size()==2){++counts["pair_only"];continue;}
    // Recreate the earlier reserved-pixel diagnostic for every original 3+ component.
    Fit training=fit(views,obs,static_cast<int>(obs.size()-1));
    double reserved=training.valid?error(views[obs.back().view],training.x,obs.back().pixel):1e9;
    bool reserved_valid=training.valid&&reserved<1e9;
    if(!reserved_valid)++invalid_reserved;
    if(training.valid&&training.max<=4&&reserved_valid)residuals["old_training_fit_reserved"].push_back(reserved);
    bool cycle=cycle_supported(obs,links);
    Fit full=fit(views,obs,static_cast<int>(obs.size()));
    std::string category;
    if(!cycle)category="no_cycle";
    else if(!full.valid)category="no_valid_triangulation";
    else if(full.max>4)category="multiview_reprojection";
    else category="accepted";
    ++counts[category];
    if(reserved_valid)residuals[category+"_reserved"].push_back(reserved);
    if(full.valid)residuals[category+"_full_max"].push_back(full.max);
    std::vector<cv::Vec3d> edge_points;
    for(size_t i=0;i<obs.size();++i)for(size_t j=i+1;j<obs.size();++j){
      if(!links.count(std::minmax(obs[i].global,obs[j].global)))continue;
      cv::Vec3d x;if(triangulate(views[obs[i].view],views[obs[j].view],obs[i].pixel,obs[j].pixel,x))edge_points.push_back(x);
    }
    double pair_disagreement=0;
    for(size_t i=0;i<edge_points.size();++i)for(size_t j=i+1;j<edge_points.size();++j)
      pair_disagreement=std::max(pair_disagreement,cv::norm(edge_points[i]-edge_points[j]));
    if(edge_points.size()>=2)residuals[category+"_pair_3d_disagreement"].push_back(pair_disagreement);
    std::vector<double> loo;int invalid_loo=0;
    for(size_t omit=0;omit<obs.size();++omit){
      std::vector<Obs> other;for(size_t k=0;k<obs.size();++k)if(k!=omit)other.push_back(obs[k]);
      bool linked=false;
      for(size_t i=0;i<other.size();++i)for(size_t j=i+1;j<other.size();++j)
        linked|=links.count(std::minmax(other[i].global,other[j].global))!=0;
      Fit f=linked?fit(views,other,static_cast<int>(other.size())):Fit{};
      double e=f.valid?error(views[obs[omit].view],f.x,obs[omit].pixel):1e9;
      if(e<1e9){loo.push_back(e);residuals[category+"_leave_one_out"].push_back(e);}
      else {++invalid_loo;++invalid_leave_one_out;}
    }
    std::ostringstream detail;detail<<std::setprecision(10);
    detail<<"{\"root\": "<<root<<", \"views\": [";
    for(size_t k=0;k<obs.size();++k){if(k)detail<<", ";detail<<obs[k].view;}
    detail<<"], \"category\": \""<<category<<"\", \"edge_points\": "<<edge_points.size()
          <<", \"pair_3d_max_disagreement_source_units\": "<<pair_disagreement
          <<", \"leave_one_out_valid\": "<<loo.size()<<", \"leave_one_out_invalid\": "<<invalid_loo
          <<", \"leave_one_out_max_px\": "<<percentile(loo,1)
          <<", \"full_fit_max_px\": "<<(full.valid?full.max:-1)
          <<", \"old_reserved_px\": "<<(reserved_valid?reserved:-1)<<"}";
    details.push_back(detail.str());
    if(category=="accepted"){
      ++accepted_lengths[obs.size()];accepted_observations+=obs.size();
      for(const Obs& o:obs)accepted_errors.push_back(error(views[o.view],full.x,o.pixel));
    }
    if(training.valid&&training.max<=4&&reserved_valid){
      ++all_observations;all_errors.push_back(reserved);
    }
  }
  std::ofstream out(argv[2]);if(!out)return 6;out<<std::setprecision(10);
  out<<"{\n  \"settings\": {\"descriptor\": \""<<(sift?"SIFT_L2":"ORB_HAMMING")<<"\", \"features_per_view\": 3000, \"ratio\": 0.8, \"pair_reprojection_px\": 2, \"minimum_angle_deg\": 1, \"multiview_max_px\": 4, \"minimum_cycle_views\": 3, \"camera_poses_fixed\": true},\n";
  out<<"  \"pair_links\": "<<edges.size()<<", \"same_view_conflicts\": "<<conflicts<<", \"components\": "<<components.size()<<",\n";
  auto write_map=[&](const char* name,const auto& map){out<<"  \""<<name<<"\": {";bool first=true;
    for(const auto& [key,val]:map){if(!first)out<<", ";first=false;out<<"\""<<key<<"\": "<<val;}out<<"},\n";};
  write_map("lengths_all",lengths);write_map("categories",counts);write_map("lengths_accepted",accepted_lengths);
  out<<"  \"original_reserved_count\": "<<all_observations<<", \"invalid_reserved\": "<<invalid_reserved<<", \"invalid_leave_one_out\": "<<invalid_leave_one_out<<", \"original_reserved_median_px\": "<<percentile(all_errors,.5)<<", \"original_reserved_p90_px\": "<<percentile(all_errors,.9)<<", \"original_reserved_max_px\": "<<percentile(all_errors,1)<<",\n";
  out<<"  \"accepted_observation_count\": "<<accepted_observations<<", \"accepted_observation_median_px\": "<<percentile(accepted_errors,.5)<<", \"accepted_observation_p90_px\": "<<percentile(accepted_errors,.9)<<", \"accepted_observation_max_px\": "<<percentile(accepted_errors,1)<<",\n";
  out<<"  \"diagnostic_populations\": {";bool first=true;
  for(const auto& [name,values]:residuals){if(!first)out<<",";first=false;
    const char* unit=name.find("pair_3d_disagreement")!=std::string::npos?"source_units":"px";
    out<<"\n    \""<<name<<"\": {\"count\": "<<values.size()
       <<", \"median_"<<unit<<"\": "<<percentile(values,.5)
       <<", \"p90_"<<unit<<"\": "<<percentile(values,.9)
       <<", \"max_"<<unit<<"\": "<<percentile(values,1)<<"}";}
  out<<"\n  },\n  \"multiview_components\": [\n";
  for(size_t i=0;i<details.size();++i){if(i)out<<",\n";out<<"    "<<details[i];}
  out<<"\n  ]\n}\n";
  return 0;
}
