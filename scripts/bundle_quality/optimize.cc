// Test-only anchored joint camera/observed-track bundle adjustment.
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Observation {int camera, point; double u,v;};
struct Marker {int camera; double x,y,z,u,v;};

struct TrackResidual {
  double u,v;
  template<class T> bool operator()(const T* camera,const T* point,T* residual) const {
    T q[3]; ceres::AngleAxisRotatePoint(camera,point,q);
    q[0]+=camera[3];q[1]+=camera[4];q[2]+=camera[5];
    residual[0]=T(1500)*q[0]/q[2]+T(800)-T(u);
    residual[1]=T(1500)*q[1]/q[2]+T(600)-T(v);
    return true;
  }
};
struct MarkerResidual {
  double x,y,z,u,v;
  template<class T> bool operator()(const T* camera,T* residual) const {
    T point[3]={T(x),T(y),T(z)},q[3];
    ceres::AngleAxisRotatePoint(camera,point,q);
    q[0]+=camera[3];q[1]+=camera[4];q[2]+=camera[5];
    residual[0]=T(1500)*q[0]/q[2]+T(800)-T(u);
    residual[1]=T(1500)*q[1]/q[2]+T(600)-T(v);
    return true;
  }
};

struct Diagnostics {double track_rms,marker_rms,min_z;int nonpositive;};
Diagnostics diagnostic(const std::vector<std::array<double,6>>& cams,
                       const std::vector<std::array<double,3>>& pts,
                       const std::vector<Observation>& obs,const std::vector<Marker>& markers) {
  double tracks=0,board=0,min_z=std::numeric_limits<double>::infinity();int nonpositive=0;
  auto projected=[&](int id,const double* point,double u,double v,double& sum) {
    double q[3];ceres::AngleAxisRotatePoint(cams[id].data(),point,q);
    for(int i=0;i<3;++i)q[i]+=cams[id][i+3];
    min_z=std::min(min_z,q[2]);nonpositive+=(q[2]<=0);
    double dx=1500*q[0]/q[2]+800-u,dy=1500*q[1]/q[2]+600-v;
    sum+=dx*dx+dy*dy;
  };
  for(auto& o:obs)projected(o.camera,pts[o.point].data(),o.u,o.v,tracks);
  for(auto& m:markers) {double p[3]={m.x,m.y,m.z};projected(m.camera,p,m.u,m.v,board);}
  return {std::sqrt(tracks/obs.size()),std::sqrt(board/markers.size()),min_z,nonpositive};
}

int self_test() {
  double c[6]={.02,-.01,.005,5,-3,500},p[3]={30,40,-45};
  ceres::AutoDiffCostFunction<TrackResidual,2,6,3> cost(new TrackResidual{890,585});
  double r[2],jcam[12],jpoint[6];
  double const* blocks[]={c,p};double* jac[]={jcam,jpoint};
  if(!cost.Evaluate(blocks,r,jac))return 2;
  for(int block=0;block<2;++block)for(int k=0;k<(block?3:6);++k) {
    double* q=block?p:c;double saved=q[k],h=1e-6;
    q[k]=saved+h;double plus[2];cost.Evaluate(blocks,plus,nullptr);
    q[k]=saved-h;double minus[2];cost.Evaluate(blocks,minus,nullptr);
    q[k]=saved;
    for(int row=0;row<2;++row) {
      double analytic=block?jpoint[row*3+k]:jcam[row*6+k];
      if(std::abs(analytic-(plus[row]-minus[row])/(2*h))>1e-5)return 3;
    }
  }
  ceres::AutoDiffCostFunction<MarkerResidual,2,6> marker_cost(new MarkerResidual{40,20,0,890,585});
  double marker_jac[12], marker_r[2];
  double const* marker_blocks[]={c};double* marker_jac_blocks[]={marker_jac};
  if(!marker_cost.Evaluate(marker_blocks,marker_r,marker_jac_blocks))return 5;
  for(int k=0;k<6;++k) {
    double saved=c[k],h=1e-6,plus[2],minus[2];
    c[k]=saved+h;marker_cost.Evaluate(marker_blocks,plus,nullptr);
    c[k]=saved-h;marker_cost.Evaluate(marker_blocks,minus,nullptr);c[k]=saved;
    for(int row=0;row<2;++row)
      if(std::abs(marker_jac[row*6+k]-(plus[row]-minus[row])/(2*h))>1e-5)return 6;
  }
  // Known synthetic marker case: perturb camera, recover reprojection fit.
  std::array<double,6> actual={.015,-.01,.008,8,-4,700};
  std::array<double,6> estimate={.025,-.02,.014,10,-7,703};
  std::vector<Marker> markers;
  for(auto xy:std::vector<std::array<double,2>>{{0,0},{40,0},{40,40},{0,40},{260,0},{300,40},{0,300},{300,300}}) {
    double pt[3]={xy[0],xy[1],0},q[3];ceres::AngleAxisRotatePoint(actual.data(),pt,q);
    for(int i=0;i<3;++i)q[i]+=actual[i+3];
    markers.push_back({0,xy[0],xy[1],0,1500*q[0]/q[2]+800,1500*q[1]/q[2]+600});
  }
  std::vector<std::array<double,6>> initial={estimate};
  double before=diagnostic(initial,{{0,0,0}},{{0,0,800,600}},markers).marker_rms;
  ceres::Problem problem;
  for(auto& m:markers)problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<MarkerResidual,2,6>(new MarkerResidual{m.x,m.y,m.z,m.u,m.v}),
      new ceres::HuberLoss(1),estimate.data());
  ceres::Solver::Options options;options.linear_solver_type=ceres::DENSE_QR;options.max_num_iterations=30;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  double after=diagnostic({estimate},{{0,0,0}},{{0,0,800,600}},markers).marker_rms;
  if(!summary.IsSolutionUsable() || after>1e-5 || after>=before)return 4;
  // Joint two-camera/two-point problem, anchored by fixed board markers.
  std::vector<std::array<double,6>> joint_true={actual,{.01,.005,-.004,-30,6,710}};
  std::vector<std::array<double,6>> joint=joint_true;
  joint[0][3]+=2;joint[1][4]-=2;
  std::vector<std::array<double,3>> point_true={{80,120,-40},{180,170,-80}};
  std::vector<std::array<double,3>> point_guess={{82,118,-38},{178,172,-82}};
  std::vector<Marker> joint_markers;
  std::vector<Observation> joint_tracks;
  for(int vi=0;vi<2;++vi) {
    for(auto m:markers) {
      double pt[3]={m.x,m.y,m.z},q[3];ceres::AngleAxisRotatePoint(joint_true[vi].data(),pt,q);
      for(int k=0;k<3;++k)q[k]+=joint_true[vi][k+3];
      joint_markers.push_back({vi,m.x,m.y,m.z,1500*q[0]/q[2]+800,1500*q[1]/q[2]+600});
    }
    for(int pi=0;pi<2;++pi) {
      double q[3];ceres::AngleAxisRotatePoint(joint_true[vi].data(),point_true[pi].data(),q);
      for(int k=0;k<3;++k)q[k]+=joint_true[vi][k+3];
      joint_tracks.push_back({vi,pi,1500*q[0]/q[2]+800,1500*q[1]/q[2]+600});
    }
  }
  double joint_before=diagnostic(joint,point_guess,joint_tracks,joint_markers).track_rms;
  ceres::Problem joint_problem;
  for(auto& m:joint_markers)joint_problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<MarkerResidual,2,6>(new MarkerResidual{m.x,m.y,m.z,m.u,m.v}),
      new ceres::HuberLoss(1),joint[m.camera].data());
  for(auto& o:joint_tracks)joint_problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<TrackResidual,2,6,3>(new TrackResidual{o.u,o.v}),
      new ceres::HuberLoss(1),joint[o.camera].data(),point_guess[o.point].data());
  ceres::Solver::Options joint_options; joint_options.max_num_iterations=30;
  joint_options.linear_solver_type=ceres::DENSE_SCHUR;
  ceres::Solver::Summary joint_summary;ceres::Solve(joint_options,&joint_problem,&joint_summary);
  auto joint_after=diagnostic(joint,point_guess,joint_tracks,joint_markers);
  if(!joint_summary.IsSolutionUsable() || joint_after.track_rms>1e-4 ||
     joint_after.track_rms>=joint_before || joint_after.marker_rms>1e-4)return 7;
  std::cout << "Both Jacobians and synthetic anchored camera+point fit passed; marker RMS "
            << before << " -> " << after << ", joint track RMS " << joint_before
            << " -> " << joint_after.track_rms << " px\n";
  return 0;
}

int main(int argc,char** argv) {
  if(argc==2 && std::string(argv[1])=="--self-test")return self_test();
  if(argc!=3)return 2;
  if(std::filesystem::exists(argv[2]))return 12;
  std::ifstream in(argv[1]);
  char tag;int n;
  if(!(in>>tag>>n) || tag!='C' || n!=3)return 3;
  std::vector<std::array<double,6>> cameras(n);
  for(auto& c:cameras) {
    double R[9]={};
    for(double& v:R)if(!(in>>v) || !std::isfinite(v))return 9;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
      double dot=0;for(int k=0;k<3;++k)dot+=R[3*i+k]*R[3*j+k];
      if(std::abs(dot-(i==j?1:0))>1e-3)return 9;
    }
    double determinant=R[0]*(R[4]*R[8]-R[5]*R[7])-R[1]*(R[3]*R[8]-R[5]*R[6])+
                       R[2]*(R[3]*R[7]-R[4]*R[6]);
    if(std::abs(determinant-1)>1e-3)return 9;
    double colmajor[9];for(int i=0;i<3;++i)for(int j=0;j<3;++j)colmajor[3*j+i]=R[3*i+j];
    ceres::RotationMatrixToAngleAxis(colmajor,c.data());
    for(int j=3;j<6;++j)if(!(in>>c[j]) || !std::isfinite(c[j]))return 9;
  }
  if(!(in>>tag>>n) || tag!='P' || n!=1189)return 4;
  std::vector<std::array<double,3>> points(n);
  for(auto& p:points)for(double& v:p)in>>v;
  if(!(in>>tag>>n) || tag!='O' || n!=2843)return 5;
  std::vector<Observation> observations(n);
  for(auto& o:observations)in>>o.camera>>o.point>>o.u>>o.v;
  if(!(in>>tag>>n) || tag!='M' || n!=48)return 6;
  std::vector<Marker> markers(n);
  for(auto& m:markers)in>>m.camera>>m.x>>m.y>>m.z>>m.u>>m.v;
  if(!in)return 7;
  auto finite=[](double v){return std::isfinite(v) && std::abs(v)<1e7;};
  for(auto& c:cameras)for(double v:c)if(!finite(v))return 9;
  for(auto& p:points)for(double v:p)if(!finite(v))return 9;
  for(auto& o:observations)if(o.camera<0 || o.camera>=3 || o.point<0 || o.point>=1189 ||
      !finite(o.u) || !finite(o.v))return 9;
  for(auto& m:markers)if(m.camera<0 || m.camera>=3 || !finite(m.x) || !finite(m.y) ||
      !finite(m.z) || !finite(m.u) || !finite(m.v))return 9;
  Diagnostics initial=diagnostic(cameras,points,observations,markers);
  if(initial.nonpositive || !std::isfinite(initial.track_rms) ||
     !std::isfinite(initial.marker_rms) || !std::isfinite(initial.min_z))return 13;
  ceres::Problem problem;
  for(auto& o:observations)problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<TrackResidual,2,6,3>(new TrackResidual{o.u,o.v}),
      new ceres::HuberLoss(1),cameras[o.camera].data(),points[o.point].data());
  for(auto& m:markers)problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<MarkerResidual,2,6>(new MarkerResidual{m.x,m.y,m.z,m.u,m.v}),
      new ceres::HuberLoss(1),cameras[m.camera].data());
  ceres::Solver::Options options;
  options.linear_solver_type=ceres::DENSE_SCHUR;
  options.max_num_iterations=30;
  options.num_threads=2;
  options.minimizer_progress_to_stdout=true;
  ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
  Diagnostics final=diagnostic(cameras,points,observations,markers);
  for(auto& c:cameras)for(double v:c)if(!finite(v))return 10;
  for(auto& p:points)for(double v:p)if(!finite(v))return 10;
  if(!std::isfinite(final.track_rms) || !std::isfinite(final.marker_rms) ||
     !std::isfinite(final.min_z) || !std::isfinite(summary.final_cost))return 10;
  std::ofstream out(argv[2]);out<<std::setprecision(17);
  if(!out)return 11;
  out<<"{\"initial_cost\":"<<summary.initial_cost<<",\"final_cost\":"<<summary.final_cost
     <<",\"iterations\":"<<summary.iterations.size()<<",\"termination\":\""
     <<ceres::TerminationTypeToString(summary.termination_type)<<"\",\"usable\":"
     <<(summary.IsSolutionUsable()?"true":"false")<<",\"initial\":{\"orb_rms_px\":"
     <<initial.track_rms<<",\"marker_rms_px\":"<<initial.marker_rms<<",\"min_depth_mm\":"
     <<initial.min_z<<",\"nonpositive_depths\":"<<initial.nonpositive<<"},\"final\":{\"orb_rms_px\":"
     <<final.track_rms<<",\"marker_rms_px\":"<<final.marker_rms<<",\"min_depth_mm\":"
     <<final.min_z<<",\"nonpositive_depths\":"<<final.nonpositive<<"},\"cameras\":[";
  for(size_t i=0;i<cameras.size();++i) {
    if(i)out<<',';double R[9];ceres::AngleAxisToRotationMatrix(cameras[i].data(),R);
    out<<"{\"rotation\":[";
    for(int row=0;row<3;++row)for(int col=0;col<3;++col) {
      if(row||col)out<<',';out<<R[3*col+row];
    }
    out<<"],\"translationMm\":["<<cameras[i][3]<<','<<cameras[i][4]<<','<<cameras[i][5]<<"]}";
  }
  out<<"],\"points\":[";
  for(size_t i=0;i<points.size();++i) {
    if(i)out<<',';out<<'['<<points[i][0]<<','<<points[i][1]<<','<<points[i][2]<<']';
  }
  out<<"]}\n";
  if(!out)return 11;
  if(!summary.IsSolutionUsable() || final.nonpositive || final.marker_rms>initial.marker_rms+1e-6)return 8;
  return 0;
}
