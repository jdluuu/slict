#include "internal.h"
#include "basalt/utils/sophus_utils.hpp"
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace slict::comparison {
Backend ParseBackend(const std::string& name) {
  if (name=="native" || name=="A") return Backend::Native;
  if (name=="ceres_scalar" || name=="B") return Backend::CeresScalar;
  if (name=="ceres_batch" || name=="C") return Backend::CeresBatch;
  if (name=="native_batch" || name=="D") return Backend::NativeBatch;
  throw std::invalid_argument("Unknown solver backend: "+name);
}
const char* BackendName(Backend b) {
  switch (b) {case Backend::Native:return "native";case Backend::CeresScalar:return "ceres_scalar";case Backend::CeresBatch:return "ceres_batch";case Backend::NativeBatch:return "native_batch";}
  throw std::invalid_argument("Invalid backend enum");
}
bool UsesCeresSolver(Backend backend) {
  switch (backend) {
    case Backend::Native: case Backend::NativeBatch: return false;
    case Backend::CeresScalar: case Backend::CeresBatch: return true;
  }
  throw std::invalid_argument("Invalid backend enum");
}
namespace {
void Require(bool condition,const char* message) {if (!condition) throw std::invalid_argument(message);}
void CheckState(const State& state) {
  Require(state.rotations.size()==state.positions.size(),"Rotation/position count mismatch");
  Require(state.gyro_bias.allFinite() && state.accel_bias.allFinite(),"Nonfinite bias");
  for (std::size_t i=0;i<state.rotations.size();++i) {
    Require(state.positions[i].allFinite(),"Nonfinite position");
    const auto q=state.rotations[i].unit_quaternion();
    Require(q.coeffs().allFinite() && std::abs(q.norm()-1)<1e-8,"Invalid quaternion");
  }
}
Eigen::MatrixXd PseudoInverse(const Eigen::MatrixXd& matrix) {
  if (matrix.rows()==0) return matrix;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(0.5*(matrix+matrix.transpose()));
  if (eig.info()!=Eigen::Success) throw std::runtime_error("Marginalization eigensolver failed");
  const double cutoff=1e-12*std::max(1.0,eig.eigenvalues().cwiseAbs().maxCoeff());
  Eigen::VectorXd inverse=eig.eigenvalues().unaryExpr([&](double v){return v>cutoff?1.0/v:0.0;});
  return eig.eigenvectors()*inverse.asDiagonal()*eig.eigenvectors().transpose();
}
}
void Snapshot::Canonicalize() {
  // Common input order makes residual rows comparable; no backend-specific sampling.
  auto by_span=[](const auto& a,const auto& b){return a.span<b.span;};
  std::stable_sort(imu.begin(),imu.end(),by_span);std::stable_sort(lidar.begin(),lidar.end(),by_span);
  std::sort(fixed_knots.begin(),fixed_knots.end());
  fixed_knots.erase(std::unique(fixed_knots.begin(),fixed_knots.end()),fixed_knots.end());
}
void Snapshot::Validate() const {
  CheckState(initial);
  const int knots=initial.rotations.size();
  Require(knots>=4 && knots<=512,"Snapshot requires 4..512 cubic control points");
  Require(std::isfinite(dt) && dt>0 && std::isfinite(start_time),"Invalid spline time");
  Require(std::isfinite(native_damping) && native_damping>0,"Invalid native damping");
  Require(imu.size()<=1000000 && lidar.size()<=1000000,"Too many observations");
  Require(gravity.allFinite() && gyro_reference.allFinite() && accel_reference.allFinite(),"Nonfinite model reference");
  Require(imu_weights.allFinite() && (imu_weights.array()>0).all(),"IMU weights must be positive");
  auto check_binding=[&](const auto& o) {
    Require(o.span>=0 && o.span+4<=knots,"Observation support outside control storage");
    Require(std::isfinite(o.u) && o.u>=-1e-9 && o.u<=1+1e-9 && std::isfinite(o.timestamp),"Invalid observation time");
  };
  for (const auto& o:imu) {check_binding(o);Require(o.gyro.allFinite() && o.accel.allFinite(),"Nonfinite IMU measurement");}
  for (const auto& o:lidar) {
    check_binding(o);Require(o.point.allFinite() && o.normal.allFinite() && o.normal.norm()>0,"Invalid LiDAR geometry");
    Require(std::isfinite(o.offset) && std::isfinite(o.weight) && o.weight>0,"Invalid LiDAR weight");
  }
  Require(std::set<int>(fixed_knots.begin(),fixed_knots.end()).size()==fixed_knots.size(),"Duplicate fixed knot");
  for (int i:fixed_knots) Require(i>=0 && i<knots,"Invalid fixed knot");
  if (!prior.Empty()) {
    CheckState(prior.reference);
    Require(prior.reference.rotations.size()==prior.knots.size(),"Prior reference count mismatch");
    Require(prior.sqrt_information.rows()==prior.offset.size() && prior.sqrt_information.cols()==6*static_cast<int>(prior.knots.size())+6,"Invalid prior dimensions");
    Require(prior.sqrt_information.allFinite() && prior.offset.allFinite(),"Nonfinite prior");
    Require(std::set<int>(prior.knots.begin(),prior.knots.end()).size()==prior.knots.size(),"Duplicate prior knot");
    for (int i:prior.knots) Require(i>=0 && i<knots,"Prior knot outside current window");
  }
}
void ApplyIncrement(State& state,const Eigen::VectorXd& delta) {
  Require(delta.size()==state.Dimension(),"Increment dimension mismatch");
  for (std::size_t i=0;i<state.rotations.size();++i) {
    state.rotations[i]*=Rotation::exp(delta.segment<3>(6*i));
    state.positions[i]+=delta.segment<3>(6*i+3);
  }
  state.gyro_bias+=delta.tail<6>().head<3>();state.accel_bias+=delta.tail<3>();
}
double StateDistance(const State& a,const State& b) {
  Require(a.rotations.size()==b.rotations.size(),"State count mismatch");
  double max_error=std::max((a.gyro_bias-b.gyro_bias).norm(),(a.accel_bias-b.accel_bias).norm());
  for (std::size_t i=0;i<a.rotations.size();++i) {
    max_error=std::max(max_error,(a.rotations[i].inverse()*b.rotations[i]).log().norm());
    max_error=std::max(max_error,(a.positions[i]-b.positions[i]).norm());
  }
  return max_error;
}
Linearization LinearizePrior(const Prior& prior,const State& state) {
  Linearization out;
  if (prior.Empty()) {out.jacobian.resize(0,state.Dimension());return out;}
  const int count=prior.knots.size(),dim=6*count+6;
  Eigen::VectorXd delta(dim);
  out.jacobian=Eigen::MatrixXd::Zero(prior.offset.size(),state.Dimension());
  for (int i=0;i<count;++i) {
    const int knot=prior.knots[i];
    const Vector3 dr=(prior.reference.rotations[i].inverse()*state.rotations[knot]).log();
    delta.segment<3>(6*i)=dr;
    delta.segment<3>(6*i+3)=state.positions[knot]-prior.reference.positions[i];
    Eigen::Matrix3d J;Sophus::rightJacobianInvSO3(dr,J);
    out.jacobian.middleCols(6*knot,3)=prior.sqrt_information.middleCols(6*i,3)*J;
    out.jacobian.middleCols(6*knot+3,3)=prior.sqrt_information.middleCols(6*i+3,3);
  }
  delta.tail<6>().head<3>()=state.gyro_bias-prior.reference.gyro_bias;
  delta.tail<3>()=state.accel_bias-prior.reference.accel_bias;
  out.jacobian.rightCols(6)=prior.sqrt_information.rightCols(6);
  out.residual=prior.offset+prior.sqrt_information*delta;
  return out;
}

Result Solve(const Snapshot& snapshot,const Options& options) {
  snapshot.Validate();
  Require(options.threads>0 && options.threads<=256,"threads must be in 1..256");
  Require(options.iterations>0 && options.iterations<=10000,"iterations must be in 1..10000");
  Require(std::isfinite(options.step_limit) && options.step_limit>=0,"Invalid step limit");
  Require(std::isfinite(options.ceres_initial_trust_region_radius) && options.ceres_initial_trust_region_radius>0,
          "Invalid Ceres initial trust region radius");
  auto result=UsesCeresSolver(options.backend)?SolveCeres(snapshot,options):SolveNative(snapshot,options);
  // Quality checks are explicit and outside total_ms for every backend.
  const auto diagnostics=Clock::now();
  const auto residual=EvaluateResiduals(snapshot,result.state);
  result.metrics.final_cost=0.5*residual.squaredNorm();
  result.metrics.usable=result.metrics.usable && residual.allFinite();
  if (!result.metrics.usable) result.metrics.termination+="_unusable";
  result.metrics.diagnostic_ms=Milliseconds(diagnostics);
  return result;
}

Prior Marginalize(const Snapshot& source,const State& solved,int first_kept,int threads) {
  Require(first_kept>0 && first_kept<static_cast<int>(solved.rotations.size()),"Invalid marginalization cutoff");
  Snapshot subset=source;subset.initial=solved;
  subset.imu.erase(std::remove_if(subset.imu.begin(),subset.imu.end(),[&](const auto& o){return o.span>=first_kept;}),subset.imu.end());
  subset.lidar.erase(std::remove_if(subset.lidar.begin(),subset.lidar.end(),[&](const auto& o){return o.span>=first_kept;}),subset.lidar.end());
  std::set<int> kept;
  for (const auto& o:subset.imu) for (int j=o.span;j<o.span+4;++j) if (j>=first_kept) kept.insert(j);
  for (const auto& o:subset.lidar) for (int j=o.span;j<o.span+4;++j) if (j>=first_kept) kept.insert(j);
  for (int j:source.prior.knots) if (j>=first_kept) kept.insert(j);
  const auto linear=LinearizeNative(subset,solved,threads);
  if (linear.residual.size()==0) return {};
  const Eigen::MatrixXd Jm=linear.jacobian.leftCols(6*first_kept);
  Eigen::MatrixXd Jk(linear.residual.size(),6*kept.size()+6);
  Prior prior;int i=0;
  for (int knot:kept) {
    Jk.middleCols(6*i,6)=linear.jacobian.middleCols(6*knot,6);
    prior.knots.push_back(knot-first_kept);
    prior.reference.rotations.push_back(solved.rotations[knot]);
    prior.reference.positions.push_back(solved.positions[knot]);++i;
  }
  Jk.rightCols(6)=linear.jacobian.rightCols(6);
  const Eigen::MatrixXd Hmm=Jm.transpose()*Jm,Hkm=Jk.transpose()*Jm;
  const Eigen::MatrixXd multiplier=Hkm*PseudoInverse(Hmm);
  Eigen::MatrixXd H=Jk.transpose()*Jk-multiplier*Hkm.transpose();
  H=(0.5*(H+H.transpose())).eval();
  const Eigen::VectorXd g=Jk.transpose()*linear.residual-multiplier*(Jm.transpose()*linear.residual);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H);
  if (eig.info()!=Eigen::Success || !g.allFinite()) throw std::runtime_error("Marginalization produced a nonfinite prior");
  const double cutoff=1e-12*std::max(1.0,eig.eigenvalues().cwiseAbs().maxCoeff());
  Eigen::VectorXd sqrt=eig.eigenvalues().unaryExpr([&](double v){return v>cutoff?std::sqrt(v):0.;});
  const Eigen::VectorXd inverse=sqrt.unaryExpr([](double v){return v>0?1./v:0.;});
  prior.sqrt_information=sqrt.asDiagonal()*eig.eigenvectors().transpose();
  prior.offset=inverse.asDiagonal()*eig.eigenvectors().transpose()*g;
  prior.reference.gyro_bias=solved.gyro_bias;prior.reference.accel_bias=solved.accel_bias;
  return prior;
}

Snapshot MakeSyntheticSnapshot(int knots,int lidar_count,int imu_count,std::uint64_t seed) {
  Require(knots>=4 && knots<=512 && lidar_count>=0 && imu_count>=0,"Invalid synthetic dimensions");
  Snapshot s;s.dt=0.02;s.imu_weights<<5,1,5,10;s.fixed_knots={0};s.frame=seed;
  std::mt19937_64 rng(seed);std::normal_distribution<double> normal(0,1);
  auto random_vector=[&](){return Vector3(normal(rng),normal(rng),normal(rng));};
  for (int i=0;i<knots;++i) {
    const double t=i*s.dt;
    s.initial.rotations.push_back(Rotation::exp(Vector3(0.08*std::sin(t),0.03*std::cos(t),0.15*t)));
    s.initial.positions.emplace_back(t,0.4*std::sin(t),0.2*std::cos(2*t));
  }
  s.initial.gyro_bias=Vector3(0.01,-0.02,0.005);s.initial.accel_bias=Vector3(0.02,0.01,-0.03);
  s.gyro_reference=s.initial.gyro_bias;s.accel_reference=s.initial.accel_bias;
  for (int i=0;i<lidar_count;++i) {
    LidarObservation o;o.span=i%(knots-3);o.u=(i+0.5)/(lidar_count+1.0);o.timestamp=(o.span+o.u)*s.dt;
    o.point=5*random_vector();o.normal=random_vector().normalized();o.weight=10;
    o.offset=-NativeLidarFactor(s,s.initial,o,false).residual[0]/o.weight;
    s.lidar.push_back(o);
  }
  for (int i=0;i<imu_count;++i) {
    ImuObservation o;o.span=i%(knots-3);o.u=(i+0.5)/(imu_count+1.0);o.timestamp=(o.span+o.u)*s.dt;
    const auto r=NativeImuFactor(s,s.initial,o,false).residual;
    o.gyro=r.head<3>()/s.imu_weights[0];o.accel=r.segment<3>(3)/s.imu_weights[1];s.imu.push_back(o);
  }
  for (int i=1;i<knots;++i) {
    s.initial.rotations[i]*=Rotation::exp(0.003*random_vector());
    s.initial.positions[i]+=0.01*random_vector();
  }
  s.initial.gyro_bias+=0.002*random_vector();s.initial.accel_bias+=0.003*random_vector();
  s.Canonicalize();s.Validate();return s;
}
}  // namespace slict::comparison
