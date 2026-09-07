#include "internal.h"
#include "factor/Point2PlaneFactorTMN.hpp"
#include "factor/GyroAcceBiasFactorTMN.hpp"
#include <Eigen/SparseLU>
#include <algorithm>

namespace slict::comparison {
namespace {
void CopySupport(const State& state, int span, std::vector<Rotation>& r, std::vector<Vector3>& p) {
  // Same per-observation control copies as tmnSolver::Evaluate*Factors.
  for (int j=0; j<4; ++j) r.push_back(state.rotations[span+j]);
  for (int j=0; j<4; ++j) p.push_back(state.positions[span+j]);
}
Point2PlaneFactorTMN MakeLidar(const Snapshot& s, const LidarObservation& o) {
  Eigen::Vector4d plane; plane << o.normal, o.offset;
  return Point2PlaneFactorTMN(Vector3::Zero(), o.point, plane, o.weight, 4, s.dt, o.u);
}
GyroAcceBiasFactorTMN MakeImu(const Snapshot& s, const ImuObservation& o) {
  return GyroAcceBiasFactorTMN(ImuSample(o.timestamp, o.gyro, o.accel),
      ImuBias(s.gyro_reference, s.accel_reference), s.gravity,
      s.imu_weights[0], s.imu_weights[1], s.imu_weights[2], s.imu_weights[3], 4, s.dt, o.u);
}
}

Linearization NativeLidarFactor(const Snapshot& s, const State& state, const LidarObservation& o, bool jacobian) {
  std::vector<Rotation> rotations; std::vector<Vector3> positions;
  CopySupport(state, o.span, rotations, positions);
  auto factor = MakeLidar(s,o);
  factor.Evaluate(rotations, positions, jacobian);
  Linearization result;
  result.residual = Eigen::VectorXd::Constant(1, factor.residual);
  if (jacobian) result.jacobian = factor.jacobian;
  return result;
}
Linearization NativeImuFactor(const Snapshot& s, const State& state, const ImuObservation& o, bool jacobian) {
  std::vector<Rotation> rotations; std::vector<Vector3> positions;
  CopySupport(state, o.span, rotations, positions);
  Vector3 bg=state.gyro_bias, ba=state.accel_bias;
  auto factor = MakeImu(s,o);
  factor.Evaluate(rotations, positions, bg, ba, jacobian);
  Linearization result; result.residual=factor.residual;
  if (jacobian) result.jacobian=factor.jacobian;
  return result;
}

void EvaluateNativeInto(const Snapshot& s, const State& state, int threads,
                        Eigen::VectorXd& residual, Eigen::MatrixXd* jacobian) {
  const int imu_rows=12*s.imu.size(), measurement_rows=imu_rows+s.lidar.size();
  const int rows=measurement_rows+s.prior.offset.size(), columns=state.Dimension();
  residual.resize(rows);
  if (jacobian) jacobian->resize(rows, columns);
  #pragma omp parallel for num_threads(threads)
  for (std::size_t i=0; i<s.imu.size(); ++i) {
    const auto& o=s.imu[i];
    std::vector<Rotation> rotations; std::vector<Vector3> positions;
    CopySupport(state,o.span,rotations,positions);
    Vector3 bg=state.gyro_bias, ba=state.accel_bias;
    auto factor=MakeImu(s,o);
    factor.Evaluate(rotations,positions,bg,ba,jacobian!=nullptr);
    residual.segment<12>(12*i)=factor.residual;
    if (jacobian) {
      jacobian->middleRows<12>(12*i).setZero();
      jacobian->block(12*i,6*o.span,12,24)=factor.jacobian.leftCols(24);
      jacobian->block(12*i,columns-6,12,6)=factor.jacobian.rightCols(6);
    }
  }
  #pragma omp parallel for num_threads(threads)
  for (std::size_t i=0; i<s.lidar.size(); ++i) {
    const auto& o=s.lidar[i];
    std::vector<Rotation> rotations; std::vector<Vector3> positions;
    CopySupport(state,o.span,rotations,positions);
    auto factor=MakeLidar(s,o);
    factor.Evaluate(rotations,positions,jacobian!=nullptr);
    residual[imu_rows+i]=factor.residual;
    if (jacobian) {
      jacobian->row(imu_rows+i).setZero();
      jacobian->block(imu_rows+i,6*o.span,1,24)=factor.jacobian;
    }
  }
  if (!s.prior.Empty()) {
    const auto prior=LinearizePrior(s.prior,state);
    residual.tail(prior.residual.size())=prior.residual;
    if (jacobian) jacobian->bottomRows(prior.residual.size())=prior.jacobian;
  }
  if (jacobian) for (int knot:s.fixed_knots) jacobian->middleCols(6*knot,6).setZero();
}
Linearization LinearizeNative(const Snapshot& s, const State& state, int threads) {
  Linearization result;
  EvaluateNativeInto(s,state,threads,result.residual,&result.jacobian);
  return result;
}
Eigen::VectorXd EvaluateResiduals(const Snapshot& s,const State& state) {
  Eigen::VectorXd residual; EvaluateNativeInto(s,state,1,residual,nullptr); return residual;
}
Result SolveNative(const Snapshot& s,const Options& options) {
  const auto total=Clock::now();
  Result result;
  auto phase=Clock::now(); result.state=s.initial; result.metrics.reset_ms=Milliseconds(phase);
  auto& m=result.metrics;
  phase=Clock::now();
  Eigen::VectorXd residual(12*s.imu.size()+s.lidar.size()+s.prior.offset.size());
  Eigen::MatrixXd jacobian(residual.size(),s.initial.Dimension());
  std::unique_ptr<NativeBatchProblem> batch;
  if (options.backend==Backend::NativeBatch) batch=BuildNativeBatchProblem(s,result.state);
  m.build_ms=Milliseconds(phase);
  m.residual_blocks=batch ? batch->ResidualBlocks() : s.imu.size()+s.lidar.size()+(!s.prior.Empty());
  m.scalar_residuals=residual.size();
  const auto solving=Clock::now();
  m.usable=true; m.termination="iteration_limit";
  for (int iteration=0;iteration<options.iterations;++iteration) {
    phase=Clock::now();
    if (batch) batch->Evaluate(options.threads,residual,&jacobian);
    else EvaluateNativeInto(s,result.state,options.threads,residual,&jacobian);
    m.evaluate_ms+=Milliseconds(phase);
    if (iteration==0) m.initial_cost=0.5*residual.squaredNorm();
    if (!residual.allFinite() || !jacobian.allFinite()) { m.usable=false; m.termination="nonfinite_linearization"; break; }
    phase=Clock::now();
    Eigen::SparseMatrix<double> sparse=jacobian.sparseView(); sparse.makeCompressed();
    const Eigen::SparseMatrix<double> transpose=sparse.transpose();
    Eigen::SparseMatrix<double> H=transpose*sparse;
    const Eigen::VectorXd rhs=-transpose*residual;
    Eigen::SparseMatrix<double> identity(H.rows(),H.cols()); identity.setIdentity();
    H += (s.native_damping/std::pow(2.0,iteration))*identity;
    m.assemble_ms+=Milliseconds(phase);
    phase=Clock::now();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(H); solver.factorize(H);
    if (solver.info()!=Eigen::Success) { m.linear_ms+=Milliseconds(phase);m.usable=false;m.termination="factorization_failure";break; }
    Eigen::VectorXd step=solver.solve(rhs);
    const bool valid=solver.info()==Eigen::Success && step.allFinite();
    m.linear_ms+=Milliseconds(phase);
    if (!valid) { m.usable=false;m.termination="linear_solve_failure";break; }
    phase=Clock::now();
    if (options.step_limit>0 && step.norm()>options.step_limit) step*=options.step_limit/step.norm();
    ApplyIncrement(result.state,step);
    m.update_ms+=Milliseconds(phase);
    ++m.iterations; ++m.successful_steps;
    if (step.norm()<=options.parameter_tolerance) {m.termination="parameter_tolerance";break;}
  }
  m.solve_ms=Milliseconds(solving);
  phase=Clock::now(); batch.reset();jacobian.resize(0,0);residual.resize(0);m.destroy_ms=Milliseconds(phase);
  m.total_ms=Milliseconds(total);
  return result;
}
}  // namespace slict::comparison
