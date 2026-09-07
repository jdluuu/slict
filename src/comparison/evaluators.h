#pragma once

// Analytic SLICT equations from Point2PlaneFactorTMN and GyroAcceBiasFactorTMN.
// Keep the original factors as the independent reference used by backend A.
#include "slict/solver_comparison.h"
#include "basalt/spline/spline_common.h"
#include "basalt/utils/sophus_utils.hpp"
#include <splbatch/batch_binding.hpp>
#include <splbatch/evaluator_base.hpp>
#include <splbatch/inline_binding_input.hpp>
#include <array>

namespace slict::comparison {
struct SplineContext { double dt = 0.01; };
using Binding = splbatch::BatchBinding<SplineContext>;
using Matrix3 = Eigen::Matrix3d;

struct PreparedCoefficients {
  Eigen::Vector4d rotation, position, rotation_dt, position_ddt;
};
inline PreparedCoefficients PrepareCoefficients(double u, double dt) {
  static const Eigen::Matrix4d B = basalt::computeBlendingMatrix<double, false>(4);
  static const Eigen::Matrix4d BR = basalt::computeBlendingMatrix<double, true>(4);
  const Eigen::Vector4d U(1, u, u*u, u*u*u);
  const Eigen::Vector4d Ud(0, 1, 2*u, 3*u*u), Udd(0, 0, 2, 6*u);
  return {BR*U, B*U, (BR*Ud)/dt, (B*Udd)/(dt*dt)};
}
struct Runtime {
  Rotation first;
  std::array<Vector3, 4> positions, delta;
  std::array<Matrix3, 4> delta_jacobian;
  bool rotation_jacobians = false;
};
inline Runtime PrepareRuntime(double const* const* parameters,
                              const splbatch::EvaluationRequest& request) {
  Runtime runtime;
  runtime.first = Eigen::Map<const Rotation>(parameters[0]);
  for (int i=0; i<4; ++i) {
    runtime.positions[i] = Eigen::Map<const Vector3>(parameters[4+i]);
    runtime.rotation_jacobians |= request.JacobianRequested(i);
  }
  runtime.delta[0].setZero();
  for (int i=1; i<4; ++i) {
    const Eigen::Map<const Rotation> left(parameters[i-1]), right(parameters[i]);
    runtime.delta[i] = (left.inverse()*right).log();
    if (runtime.rotation_jacobians)
      Sophus::rightJacobianInvSO3(runtime.delta[i], runtime.delta_jacobian[i]);
  }
  return runtime;
}
struct SplineValue {
  Rotation rotation;
  std::array<Rotation, 4> exponential, post_inverse;
  std::array<Matrix3, 4> rotation_jacobian, exponential_jacobian;
};
inline SplineValue EvaluateSpline(const PreparedCoefficients& c, const Runtime& r) {
  SplineValue v;
  v.exponential[0] = r.first;
  for (int j=1; j<4; ++j) {
    const Vector3 argument = c.rotation[j] * r.delta[j];
    v.exponential[j] = Rotation::exp(argument);
    if (r.rotation_jacobians)
      Sophus::rightJacobianSO3(argument, v.exponential_jacobian[j]);
  }
  v.post_inverse[3] = Rotation();
  for (int j=3; j>=1; --j)
    v.post_inverse[j-1] = v.post_inverse[j] * v.exponential[j].inverse();
  v.rotation = (v.post_inverse[0] * r.first.inverse()).inverse();
  if (r.rotation_jacobians) {
    v.rotation_jacobian[0] = v.post_inverse[0].matrix();
    for (int j=1; j<4; ++j) {
      const Matrix3 helper = c.rotation[j] * v.post_inverse[j].matrix() * v.exponential_jacobian[j];
      v.rotation_jacobian[j] = helper * r.delta_jacobian[j];
      v.rotation_jacobian[j-1] -= helper * r.delta_jacobian[j].transpose();
    }
  }
  return v;
}
struct PreparedLidar { LidarObservation observation; PreparedCoefficients coefficients; };
class LidarEvaluator final : public splbatch::EvaluatorBase<
    LidarEvaluator, LidarObservation, Binding, PreparedLidar, Runtime, 1> {
 public:
  LidarEvaluator() {}
  static constexpr bool kOverwritesActiveJacobian = true;
  splbatch::JacobianStructure JacobianStructureImpl(const Binding&) const {
    splbatch::JacobianGroup group;
    group.residual_rows = {0, 1};
    for (int i=0; i<8; ++i) group.parameter_slices.push_back({static_cast<std::size_t>(i),0,3});
    return {{std::move(group)}};
  }
  PreparedObservation PrepareObservationImpl(const Binding& b, const Observation& o) const {
    return {o, PrepareCoefficients(o.u, b.context.dt)};
  }
  Runtime PrepareBatchImpl(const Binding&, double const* const* p,
                           const splbatch::EvaluationRequest& request) const {
    return PrepareRuntime(p, request);
  }
  template <typename Output>
  bool EvaluateObservationImpl(const Binding&, const PreparedObservation& prepared,
                               const Runtime& runtime, double const* const*, const Output& out) const {
    const auto& o = prepared.observation;
    const auto& c = prepared.coefficients;
    const auto v = EvaluateSpline(c, runtime);
    Vector3 position = Vector3::Zero();
    for (int j=0; j<4; ++j) position += c.position[j]*runtime.positions[j];
    out.Residual()[0] = o.weight * (o.normal.dot(v.rotation*o.point + position) + o.offset);
    const Eigen::RowVector3d derivative = -o.weight * o.normal.transpose()*v.rotation.matrix()*Rotation::hat(o.point);
    for (int j=0; j<4; ++j) {
      if (out.JacobianRequested(j)) out.Jacobian(j).template leftCols<3>() = derivative*v.rotation_jacobian[j];
      if (out.JacobianRequested(4+j)) out.Jacobian(4+j) = o.weight*c.position[j]*o.normal.transpose();
    }
    return true;
  }
};

struct PreparedImu { ImuObservation observation; PreparedCoefficients coefficients; };
class ImuEvaluator final : public splbatch::EvaluatorBase<
    ImuEvaluator, ImuObservation, Binding, PreparedImu, Runtime, 12> {
 public:
  ImuEvaluator(const Snapshot& snapshot)
      : gravity_(snapshot.gravity), bg_reference_(snapshot.gyro_reference),
        ba_reference_(snapshot.accel_reference), weights_(snapshot.imu_weights) {}
  static constexpr bool kOverwritesActiveJacobian = true;
  splbatch::JacobianStructure JacobianStructureImpl(const Binding&) const {
    splbatch::JacobianStructure s;
    s.groups.resize(4);
    for (int row=0; row<4; ++row) s.groups[row].residual_rows = {3*row,3};
    for (int j=0; j<4; ++j) {
      s.groups[0].parameter_slices.push_back({static_cast<std::size_t>(j),0,3});
      s.groups[1].parameter_slices.push_back({static_cast<std::size_t>(j),0,3});
      s.groups[1].parameter_slices.push_back({static_cast<std::size_t>(4+j),0,3});
    }
    s.groups[0].parameter_slices.push_back({8,0,3});
    s.groups[1].parameter_slices.push_back({9,0,3});
    s.groups[2].parameter_slices.push_back({8,0,3});
    s.groups[3].parameter_slices.push_back({9,0,3});
    return s;
  }
  PreparedObservation PrepareObservationImpl(const Binding& b, const Observation& o) const {
    return {o, PrepareCoefficients(o.u, b.context.dt)};
  }
  Runtime PrepareBatchImpl(const Binding&, double const* const* p,
                           const splbatch::EvaluationRequest& request) const {
    return PrepareRuntime(p, request);
  }
  template <typename Output>
  bool EvaluateObservationImpl(const Binding&, const PreparedObservation& prepared,
                               const Runtime& runtime, double const* const* p, const Output& out) const {
    const auto& o = prepared.observation;
    const auto& c = prepared.coefficients;
    const auto v = EvaluateSpline(c, runtime);
    std::array<Vector3,5> omega;
    omega[1].setZero();
    for (int j=1; j<4; ++j)
      omega[j+1] = v.exponential[j].inverse()*omega[j] + c.rotation_dt[j]*runtime.delta[j];
    Vector3 acceleration = gravity_;
    for (int j=0; j<4; ++j) acceleration += c.position_ddt[j]*runtime.positions[j];
    acceleration = v.rotation.inverse()*acceleration;
    const Eigen::Map<const Vector3> bg(p[8]), ba(p[9]);
    out.Residual().template segment<3>(0) = weights_[0]*(omega[4]+bg-o.gyro);
    out.Residual().template segment<3>(3) = weights_[1]*(acceleration+ba-o.accel);
    out.Residual().template segment<3>(6) = weights_[2]*(bg-bg_reference_);
    out.Residual().template segment<3>(9) = weights_[3]*(ba-ba_reference_);
    if (runtime.rotation_jacobians) {
      std::array<Matrix3,4> omega_delta, omega_rotation;
      for (int j=1; j<4; ++j)
        omega_delta[j] = v.post_inverse[j].matrix() *
          (c.rotation[j]*v.exponential[j].matrix().transpose()*Rotation::hat(omega[j])*
           v.exponential_jacobian[j].transpose() + c.rotation_dt[j]*Matrix3::Identity());
      omega_rotation[0].setZero();
      for (int j=1; j<4; ++j) {
        omega_rotation[j] = omega_delta[j]*runtime.delta_jacobian[j];
        omega_rotation[j-1] -= omega_delta[j]*runtime.delta_jacobian[j].transpose();
      }
      for (int j=0; j<4; ++j) if (out.JacobianRequested(j)) {
        out.Jacobian(j).template block<3,3>(0,0) = weights_[0]*omega_rotation[j];
        out.Jacobian(j).template block<3,3>(3,0) = weights_[1]*Rotation::hat(acceleration)*v.rotation_jacobian[j];
      }
    }
    for (int j=0; j<4; ++j) if (out.JacobianRequested(4+j))
      out.Jacobian(4+j).template block<3,3>(3,0) = weights_[1]*c.position_ddt[j]*v.rotation.inverse().matrix();
    if (out.JacobianRequested(8)) {
      out.Jacobian(8).template block<3,3>(0,0) = weights_[0]*Matrix3::Identity();
      out.Jacobian(8).template block<3,3>(6,0) = weights_[2]*Matrix3::Identity();
    }
    if (out.JacobianRequested(9)) {
      out.Jacobian(9).template block<3,3>(3,0) = weights_[1]*Matrix3::Identity();
      out.Jacobian(9).template block<3,3>(9,0) = weights_[3]*Matrix3::Identity();
    }
    return true;
  }
 private:
  Vector3 gravity_, bg_reference_, ba_reference_;
  Eigen::Vector4d weights_;
};

inline Binding Bind(State& state, int span, double dt, bool imu) {
  Binding b;
  b.context.dt = dt;
  for (int j=0; j<4; ++j) {
    b.parameters.blocks.push_back(state.rotations[span+j].data());
    b.parameters.sizes.push_back(4);
  }
  for (int j=0; j<4; ++j) {
    b.parameters.blocks.push_back(state.positions[span+j].data());
    b.parameters.sizes.push_back(3);
  }
  if (imu) {
    b.parameters.blocks.push_back(state.gyro_bias.data()); b.parameters.sizes.push_back(3);
    b.parameters.blocks.push_back(state.accel_bias.data()); b.parameters.sizes.push_back(3);
  }
  return b;
}

using BindingInput = splbatch::InlineBindingInput<SplineContext, 16>;
inline BindingInput BindInput(State& state, int span, double dt, bool imu) {
  BindingInput input;
  input.context.dt = dt;
  for (int j=0; j<4; ++j)
    input.AddParameter(state.rotations[span+j].data(), 4);
  for (int j=0; j<4; ++j)
    input.AddParameter(state.positions[span+j].data(), 3);
  if (imu) {
    input.AddParameter(state.gyro_bias.data(), 3);
    input.AddParameter(state.accel_bias.data(), 3);
  }
  return input;
}
}  // namespace slict::comparison

namespace splbatch {
// Both SLICT preparation functions consume only the observation and dt and
// return owning values. Support is checked by the channel before this test.
// Keep this opt-in local to these evaluators, not to arbitrary Context types.
template <> struct BindingPreparationReuse<slict::comparison::LidarEvaluator> {
  static bool Compatible(const slict::comparison::Binding& stored,
                         const slict::comparison::SplineContext& incoming) noexcept {
    return stored.context.dt == incoming.dt;
  }
};
template <> struct BindingPreparationReuse<slict::comparison::ImuEvaluator>
    : BindingPreparationReuse<slict::comparison::LidarEvaluator> {};
}  // namespace splbatch
