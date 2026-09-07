#pragma once

#include <Eigen/Core>
#include <sophus/so3.hpp>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace slict::comparison {
using Vector3 = Eigen::Vector3d;
using Rotation = Sophus::SO3d;

enum class Backend { Native, CeresScalar, CeresBatch, NativeBatch };
Backend ParseBackend(const std::string& name);
const char* BackendName(Backend backend);
bool UsesCeresSolver(Backend backend);

struct State {
  std::vector<Rotation> rotations;
  std::vector<Vector3> positions;
  Vector3 gyro_bias = Vector3::Zero();
  Vector3 accel_bias = Vector3::Zero();
  int Dimension() const { return 6 * static_cast<int>(rotations.size()) + 6; }
};

struct LidarObservation {
  int span = 0;
  double u = 0, timestamp = 0;
  Vector3 point = Vector3::Zero();
  Vector3 normal = Vector3::UnitZ();
  double offset = 0, weight = 1;
};
struct ImuObservation {
  int span = 0;
  double u = 0, timestamp = 0;
  Vector3 gyro = Vector3::Zero(), accel = Vector3::Zero();
};

// r_prior = offset + sqrt_information * [Log(Rref^-1 R), p-pref, bg-bgref, ba-baref].
// Knot indices address the current State; reference knots have the same order.
struct Prior {
  std::vector<int> knots;
  State reference;
  Eigen::MatrixXd sqrt_information;
  Eigen::VectorXd offset;
  bool Empty() const { return offset.size() == 0; }
};

struct Snapshot {
  std::uint64_t frame = 0;
  int outer_iteration = 0;
  double start_time = 0, dt = 0.01, native_damping = 1;
  State initial;
  std::vector<ImuObservation> imu;
  std::vector<LidarObservation> lidar;
  Vector3 gravity = Vector3(0, 0, 9.82);
  Vector3 gyro_reference = Vector3::Zero(), accel_reference = Vector3::Zero();
  Eigen::Vector4d imu_weights = Eigen::Vector4d::Ones();
  std::vector<int> fixed_knots;
  Prior prior;
  void Validate() const;
  void Canonicalize();
};

struct Options {
  Backend backend = Backend::Native;
  int threads = 1, iterations = 1;
  double step_limit = 0.5;
  double function_tolerance = 1e-8, gradient_tolerance = 1e-10;
  double parameter_tolerance = 1e-10;
  double ceres_initial_trust_region_radius = 1e4;
};

struct Metrics {
  double reset_ms = 0, build_ms = 0, evaluate_ms = 0, assemble_ms = 0;
  double linear_ms = 0, update_ms = 0, solve_ms = 0, destroy_ms = 0, total_ms = 0;
  double diagnostic_ms = 0, initial_cost = 0, final_cost = 0;
  double residual_evaluation_ms = 0, jacobian_evaluation_ms = 0;
  double ceres_final_trust_region_radius = 0;
  int iterations = 0, successful_steps = 0, unsuccessful_steps = 0;
  int residual_blocks = 0, scalar_residuals = 0;
  bool usable = false;
  std::string termination;
};
struct Result { State state; Metrics metrics; };
struct Linearization { Eigen::VectorXd residual; Eigen::MatrixXd jacobian; };

Result Solve(const Snapshot& snapshot, const Options& options);
Linearization LinearizeNative(const Snapshot& snapshot, const State& state, int threads = 1);
Linearization LinearizeNativeBatch(const Snapshot& snapshot, const State& state, int threads = 1);
Linearization LinearizeCeres(const Snapshot& snapshot, const State& state, Backend backend, int threads = 1);
// Independent original SLICT factor implementation, with columns [R0,p0,...,bg,ba].
Linearization NativeLidarFactor(const Snapshot&, const State&, const LidarObservation&, bool jacobian = true);
Linearization NativeImuFactor(const Snapshot&, const State&, const ImuObservation&, bool jacobian = true);
Linearization LinearizePrior(const Prior&, const State&);
void ApplyIncrement(State&, const Eigen::VectorXd&);
double StateDistance(const State&, const State&);

// Only outgoing measurements plus the existing prior are marginalized. All
// backends use this same routine, outside the measured optimization interval.
Prior Marginalize(const Snapshot&, const State& solved, int first_kept_knot, int threads);

void SaveSnapshot(const Snapshot&, const std::string& path);
Snapshot LoadSnapshot(const std::string& path);
std::string SnapshotDigest(const Snapshot&);
Snapshot MakeSyntheticSnapshot(int knots, int lidar_count, int imu_count, std::uint64_t seed);
void WriteCsvHeader(std::ostream&);
void WriteCsvRow(std::ostream&, const Snapshot&, const Options&, const Metrics&, int repeat,
                 const std::string& digest, double marginalize_ms = 0, double input_ms = 0);
}  // namespace slict::comparison
