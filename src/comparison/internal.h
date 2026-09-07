#pragma once
#include "slict/solver_comparison.h"
#include <chrono>
#include <memory>
#include <ceres/ceres.h>

namespace slict::comparison {
using Clock = std::chrono::steady_clock;
inline double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct CeresProblem {
  // State storage must outlive the problem. Ceres owns all registered costs.
  std::unique_ptr<ceres::Problem> problem;
  std::vector<double*> parameters;
  std::vector<int> full_column_indices;
};
CeresProblem BuildCeresProblem(const Snapshot&, State&, Backend);
Result SolveNative(const Snapshot&, const Options&);
// Owns batch costs/buffers; the snapshot and state storage must outlive it.
class NativeBatchProblem {
 public:
  virtual ~NativeBatchProblem() = default;
  virtual void Evaluate(int threads, Eigen::VectorXd&, Eigen::MatrixXd*) = 0;
  virtual int ResidualBlocks() const = 0;
};
std::unique_ptr<NativeBatchProblem> BuildNativeBatchProblem(const Snapshot&, State&);
Result SolveCeres(const Snapshot&, const Options&);
void EvaluateNativeInto(const Snapshot&, const State&, int threads,
                        Eigen::VectorXd&, Eigen::MatrixXd*);
Eigen::VectorXd EvaluateResiduals(const Snapshot&, const State&);
}  // namespace slict::comparison
