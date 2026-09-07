#include "internal.h"
#include "evaluators.h"
#include <splbatch/problem_batcher.hpp>
#include <exception>
#include <set>
#include <unordered_map>

namespace slict::comparison {
namespace {
// Consumes the same SegmentBatchCostFunction objects as Ceres, without creating
// a ceres::Problem or invoking its solver. Only registration and scatter differ.
class NativeRegistry final : public NativeBatchProblem, public splbatch::ResidualRegistry {
 public:
  NativeRegistry(const Snapshot& snapshot, State& state) : snapshot_(snapshot), state_(state) {
    const std::set<int> fixed(snapshot.fixed_knots.begin(), snapshot.fixed_knots.end());
    for (std::size_t i=0; i<state.rotations.size(); ++i) {
      const bool constant=fixed.count(i);
      parameters_.emplace(state.rotations[i].data(), Parameter{4, constant ? -1 : int(6*i)});
      parameters_.emplace(state.positions[i].data(), Parameter{3, constant ? -1 : int(6*i+3)});
    }
    parameters_.emplace(state.gyro_bias.data(), Parameter{3, state.Dimension()-6});
    parameters_.emplace(state.accel_bias.data(), Parameter{3, state.Dimension()-3});
    splbatch::ProblemBatcher batcher(*this);
    auto& imu=batcher.RegisterEvaluator(ImuEvaluator(snapshot));
    auto& lidar=batcher.RegisterEvaluator(LidarEvaluator());
    for (const auto& o:snapshot.imu) {
      SPLBATCH_DIAG_OBSERVATION(selection);
      SPLBATCH_DIAG_SCOPE(binding_lifetime, BindingCleanup, true);
      imu.AddObservation(SPLBATCH_DIAG_MEASURE(Binding, true, BindInput(state,o.span,snapshot.dt,true)),o);
    }
    for (const auto& o:snapshot.lidar) {
      SPLBATCH_DIAG_OBSERVATION(selection);
      SPLBATCH_DIAG_SCOPE(binding_lifetime, BindingCleanup, true);
      lidar.AddObservation(SPLBATCH_DIAG_MEASURE(Binding, true, BindInput(state,o.span,snapshot.dt,false)),o);
    }
    batcher.Commit();
    if (measurement_rows_ != int(12*snapshot.imu.size()+snapshot.lidar.size()))
      throw std::runtime_error("Native batch registration changed residual row count");
  }

  void AddResidualBlock(std::unique_ptr<ceres::CostFunction> cost,
                        std::vector<double*> pointers) override {
    SPLBATCH_DIAG_SCOPE(registration, NativeRegistration, false);
    if (!cost || cost->num_residuals() <= 0 || pointers.size()!=cost->parameter_block_sizes().size())
      throw std::invalid_argument("Invalid native batch registration");
    Entry entry;
    entry.row=measurement_rows_;
    entry.cost=std::move(cost);
    const auto& sizes=entry.cost->parameter_block_sizes();
    entry.buffers.resize(pointers.size());
    for (std::size_t b=0; b<pointers.size(); ++b) {
      const auto found=parameters_.find(pointers[b]);
      if (found==parameters_.end() || found->second.ambient_size!=sizes[b])
        throw std::invalid_argument("Native batch parameter is unregistered or has wrong size");
      entry.pointers.push_back(pointers[b]);
      const int column=found->second.column;
      entry.columns.push_back(column);
      // Batch outputs are contiguous row-major blocks. A's global J is
      // column-major: use persistent scratch blocks, then explicitly scatter.
      if (column>=0) {
        SPLBATCH_DIAG_SCOPE(allocation, JacobianAllocation, false);
        entry.buffers[b].resize(entry.cost->num_residuals(),sizes[b]);
      }
      entry.jacobians.push_back(column>=0 ? entry.buffers[b].data() : nullptr);
    }
    measurement_rows_+=entry.cost->num_residuals();
    entries_.push_back(std::move(entry));
  }

  void Evaluate(int threads, Eigen::VectorXd& residual, Eigen::MatrixXd* jacobian) override {
    const int rows=measurement_rows_+snapshot_.prior.offset.size();
    residual.resize(rows);
    if (jacobian) jacobian->resize(rows,state_.Dimension());
    #pragma omp parallel for num_threads(threads)
    for (std::size_t i=0; i<entries_.size(); ++i) {
      auto& entry=entries_[i];
      entry.failure=nullptr;
      try {
        const int count=entry.cost->num_residuals();
        if (!entry.cost->Evaluate(entry.pointers.data(),residual.data()+entry.row,
                                  jacobian ? entry.jacobians.data() : nullptr))
          throw std::runtime_error("Native batch Evaluate failed");
        if (jacobian) {
          jacobian->middleRows(entry.row,count).setZero();
          for (std::size_t b=0; b<entry.columns.size(); ++b) if (entry.columns[b]>=0)
            // SLICT's LieAnalyticLocalParameterization expects [J_SO3, 0]
            // for 4D rotation storage. The first 3 columns are already tangent
            // derivatives; applying a quaternion Plus Jacobian again is wrong.
            jacobian->block(entry.row,entry.columns[b],count,3)=entry.buffers[b].leftCols(3);
        }
      } catch (...) { entry.failure=std::current_exception(); }
    }
    for (const auto& entry:entries_) if (entry.failure) std::rethrow_exception(entry.failure);
    if (!snapshot_.prior.Empty()) {
      const auto prior=LinearizePrior(snapshot_.prior,state_);
      residual.tail(prior.residual.size())=prior.residual;
      if (jacobian) {
        jacobian->bottomRows(prior.residual.size())=prior.jacobian;
        for (int knot:snapshot_.fixed_knots)
          jacobian->block(measurement_rows_,6*knot,prior.residual.size(),6).setZero();
      }
    }
  }

  int ResidualBlocks() const override { return entries_.size()+(!snapshot_.prior.Empty()); }

 private:
  struct Parameter { int ambient_size, column; };
  struct Entry {
    std::unique_ptr<ceres::CostFunction> cost;
    std::vector<const double*> pointers;
    std::vector<int> columns;
    std::vector<splbatch::DynamicRowMajorMatrix> buffers;
    std::vector<double*> jacobians;
    int row=0;
    std::exception_ptr failure;
  };
  const Snapshot& snapshot_;
  State& state_;
  std::unordered_map<double*,Parameter> parameters_;
  std::vector<Entry> entries_;
  int measurement_rows_=0;
};
}  // namespace

std::unique_ptr<NativeBatchProblem> BuildNativeBatchProblem(const Snapshot& snapshot, State& state) {
  return std::make_unique<NativeRegistry>(snapshot,state);
}

Linearization LinearizeNativeBatch(const Snapshot& snapshot, const State& input, int threads) {
  State state=input;
  auto problem=BuildNativeBatchProblem(snapshot,state);
  Linearization result;
  problem->Evaluate(threads,result.residual,&result.jacobian);
  return result;
}
}  // namespace slict::comparison
