#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <ceres/cost_function.h>

#include "splbatch/evaluator_base.hpp"
#include "splbatch/evaluation_request.hpp"
#include "splbatch/jacobian_writer.hpp"
#include "splbatch/loss_policy.hpp"

namespace splbatch
{
  template <typename Evaluator, typename LossPolicy>
  class SegmentBatchCostFunction final : public ceres::CostFunction
  {
    static_assert(
        kIsEvaluator<Evaluator>,
        "Evaluator must derive from splbatch::EvaluatorBase");

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Binding = typename Evaluator::Binding;
    using SharedBinding = std::shared_ptr<const Binding>;
    using PreparedObservation = typename Evaluator::PreparedObservation;
    using SharedPreparedObservations =
        std::shared_ptr<std::vector<PreparedObservation>>;
    using Runtime = typename Evaluator::Runtime;

    SegmentBatchCostFunction(
        std::shared_ptr<const Evaluator> evaluator, Binding binding,
        std::vector<PreparedObservation> observations, LossPolicy loss_policy)
        : SegmentBatchCostFunction(
              std::move(evaluator), std::move(binding),
              std::move(observations), {}, std::move(loss_policy))
    {
    }

    SegmentBatchCostFunction(
        std::shared_ptr<const Evaluator> evaluator, Binding binding,
        std::vector<PreparedObservation> observations,
        std::vector<double> observation_loss_scales,
        LossPolicy loss_policy)
        : evaluator_(std::move(evaluator)), binding_(std::make_shared<Binding>(std::move(binding))),
          observations_(std::move(observations)),
          observation_loss_scales_(std::move(observation_loss_scales)),
          loss_policy_(std::move(loss_policy))
    {
      Initialize();
    }

    /// Incremental batches keep prepared observations in shared contiguous
    /// storage with stable indices.
    /// When spline semantics change without changing parameter support, the
    /// channel can refresh that slot in place and the already registered
    /// Ceres CostFunction immediately sees the new coefficients.
    SegmentBatchCostFunction(
        std::shared_ptr<const Evaluator> evaluator, Binding binding,
        SharedPreparedObservations observations,
        std::vector<double> observation_loss_scales,
        LossPolicy loss_policy)
        : SegmentBatchCostFunction(std::move(evaluator),
              std::make_shared<Binding>(std::move(binding)),
              std::move(observations), std::move(observation_loss_scales),
              std::move(loss_policy), nullptr)
    {
    }

    // The channel supplies a validated immutable support and a write plan
    // compiled for this exact layout. Coefficient versions are not plan keys.
    SegmentBatchCostFunction(
        std::shared_ptr<const Evaluator> evaluator, SharedBinding binding,
        SharedPreparedObservations observations,
        std::vector<double> observation_loss_scales, LossPolicy loss_policy,
        std::shared_ptr<const JacobianWritePlan> write_plan)
        : evaluator_(std::move(evaluator)), binding_(std::move(binding)),
          shared_observations_(std::move(observations)),
          observation_loss_scales_(std::move(observation_loss_scales)),
          loss_policy_(std::move(loss_policy)), write_plan_(std::move(write_plan))
    {
      Initialize();
    }

    bool Evaluate(double const *const *parameters, double *residuals,
                  double **jacobians) const override
    {
      const EvaluationRequest request(
          jacobians, binding_->parameters.sizes.size());
      JacobianWriter writer(
          residuals, jacobians, num_residuals(), binding_->parameters.sizes,
          Evaluator::kResidualDim, write_plan_.get());
      if constexpr (Evaluator::kOverwritesActiveJacobian)
        writer.ZeroInactiveJacobians(ObservationCount());
      else
        writer.ZeroRequestedJacobians();

      const Runtime runtime =
          evaluator_->PrepareBatch(*binding_, parameters, request);
      double accumulated_cost_gap = 0.0;
      for (std::size_t index = 0; index < ObservationCount(); ++index)
      {
        const ObservationOutput output = writer.Observation(index);
        if (!evaluator_->EvaluateObservation(
                *binding_, ObservationAt(index), runtime, parameters, output))
        {
          return false;
        }
        accumulated_cost_gap +=
            loss_policy_.Apply(output, observation_loss_scales_[index]);
      }
      loss_policy_.Finalize(
          residuals, data_row_count_, accumulated_cost_gap);
      if constexpr (Evaluator::kHasResidualProjection)
      {
        // NoLossPolicy has no slack row of its own.
        if (loss_policy_.ExtraResidualRows() == 0)
          residuals[data_row_count_] = std::sqrt(std::max(0.0, accumulated_cost_gap));
      }
      return true;
    }

  private:
    std::size_t ObservationCount() const
    {
      return shared_observations_
                 ? shared_observations_->size()
                 : observations_.size();
    }

    const PreparedObservation &ObservationAt(const std::size_t index) const
    {
      if (!shared_observations_)
        return observations_.at(index);
      return shared_observations_->at(index);
    }

    void Initialize()
    {
      static_assert(Evaluator::kResidualDim > 0,
                    "An evaluator must have a positive residual dimension");
      if (!evaluator_ || !binding_)
        throw std::invalid_argument("Batch evaluator is null");
      if (!observations_.empty() && shared_observations_)
      {
        throw std::invalid_argument(
            "A segment batch cannot mix owned and shared observations");
      }
      if (ObservationCount() == 0)
        throw std::invalid_argument("A segment batch cannot be empty");
      if (observation_loss_scales_.empty())
        observation_loss_scales_.assign(ObservationCount(), 1.0);
      if (observation_loss_scales_.size() != ObservationCount())
      {
        throw std::invalid_argument(
            "Observation and loss-scale counts differ");
      }
      for (const double loss_scale : observation_loss_scales_)
        ValidateObservationLossScale(loss_scale);

      if (!write_plan_)
      {
        binding_->parameters.Validate();
        write_plan_ = std::make_shared<JacobianWritePlan>(
            evaluator_->GetJacobianWriteStructure(*binding_), Evaluator::kResidualDim,
            binding_->parameters.sizes);
      }
      data_row_count_ = static_cast<int>(ObservationCount()) *
                        Evaluator::kResidualDim;
      set_num_residuals(
          data_row_count_ + std::max(loss_policy_.ExtraResidualRows(),
                                    Evaluator::kHasResidualProjection ? 1 : 0));
      mutable_parameter_block_sizes()->reserve(binding_->parameters.sizes.size());
      for (const int32_t size : binding_->parameters.sizes)
        mutable_parameter_block_sizes()->push_back(size);
    }

    std::shared_ptr<const Evaluator> evaluator_;
    SharedBinding binding_;
    std::vector<PreparedObservation> observations_;
    SharedPreparedObservations shared_observations_;
    std::vector<double> observation_loss_scales_;
    LossPolicy loss_policy_;
    int data_row_count_ = 0;
    std::shared_ptr<const JacobianWritePlan> write_plan_;
  };
} // namespace splbatch
