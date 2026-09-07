#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ceres/problem.h>
#include <ceres/solver.h>

#include "splbatch/batch_binding.hpp"
#include "splbatch/compressed_segment_batch_cost_function.hpp"
#include "splbatch/construction_diagnostics.hpp"
#include "splbatch/evaluator_base.hpp"
#include "splbatch/inline_binding_input.hpp"
#include "splbatch/loss_policy.hpp"
#include "splbatch/residual_registry.hpp"
#include "splbatch/segment_batch_cost_function.hpp"

namespace splbatch
{
  class ChannelBase
  {
  public:
    virtual ~ChannelBase() = default;
    virtual void CommitTo(ResidualRegistry &registry) = 0;
    void CommitTo(ceres::Problem &problem)
    {
      CeresResidualRegistry registry(problem);
      CommitTo(registry);
    }
  };

  template <typename Evaluator, typename LossPolicy = NoLossPolicy,
            template <typename, typename> class CostFunctionTemplate =
                SegmentBatchCostFunction>
  class EvaluatorChannel final : public ChannelBase
  {
    static_assert(
        kIsEvaluator<Evaluator>,
        "Evaluator must derive from splbatch::EvaluatorBase");

  public:
    using Observation = typename Evaluator::Observation;
    using Binding = typename Evaluator::Binding;
    using PreparedObservation = typename Evaluator::PreparedObservation;
    using CostFunction = CostFunctionTemplate<Evaluator, LossPolicy>;

    EvaluatorChannel(Evaluator evaluator, LossPolicy loss_policy)
        : evaluator_(
              std::make_shared<const Evaluator>(std::move(evaluator))),
          loss_policy_(std::move(loss_policy))
    {
    }

    void AddObservation(const Binding &binding,
                        const Observation &observation)
    {
      AddObservation(binding, observation, 1.0);
    }

    /// Add one observation with the outer scale of a Ceres ScaledLoss.
    /// For IndependentHuberLossPolicy this represents
    /// ScaledLoss(Huber(channel_delta), observation_loss_scale).
    void AddObservation(const Binding &binding,
                        const Observation &observation,
                        const double observation_loss_scale)
    {
      AddObservationImpl(binding, observation, observation_loss_scale);
    }

    template <typename Context, std::size_t N, std::size_t S>
    void AddObservation(const InlineBindingInput<Context, N, S> &input,
                        const Observation &observation,
                        const double observation_loss_scale = 1.0)
    {
      static_assert(std::is_same<Binding, BatchBinding<Context>>::value,
                    "Inline input must match the evaluator's owning BatchBinding");
      AddObservationImpl(input, observation, observation_loss_scale);
    }

  private:
    template <typename Input>
    void AddObservationImpl(const Input &binding,
                            const Observation &observation,
                            const double observation_loss_scale)
    {
      SPLBATCH_DIAG_SCOPE(ingestion, ObservationStorage, true);
      if (sealed_)
      {
        throw std::logic_error(
            "Cannot add an observation after a channel was committed");
      }
      ValidateObservationLossScale(observation_loss_scale);

      SPLBATCH_DIAG_SCOPE(lookup, Support, true);
      std::size_t index = last_batch_;
      std::size_t hash = 0;
      if (index == kNoBatch ||
          !detail::SameBatchSupport(pending_batches_[index].binding, binding))
      {
        index = kNoBatch;
        hash = detail::HashBatchSupport(binding);
        const auto candidates = batch_indices_.equal_range(hash);
        for (auto candidate = candidates.first;
             candidate != candidates.second; ++candidate)
        {
          if (detail::SameBatchSupport(
                  pending_batches_[candidate->second].binding, binding))
          {
            index = candidate->second;
            break;
          }
        }
      }
      // Every stored support was fully validated before publication and is
      // immutable during ingestion. Only an exact match may skip validation.
      if (index == kNoBatch)
        binding.parameters.Validate();
      SPLBATCH_DIAG_STOP(lookup);
      if (index == kNoBatch)
      {
        index = pending_batches_.size();
        PendingBatch pending{OwnBinding(binding), {}, {}};
        AppendObservation(pending, binding, observation, observation_loss_scale, true);
        pending_batches_.push_back(std::move(pending));
        SPLBATCH_DIAG_SCOPE(insert, Support, true);
        try
        {
          batch_indices_.emplace(hash, index);
        }
        catch (...)
        {
          // Allocation/rehash failure must not publish an unindexed batch.
          pending_batches_.pop_back();
          throw;
        }
      }
      else
      {
        AppendObservation(pending_batches_[index], binding, observation,
                          observation_loss_scale);
      }
      // Indices survive pending_batches_ growth; failed appends leave the
      // hint untouched. It is a candidate, never an observation assignment.
      last_batch_ = index;
    }

  public:
    std::size_t PendingBatchCount() const
    {
      return pending_batches_.size();
    }

    std::size_t PendingObservationCount() const
    {
      std::size_t count = 0;
      for (const PendingBatch &batch : pending_batches_)
        count += batch.observations.size();
      return count;
    }

    using ChannelBase::CommitTo;
    void CommitTo(ResidualRegistry &registry) override
    {
      if (sealed_)
        return;

      for (PendingBatch &pending : pending_batches_)
      {
        SPLBATCH_DIAG_SCOPE(creation, BatchCreation, false);
        std::vector<double *> parameter_blocks =
            pending.binding.parameters.blocks;
        auto cost = std::make_unique<CostFunction>(
            evaluator_, std::move(pending.binding),
            std::move(pending.observations),
            std::move(pending.observation_loss_scales), loss_policy_);
        SPLBATCH_DIAG_STOP(creation);
        registry.AddResidualBlock(std::move(cost), std::move(parameter_blocks));
      }

      SPLBATCH_DIAG_SCOPE(cleanup, ChannelCleanup, false);
      batch_indices_.clear();
      pending_batches_.clear();
      last_batch_ = kNoBatch;
      sealed_ = true;
    }

  private:
    struct PendingBatch
    {
      Binding binding;
      std::vector<PreparedObservation> observations;
      std::vector<double> observation_loss_scales;
    };

    const Binding &OwnBinding(const Binding &binding) const { return binding; }

    template <typename Context, std::size_t N, std::size_t S>
    Binding OwnBinding(const InlineBindingInput<Context, N, S> &input) const
    { return input.ToBinding(); }

    PreparedObservation PrepareInput(const Binding &input, const Binding &,
                                     const Observation &observation, bool) const
    {
      // The existing owning-input API always uses the actual incoming Binding.
      return evaluator_->PrepareObservation(input, observation);
    }

    template <typename Context, std::size_t N, std::size_t S>
    PreparedObservation PrepareInput(const InlineBindingInput<Context, N, S> &input,
                                     const Binding &stored,
                                     const Observation &observation, bool is_new) const
    {
      // A new batch was just materialized from this input. On a hit, reuse is
      // allowed only by an explicit evaluator-specific semantic contract.
      if (is_new || BindingPreparationReuse<Evaluator>::Compatible(stored, input.context))
        return evaluator_->PrepareObservation(stored, observation);
      const Binding current = input.ToBinding();
      return evaluator_->PrepareObservation(current, observation);
    }

    template <typename Input>
    void AppendObservation(PendingBatch &pending, const Input &binding,
                           const Observation &observation, const double scale,
                           bool is_new = false)
    {
      // Preserve this call's preparation semantics, even when context changed
      // on a support hit. Never mutate a stored batch's representative context.
      auto prepared = SPLBATCH_DIAG_MEASURE(
          Prepare, true, PrepareInput(binding, pending.binding, observation, is_new));
      pending.observation_loss_scales.push_back(scale);
      try
      {
        pending.observations.push_back(std::move_if_noexcept(prepared));
      }
      catch (...)
      {
        pending.observation_loss_scales.pop_back();
        throw;
      }
    }

    static constexpr std::size_t kNoBatch =
        std::numeric_limits<std::size_t>::max();
    std::shared_ptr<const Evaluator> evaluator_;
    LossPolicy loss_policy_;
    std::unordered_multimap<std::size_t, std::size_t> batch_indices_;
    std::vector<PendingBatch> pending_batches_;
    std::size_t last_batch_ = kNoBatch;
    bool sealed_ = false;
  };

  class ProblemBatcher
  {
  public:
    explicit ProblemBatcher(ceres::Problem &problem)
        : problem_(&problem),
          ceres_registry_(std::make_unique<CeresResidualRegistry>(problem)),
          registry_(ceres_registry_.get()) {}

    explicit ProblemBatcher(ResidualRegistry &registry) : registry_(&registry) {}

    ProblemBatcher(const ProblemBatcher &) = delete;
    ProblemBatcher &operator=(const ProblemBatcher &) = delete;

    template <typename Evaluator, typename LossPolicy = NoLossPolicy>
    EvaluatorChannel<Evaluator, LossPolicy> &RegisterEvaluator(
        Evaluator evaluator, LossPolicy loss_policy = LossPolicy{})
    {
      static_assert(
          kIsEvaluator<Evaluator>,
          "Evaluator must derive from splbatch::EvaluatorBase");
      if (committed_)
      {
        throw std::logic_error(
            "Cannot register an evaluator after committing the batcher");
      }
      auto channel =
          std::make_unique<EvaluatorChannel<Evaluator, LossPolicy>>(
              std::move(evaluator), std::move(loss_policy));
      auto &result = *channel;
      channels_.push_back(std::move(channel));
      return result;
    }

    /// Register the same evaluator API as RegisterEvaluator(), but expose
    /// one fixed-size square-root normal factor per batch to Ceres. H and g
    /// are rebuilt at every linearization, so callers add observations and
    /// solve exactly as they do for an ordinary evaluator channel.
    template <typename Evaluator, typename LossPolicy = NoLossPolicy>
    EvaluatorChannel<
        Evaluator, LossPolicy,
        CompressedSegmentBatchCostFunction> &
    RegisterCompressedEvaluator(
        Evaluator evaluator, LossPolicy loss_policy = LossPolicy{})
    {
      static_assert(
          kIsEvaluator<Evaluator>,
          "Evaluator must derive from splbatch::EvaluatorBase");
      if (committed_)
      {
        throw std::logic_error(
            "Cannot register an evaluator after committing the batcher");
      }
      using Channel = EvaluatorChannel<
          Evaluator, LossPolicy,
          CompressedSegmentBatchCostFunction>;
      auto channel = std::make_unique<Channel>(
          std::move(evaluator), std::move(loss_policy));
      auto &result = *channel;
      channels_.push_back(std::move(channel));
      return result;
    }

    void Commit()
    {
      if (committed_)
        return;
      for (const auto &channel : channels_)
        channel->CommitTo(*registry_);
      committed_ = true;
    }

    void Solve(const ceres::Solver::Options &options,
               ceres::Solver::Summary *summary)
    {
      if (summary == nullptr)
        throw std::invalid_argument("Ceres summary is null");
      if (problem_ == nullptr)
        throw std::logic_error("Solve is only available for a Ceres-backed batcher");
      Commit();
      ceres::Solve(options, problem_, summary);
    }

  private:
    ceres::Problem *problem_ = nullptr;
    std::unique_ptr<CeresResidualRegistry> ceres_registry_;
    ResidualRegistry *registry_;
    std::vector<std::unique_ptr<ChannelBase>> channels_;
    bool committed_ = false;
  };
} // namespace splbatch
