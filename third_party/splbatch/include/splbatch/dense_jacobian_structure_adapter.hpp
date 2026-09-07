#pragma once

#include <type_traits>
#include <utility>

#include "splbatch/evaluator_base.hpp"

namespace splbatch
{
  /// Diagnostic adapter that deliberately suppresses an evaluator's optional
  /// JacobianStructureImpl() declaration. Compressed factors then use the
  /// conservative dense fallback. This is useful for numerical and timing A/B
  /// checks; normal applications should register the evaluator directly.
  template <typename WrappedEvaluator>
  class DenseJacobianStructureAdapter final
      : public EvaluatorBase<
            DenseJacobianStructureAdapter<WrappedEvaluator>,
            typename WrappedEvaluator::Observation,
            typename WrappedEvaluator::Binding,
            typename WrappedEvaluator::PreparedObservation,
            typename WrappedEvaluator::Runtime,
            WrappedEvaluator::kResidualDim>
  {
    static_assert(
        kIsEvaluator<WrappedEvaluator>,
        "WrappedEvaluator must derive from splbatch::EvaluatorBase");

  public:
    using Observation = typename WrappedEvaluator::Observation;
    using Binding = typename WrappedEvaluator::Binding;
    using PreparedObservation =
        typename WrappedEvaluator::PreparedObservation;
    using Runtime = typename WrappedEvaluator::Runtime;
    static constexpr bool kHasResidualProjection =
        WrappedEvaluator::kHasResidualProjection;

    explicit DenseJacobianStructureAdapter(WrappedEvaluator evaluator)
        : evaluator_(std::move(evaluator))
    {
    }

    PreparedObservation PrepareObservationImpl(
        const Binding &binding, const Observation &observation) const
    {
      return evaluator_.PrepareObservation(binding, observation);
    }

    Runtime PrepareBatchImpl(
        const Binding &binding, double const *const *parameters,
        const EvaluationRequest &request) const
    {
      return evaluator_.PrepareBatch(binding, parameters, request);
    }

    bool EvaluateObservationImpl(
        const Binding &binding,
        const PreparedObservation &observation,
        const Runtime &runtime, double const *const *parameters,
        const ObservationOutput &output) const
    {
      return evaluator_.EvaluateObservation(
          binding, observation, runtime, parameters, output);
    }

  private:
    WrappedEvaluator evaluator_;
  };

  template <typename Evaluator>
  DenseJacobianStructureAdapter<typename std::decay<Evaluator>::type>
  MakeDenseJacobianStructureAdapter(Evaluator &&evaluator)
  {
    using Adapter = DenseJacobianStructureAdapter<
        typename std::decay<Evaluator>::type>;
    return Adapter(std::forward<Evaluator>(evaluator));
  }
} // namespace splbatch
