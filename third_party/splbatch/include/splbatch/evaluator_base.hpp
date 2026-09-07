#pragma once

#include <type_traits>
#include <utility>

#include "splbatch/evaluation_request.hpp"
#include "splbatch/jacobian_structure.hpp"
#include "splbatch/jacobian_writer.hpp"

namespace splbatch
{
  namespace detail
  {
    template <typename Derived>
    struct EvaluatorTag
    {
    };

    template <typename T>
    using RemoveCVRef =
        typename std::remove_cv<
            typename std::remove_reference<T>::type>::type;

    template <typename Derived, typename Binding, typename = void>
    struct HasJacobianStructureImpl : std::false_type
    {
    };

    template <typename Derived, typename Binding>
    struct HasJacobianStructureImpl<
        Derived, Binding,
        std::void_t<decltype(
            std::declval<const Derived &>().JacobianStructureImpl(
                std::declval<const Binding &>()))>> : std::true_type
    {
    };

    template <typename Derived, typename Binding, typename = void>
    struct HasJacobianWriteStructureImpl : std::false_type {};
    template <typename Derived, typename Binding>
    struct HasJacobianWriteStructureImpl<Derived, Binding, std::void_t<decltype(
        std::declval<const Derived &>().JacobianWriteStructureImpl(
            std::declval<const Binding &>()))>> : std::true_type {};

    template <typename D, typename B, typename O, typename P, typename = void>
    struct HasRefreshObservation : std::false_type {};
    template <typename D, typename B, typename O, typename P>
    struct HasRefreshObservation<D, B, O, P, std::void_t<decltype(
        std::declval<const D &>().RefreshObservationImpl(
            std::declval<const B &>(), std::declval<const B &>(),
            std::declval<const O &>(), std::declval<P &>()))>> : std::true_type {};

    // Optional two-phase numeric update. Prepare may throw but must not mutate
    // live data; Apply must be noexcept. This permits small field-only patches
    // without copying immutable measurements or unchanged coefficient arrays.
    // Evaluators without this API retain the full-copy / buffer-swap path.
    template <typename E, typename = void>
    struct ObservationRefreshTraits
    {
      using Update = typename E::PreparedObservation;
      static constexpr bool kPartial = false;
      static Update Prepare(const E &e, const typename E::Binding &old,
          const typename E::Binding &binding, const typename E::Observation &obs,
          const typename E::PreparedObservation &prepared)
      {
        Update result = prepared;
        e.RefreshObservation(old, binding, obs, result);
        return result;
      }
      static void Apply(const E &, Update &update, typename E::PreparedObservation &prepared)
      { prepared = std::move(update); }
    };

    template <typename E>
    struct ObservationRefreshTraits<E, std::void_t<typename E::ObservationRefresh>>
    {
      using Update = typename E::ObservationRefresh;
      static constexpr bool kPartial = true;
      static_assert(noexcept(std::declval<const E &>().ApplyObservationRefreshImpl(
          std::declval<const Update &>(), std::declval<typename E::PreparedObservation &>())),
          "Publishing a staged observation refresh must be noexcept");
      static Update Prepare(const E &e, const typename E::Binding &old,
          const typename E::Binding &binding, const typename E::Observation &obs,
          const typename E::PreparedObservation &prepared)
      { return e.PrepareObservationRefreshImpl(old, binding, obs, prepared); }
      static void Apply(const E &e, const Update &update, typename E::PreparedObservation &prepared) noexcept
      { e.ApplyObservationRefreshImpl(update, prepared); }
    };
  } // namespace detail

  /// Compile-time interface for a batch evaluator.
  ///
  /// Derived evaluators implement the three corresponding *Impl() methods.
  /// They may additionally implement
  ///
  ///   JacobianStructure JacobianStructureImpl(const Binding &) const;
  ///
  /// to declare structurally nonzero residual-row/parameter-column slices.
  /// The declaration is used by compressed normal-equation factors. If it is
  /// omitted, every Jacobian entry is conservatively treated as nonzero.
  /// CRTP keeps the observation, binding, prepared-observation, and runtime
  /// types specific to each factor while avoiding per-observation virtual
  /// dispatch.
  template <typename DerivedT, typename ObservationT, typename BindingT,
            typename PreparedObservationT, typename RuntimeT,
            int ResidualDimension>
  class EvaluatorBase : public detail::EvaluatorTag<DerivedT>
  {
  public:
    static_assert(ResidualDimension > 0,
                  "An evaluator residual dimension must be positive");

    using Derived = DerivedT;
    using Observation = ObservationT;
    using Binding = BindingT;
    using PreparedObservation = PreparedObservationT;
    using Runtime = RuntimeT;

    static constexpr int kResidualDim = ResidualDimension;
    // Opt in only when EvaluateObservation overwrites EVERY declared active
    // rectangle (no += into uninitialized output). Other evaluators keep the
    // conservative full-zero path, including custom/dense adapters.
    static constexpr bool kOverwritesActiveJacobian = false;
    // Projected local models need a zero-Jacobian row even without a loss.
    static constexpr bool kHasResidualProjection = false;

    PreparedObservation PrepareObservation(
        const Binding &binding, const Observation &observation) const
    {
      return derived().PrepareObservationImpl(binding, observation);
    }

    // Optional selective numeric refresh. The raw observation and ordered
    // parameter support MUST be unchanged. Work on caller-owned staging data;
    // throwing must not publish a partial update to a registered factor.
    void RefreshObservation(const Binding &old_binding, const Binding &binding,
                            const Observation &observation,
                            PreparedObservation &prepared) const
    {
      if constexpr (detail::ObservationRefreshTraits<Derived>::kPartial)
      {
        using Refresh = detail::ObservationRefreshTraits<Derived>;
        const auto update = Refresh::Prepare(derived(), old_binding, binding, observation, prepared);
        Refresh::Apply(derived(), update, prepared);
      }
      else if constexpr (detail::HasRefreshObservation<Derived, Binding, Observation,
                                                  PreparedObservation>::value)
        derived().RefreshObservationImpl(old_binding, binding, observation, prepared);
      else
        prepared = PrepareObservation(binding, observation);
    }

    Runtime PrepareBatch(
        const Binding &binding, double const *const *parameters,
        const EvaluationRequest &request) const
    {
      return derived().PrepareBatchImpl(binding, parameters, request);
    }

    template <typename Output>
    bool EvaluateObservation(
        const Binding &binding,
        const PreparedObservation &observation,
        const Runtime &runtime, double const *const *parameters,
        const Output &output) const
    {
      return derived().EvaluateObservationImpl(
          binding, observation, runtime, parameters, output);
    }

    JacobianStructure GetJacobianStructure(
        const Binding &binding) const
    {
      if constexpr (
          detail::HasJacobianStructureImpl<Derived, Binding>::value)
      {
        static_assert(
            std::is_convertible<
                decltype(std::declval<const Derived &>()
                             .JacobianStructureImpl(
                                 std::declval<const Binding &>())),
                JacobianStructure>::value,
            "JacobianStructureImpl must return splbatch::JacobianStructure");
        return derived().JacobianStructureImpl(binding);
      }
      else
      {
        return MakeDenseJacobianStructure(
            kResidualDim, binding.parameters.sizes);
      }
    }

    /// A finer partition may be useful for direct writes while compression
    /// deliberately groups rows into larger, fixed-size dense GEMM kernels.
    JacobianStructure GetJacobianWriteStructure(const Binding &binding) const
    {
      if constexpr (detail::HasJacobianWriteStructureImpl<Derived, Binding>::value)
        return derived().JacobianWriteStructureImpl(binding);
      else
        return GetJacobianStructure(binding);
    }

  protected:
    EvaluatorBase() = default;
    ~EvaluatorBase() = default;
    EvaluatorBase(const EvaluatorBase &) = default;
    EvaluatorBase(EvaluatorBase &&) = default;
    EvaluatorBase &operator=(const EvaluatorBase &) = default;
    EvaluatorBase &operator=(EvaluatorBase &&) = default;

  private:
    const Derived &derived() const
    {
      return static_cast<const Derived &>(*this);
    }
  };

  template <typename Evaluator>
  struct IsEvaluator
      : std::is_base_of<
            detail::EvaluatorTag<detail::RemoveCVRef<Evaluator>>,
            detail::RemoveCVRef<Evaluator>>
  {
  };

  template <typename Evaluator>
  constexpr bool kIsEvaluator = IsEvaluator<Evaluator>::value;
} // namespace splbatch
