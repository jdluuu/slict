// Portable tests extracted from CT-RIO; see NOTICE.md.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <ceres/ceres.h>
#include <splbatch/incremental_problem_batcher.hpp>

namespace
{
  struct ScalarContext
  {
    double preparation_offset = 0.0;
  };

  using ScalarBinding = splbatch::BatchBinding<ScalarContext>;

  struct ScalarObservation
  {
    double measurement = 0.0;
  };

  struct ScalarRuntime
  {
  };

  class ScalarEvaluator final
      : public splbatch::EvaluatorBase<
            ScalarEvaluator, ScalarObservation, ScalarBinding, double,
            ScalarRuntime, 1>
  {
  public:
    explicit ScalarEvaluator(
        std::shared_ptr<std::size_t> preparation_count = nullptr,
        std::shared_ptr<std::size_t> structure_count = nullptr,
        std::shared_ptr<bool> fail_preparation = nullptr)
        : preparation_count_(std::move(preparation_count)),
          structure_count_(std::move(structure_count)), fail_preparation_(std::move(fail_preparation))
    {
    }

    static constexpr int kStaticNormalGroupRows = 1;
    static constexpr int kStaticNormalGroupColumns = 1;

    double PrepareObservationImpl(
        const Binding &binding, const Observation &observation) const
    {
      if (preparation_count_)
        ++*preparation_count_;
      if (fail_preparation_ && *fail_preparation_ && observation.measurement == 2.0)
        throw std::runtime_error("Injected preparation failure");
      return observation.measurement +
             binding.context.preparation_offset;
    }

    ScalarRuntime PrepareBatchImpl(
        const Binding &, double const *const *,
        const splbatch::EvaluationRequest &) const
    {
      return {};
    }

    template <typename Output>
    bool EvaluateObservationImpl(
        const Binding &, const double measurement, const ScalarRuntime &,
        double const *const *parameters, const Output &output) const
    {
      output.Residual()[0] = parameters[0][0] - measurement;
      if (output.JacobianRequested(0))
      {
        output.Jacobian(0).template block<1, 1>(0, 0)
            .setConstant(1.0);
      }
      return true;
    }

    splbatch::JacobianStructure JacobianStructureImpl(
        const Binding &) const
    {
      if (structure_count_) ++*structure_count_;
      return {{{{0, 1}, {{0, 0, 1}}}}};
    }

  private:
    std::shared_ptr<std::size_t> preparation_count_;
    std::shared_ptr<std::size_t> structure_count_;
    std::shared_ptr<bool> fail_preparation_;
  };

  ScalarBinding MakeBinding(
      double *parameter, const std::size_t support,
      const double preparation_offset = 0.0)
  {
    ScalarBinding binding;
    binding.context.preparation_offset = preparation_offset;
    binding.parameters.blocks = {parameter};
    binding.parameters.sizes = {1};
    binding.support_signature = {support};
    return binding;
  }

  // A large immutable payload makes accidental full-prepared copies visible.
  struct CopyTrackedPrepared
  {
    inline static std::size_t copies = 0;
    double measurement = 0;
    std::array<double, 64> immutable{};
    CopyTrackedPrepared() = default;
    explicit CopyTrackedPrepared(double value) : measurement(value) {}
    CopyTrackedPrepared(const CopyTrackedPrepared &other)
        : measurement(other.measurement), immutable(other.immutable) { ++copies; }
    CopyTrackedPrepared &operator=(const CopyTrackedPrepared &other)
    { measurement = other.measurement; immutable = other.immutable; ++copies; return *this; }
    CopyTrackedPrepared(CopyTrackedPrepared &&) = default;
    CopyTrackedPrepared &operator=(CopyTrackedPrepared &&) = default;
  };

  class PatchScalarEvaluator final : public splbatch::EvaluatorBase<
      PatchScalarEvaluator, ScalarObservation, ScalarBinding, CopyTrackedPrepared, ScalarRuntime, 1>
  {
    ScalarEvaluator scalar_;
  public:
    explicit PatchScalarEvaluator(std::shared_ptr<bool> fail = nullptr)
        : scalar_(nullptr, nullptr, std::move(fail)) {}
    using ObservationRefresh = double;
    PreparedObservation PrepareObservationImpl(const Binding &b, const Observation &o) const
    { return PreparedObservation(scalar_.PrepareObservation(b, o)); }
    ObservationRefresh PrepareObservationRefreshImpl(const Binding &, const Binding &b,
        const Observation &o, const PreparedObservation &) const
    { return scalar_.PrepareObservation(b, o); }
    void ApplyObservationRefreshImpl(double update, PreparedObservation &prepared) const noexcept
    { prepared.measurement = update; }
    Runtime PrepareBatchImpl(const Binding &, double const *const *, const splbatch::EvaluationRequest &) const
    { return {}; }
    template <typename Output>
    bool EvaluateObservationImpl(const Binding &b, const PreparedObservation &p, const Runtime &runtime,
        double const *const *parameters, const Output &output) const
    { return scalar_.EvaluateObservationImpl(b, p.measurement, runtime, parameters, output); }
  };

  void Require(const bool condition, const std::string &message)
  {
    if (!condition)
      throw std::runtime_error(message);
  }

  double EvaluateCost(ceres::Problem &problem)
  {
    ceres::Problem::EvaluateOptions options;
    options.apply_loss_function = true;
    options.num_threads = 1;
    double cost = 0.0;
    if (!problem.Evaluate(options, &cost, nullptr, nullptr, nullptr))
      throw std::runtime_error("Ceres problem evaluation failed");
    return cost;
  }

  void TestIndexedWindowCutoff()
  {
    double parameter = 0;
    ceres::Problem problem;
    splbatch::CeresIncrementalBatchBackend backend(problem);
    splbatch::IncrementalProblemBatcher batcher(backend);
    auto &channel = batcher.RegisterEvaluator(ScalarEvaluator{});
    for (std::size_t i = 0; i < 512; ++i)
      channel.AddObservation(MakeBinding(&parameter, i), {1.0}, 1000 + 3 * i);
    batcher.Flush();
    Require(channel.StageWindowCutoff(999) == 0 && channel.LastWindowVisitedBatches() == 0,
            "Cutoff visited supports wholly newer than the boundary");
    Require(channel.StageWindowCutoff(1000) == 1 && channel.LastWindowVisitedBatches() == 1,
            "Cutoff did not visit only the single expiring support");
    Require(problem.NumResiduals() == 512, "Staged expiry changed a live factor before Flush");
    batcher.Flush();
    Require(problem.NumResiduals() == 511, "Indexed cutoff lost a retained support");
    Require(channel.StageWindowCutoff(1000) == 0 && channel.LastWindowVisitedBatches() == 0,
            "Repeated cutoff revisited retained supports");
    batcher.Clear();

    // Differential lifecycle check against a full-scan oracle. Includes
    // out-of-order insertion, arbitrary removal, time changes, support
    // split/merge, repeated/decreasing cutoffs and cancellation before Flush.
    struct Row { int64_t time; double value; };
    std::map<std::uint64_t, Row> rows;
    std::uint32_t random = 0x2a97413u;
    const auto next = [&]() {
      random ^= random << 13; random ^= random >> 17; random ^= random << 5;
      return random;
    };
    const auto verify = [&]() {
      batcher.Flush();
      double expected = 0;
      for (const auto &[id, row] : rows) expected += 0.5 * row.value * row.value;
      Require(channel.ObservationCount() == rows.size() &&
                  problem.NumResiduals() == static_cast<int>(rows.size()),
              "Indexed expiry and full-scan oracle disagree on row membership");
      Require(std::abs(EvaluateCost(problem) - expected) < 1e-10,
              "Indexed expiry changed retained residual values");
    };
    for (int step = 0; step < 1500; ++step)
    {
      const auto action = next() % 10;
      if (action < 6)
      {
        const int64_t time = static_cast<int64_t>(next() % 400) - 200;
        const double value = next() % 7 + 1;
        const auto id = channel.AddObservation(MakeBinding(&parameter, next() % 8), {value}, time);
        rows.emplace(id, Row{time, value});
      }
      else if (action == 6 && !rows.empty())
      {
        auto it = std::next(rows.begin(), next() % rows.size());
        Require(channel.RemoveObservation(it->first), "Arbitrary removal failed");
        rows.erase(it);
      }
      else if (action == 7)
      {
        const int64_t cutoff = static_cast<int64_t>(next() % 400) - 200;
        std::size_t expired = 0;
        for (auto it = rows.begin(); it != rows.end();)
          if (it->second.time <= cutoff) { it = rows.erase(it); ++expired; }
          else ++it;
        Require(channel.StageWindowCutoff(cutoff) == expired, "Indexed cutoff differs from full scan");
      }
      else if (action == 8)
      {
        const auto support_count = next() % 8 + 1;
        channel.RebindAll([&](auto id, const auto &, const auto &, const auto &) {
          return MakeBinding(&parameter, id % support_count);
        }, 0);
      }
      else if (action == 9)
      {
        const int64_t shift = static_cast<int64_t>(next() % 41) - 20;
        channel.RefreshAffectedBatches([](const auto &) { return true; },
            [&](auto, const auto &observation, const auto &binding, auto options) {
          options.time_ns += shift;
          using Channel = std::remove_reference_t<decltype(channel)>;
          return typename Channel::ObservationUpdate{observation, binding, options};
        });
        for (auto &[id, row] : rows) row.time += shift;
      }
      if (step % 13 == 0) verify();
    }
    verify();
    Require(channel.StageWindowCutoff(std::numeric_limits<int64_t>::max()) == rows.size(),
            "Maximum cutoff did not remove all observations");
    rows.clear(); verify();
  }

  void TestCompressedRegistration()
  {
    double value = 0.25;
    ceres::Problem problem;
    splbatch::CeresIncrementalBatchBackend backend(problem);
    splbatch::IncrementalProblemBatcher batcher(backend);
    auto &channel = batcher.RegisterCompressedEvaluator(
        ScalarEvaluator{}, splbatch::NoLossPolicy{});
    channel.AddObservation(
        MakeBinding(&value, 0), ScalarObservation{1.0}, 1);
    const auto summary = batcher.Flush();
    Require(summary.added_batches == 1 &&
                backend.RegisteredBatchCount() == 1 &&
                problem.NumResidualBlocks() == 1,
            "Compressed incremental channel was not registered");
    const double old_cost = EvaluateCost(problem);
    channel.RebindAll(
        [&value](
            const std::uint64_t, const ScalarObservation &,
            const ScalarBinding &,
            const splbatch::IncrementalObservationOptions &)
        { return MakeBinding(&value, 0, 2.0); },
        0);
    Require(channel.DirtyBatchCount() == 0 &&
                std::abs(EvaluateCost(problem) - old_cost) > 1e-9,
            "Compressed batch did not read an in-place prepared refresh");
    const auto refresh_summary = batcher.Flush();
    Require(refresh_summary.added_batches == 0 &&
                refresh_summary.replaced_batches == 0 &&
                refresh_summary.removed_batches == 0,
            "Compressed coefficient refresh touched the Ceres graph");
    batcher.Clear();
    Require(problem.NumResidualBlocks() == 0,
            "Clear did not remove the compressed batch");
  }

  void TestEachObservationUsesItsOwnBindingForPreparation()
  {
    double value = 0.0;
    auto preparation_count = std::make_shared<std::size_t>(0);
    ceres::Problem problem;
    splbatch::CeresIncrementalBatchBackend backend(problem);
    splbatch::IncrementalProblemBatcher batcher(backend);
    auto &channel = batcher.RegisterEvaluator(
        ScalarEvaluator{preparation_count}, splbatch::NoLossPolicy{});

    // Both bindings intentionally have the same structural key. Their
    // contexts model two generations of mutable spline metadata whose
    // observation-time coefficients differ.
    channel.AddObservation(
        MakeBinding(&value, 0, 10.0), ScalarObservation{1.0}, 1);
    channel.AddObservation(
        MakeBinding(&value, 0, 20.0), ScalarObservation{2.0}, 2);
    const auto summary = batcher.Flush();
    Require(summary.added_batches == 1 && problem.NumResiduals() == 2,
            "Same-support observations were not batched together");
    Require(*preparation_count == 2,
            "Flush recomputed cached observation preparation");

    // Residuals must be -11 and -22. Preparing both rows with the first
    // binding would instead produce -11 and -12.
    Require(std::abs(EvaluateCost(problem) - 302.5) < 1e-12,
            "Batch preparation reused the representative binding");

    channel.AddObservation(
        MakeBinding(&value, 0, 30.0), ScalarObservation{3.0}, 3);
    batcher.Flush();
    Require(*preparation_count == 3,
            "Appending one row re-prepared the existing batch rows");
  }

  void TestCachedInsertionAndSharedStorage()
  {
    double x = 0;
    auto structures = std::make_shared<std::size_t>(0);
    auto fail = std::make_shared<bool>(false);
    ceres::Problem problem;
    splbatch::CeresIncrementalBatchBackend backend(problem);
    splbatch::IncrementalProblemBatcher batcher(backend);
    auto &channel = batcher.RegisterEvaluator(ScalarEvaluator{nullptr, structures, fail});
    int factories = 0;
    auto add = [&](int time, double offset) {
      channel.AddWithBindingCache(ScalarObservation{double(time)}, {time, 1.0, 0, 0},
          [&](const auto &b) { return b.context.preparation_offset == offset; },
          [&] { ++factories; return MakeBinding(&x, 0, offset); });
    };
    *fail = true;
    bool insertion_threw = false;
    try { add(2, 0); } catch (const std::runtime_error &) { insertion_threw = true; }
    Require(insertion_threw && channel.BatchCount() == 0 && channel.ObservationCount() == 0,
            "Failed cached preparation leaked an empty support batch");
    *fail = false;
    add(1, 0); add(2, 0); add(3, 10);
    Require(factories == 2, "Ingestion did not share the numeric binding generation");
    batcher.Flush();
    Require(std::abs(EvaluateCost(problem) - 87.0) < 1e-12,
            "Cached insertion merged different preparation generations");
    const auto refresh = [&](double offset) {
      return channel.RebindAffectedBatches([](const auto &) { return true; },
          [](auto, const auto &, const auto &old, const auto &) { return old; }, 0,
          [&](const auto &, auto &context) { context.preparation_offset = offset; return true; });
    };
    refresh(5);
    const double before_failure = EvaluateCost(problem);
    *fail = true;
    bool threw = false;
    try { refresh(100); } catch (const std::runtime_error &) { threw = true; }
    Require(threw && EvaluateCost(problem) == before_failure,
            "Failed preparation published part of the shared buffer");
    *fail = false;
    // Exercise repeated buffer reuse, mutation while membership is dirty,
    // pruning, and the next coefficient-only update after each replacement.
    for (int time = 4; time < 40; ++time)
    {
      add(time, 0);
      batcher.StageWindowCutoff(time - 3);
      refresh(time * 0.1);
      batcher.Flush();
      std::vector<ceres::ResidualBlockId> ids;
      problem.GetResidualBlocks(&ids);
      refresh(time * 0.2);
      const auto summary = batcher.Flush();
      std::vector<ceres::ResidualBlockId> new_ids;
      problem.GetResidualBlocks(&new_ids);
      double expected = 0;
      for (int t = time - 2; t <= time; ++t) expected += 0.5 * std::pow(t + time * 0.2, 2);
      Require(ids == new_ids && summary.replaced_batches == 0 &&
                  std::abs(EvaluateCost(problem) - expected) < 1e-9,
              "Shared prepared storage lost a row or its latest coefficients");
    }
    Require(*structures == 1, "Stable-layout membership updates rebuilt the Jacobian plan");
  }

  void TestSupportScopedRefresh()
  {
    double x = 0.0, y = 0.0;
    auto preparations = std::make_shared<std::size_t>(0);
    ceres::Problem problem;
    splbatch::CeresIncrementalBatchBackend backend(problem);
    splbatch::IncrementalProblemBatcher batcher(backend);
    auto &channel = batcher.RegisterEvaluator(ScalarEvaluator{preparations});
    for (int i = 1; i <= 3; ++i)
    {
      channel.AddObservation(MakeBinding(&x, 1), ScalarObservation{double(i)}, i);
      channel.AddObservation(MakeBinding(&y, 2), ScalarObservation{double(i)}, i);
    }
    batcher.Flush();
    std::vector<ceres::ResidualBlockId> original_ids;
    problem.GetResidualBlocks(&original_ids);
    int support_calls = 0, resolver_calls = 0;
    const auto changed = channel.RebindAffectedBatches(
        [](const auto &) { return true; },
        [&](auto, const auto &, const auto &old, const auto &)
        {
          ++resolver_calls;
          auto result = old;
          result.context.preparation_offset = 5.0;
          return result;
        }, 0,
        [&](const auto &view, auto &context)
        {
          ++support_calls;
          if (view.binding.parameters.blocks.front() != &x)
            return false; // exercise per-observation fallback on the other support
          context.preparation_offset = 5.0;
          return true;
        });
    Require(changed == 6 && support_calls == 2 && resolver_calls == 3 &&
                *preparations == 12 && channel.DirtyBatchCount() == 0,
            "Support refresh did not reuse one layout/skip the full resolver");
    const ScalarBinding *shared_binding = nullptr;
    channel.VisitObservationsAtOrBefore(3, [&](auto, const auto &, const auto &binding, const auto &)
    {
      if (binding.parameters.blocks.front() != &x) return;
      if (!shared_binding) shared_binding = &binding;
      Require(shared_binding == &binding, "Support refresh copied the binding per row");
    });
    Require(std::abs(EvaluateCost(problem) - 149.0) < 1e-12,
            "Shared support refresh did not update registered prepared slots");
    const auto flushed = batcher.Flush();
    std::vector<ceres::ResidualBlockId> refreshed_ids;
    problem.GetResidualBlocks(&refreshed_ids);
    Require(original_ids == refreshed_ids && flushed.replaced_batches == 0,
            "Coefficient-only refresh replaced a Ceres residual");

    // No partial publication when a later support resolver fails.
    bool threw = false;
    try
    {
      channel.RebindAffectedBatches(
          [](const auto &) { return true; },
          [](auto, const auto &, const auto &, const auto &) -> ScalarBinding
          { throw std::runtime_error("resolver failure"); }, 0,
          [&](const auto &view, auto &context)
          {
            if (view.binding.parameters.blocks.front() == &y) return false;
            context.preparation_offset = 100.0;
            return true;
          });
    }
    catch (const std::runtime_error &) { threw = true; }
    Require(threw && std::abs(EvaluateCost(problem) - 149.0) < 1e-12,
            "Failed support refresh partially changed the live problem");

    // Split/merge remains a real topology change, even with a fast refresher.
    channel.RebindAffectedBatches(
        [&](const auto &view) { return view.binding.parameters.blocks.front() == &y; },
        [&](auto, const auto &, const auto &, const auto &) { return MakeBinding(&x, 1, 5.0); },
        0, [](const auto &, auto &) { return false; });
    const auto merged = batcher.Flush();
    Require(merged.removed_batches == 1 && merged.replaced_batches == 1 &&
                problem.NumResiduals() == 6 && problem.NumResidualBlocks() == 1,
            "Support-changing fallback did not merge correctly");
    batcher.StageWindowCutoff(1);
    channel.AddObservation(MakeBinding(&x, 1, 5.0), ScalarObservation{4.0}, 4);
    const auto maintained = batcher.Flush();
    Require(maintained.replaced_batches == 1 && problem.NumResiduals() == 5,
            "Window expiry and append were not committed together");
    Require(maintained.replacement_reasons[0] == 1 && maintained.replacement_reasons[1] == 1,
            "A joint expiry/append replacement lost one of its reason bits");
  }

  void TestNewBatchAdoptsPreparedStorage()
  {
    double x = 0;
    ceres::Problem problem;
    splbatch::CeresIncrementalBatchBackend backend(problem);
    splbatch::IncrementalProblemBatcher batcher(backend);
    auto &channel = batcher.RegisterEvaluator(PatchScalarEvaluator{});
    const auto handle = channel.Bind(MakeBinding(&x, 1));
    for (int i = 1; i <= 3; ++i)
      channel.AddObservation(handle, ScalarObservation{double(i)}, {i});
    CopyTrackedPrepared::copies = 0;
    Require(batcher.Flush().added_batches == 1 && CopyTrackedPrepared::copies == 0,
            "A new contiguous batch copied prepared payloads");
    Require(problem.NumResiduals() == 3 && EvaluateCost(problem) == 7,
            "Adopted storage changed initial residuals");

    // Appending after adoption must NOT grow the vector seen by Ceres before
    // its declared row count is replaced in Flush (also a bounds-safety check).
    channel.AddObservation(handle, ScalarObservation{4}, {4});
    Require(problem.NumResiduals() == 3 && EvaluateCost(problem) == 7,
            "Staged append mutated an already registered adopted vector");
    Require(batcher.Flush().replaced_batches == 1 && problem.NumResiduals() == 4 &&
                EvaluateCost(problem) == 15,
            "Adopted batch lost the next append");

    // A partial expiry before the first commit leaves holes in staged storage;
    // this must compact rather than adopting the wrong row order/count.
    double y = 0;
    const auto other = channel.Bind(MakeBinding(&y, 2));
    for (int i = 1; i <= 3; ++i)
      channel.AddObservation(other, ScalarObservation{double(i)}, {i});
    batcher.StageWindowCutoff(1);
    batcher.Flush();
    Require(problem.NumResiduals() == 5 && EvaluateCost(problem) == 21,
            "New-batch adoption failed to compact expired slots");
    // The caller can retain an ingestion handle across commits.
    channel.AddObservation(other, ScalarObservation{4}, {4});
    Require(problem.NumResiduals() == 5 && EvaluateCost(problem) == 21,
            "Append after compacted publication changed live residuals");
    batcher.Flush();
    Require(problem.NumResiduals() == 6 && EvaluateCost(problem) == 29,
            "Append after compacted publication lost data");
  }

  void TestRefreshPatchesAndDependencyIdentity()
  {
    double x = 0, y = 0;
    ceres::Problem problem;
    splbatch::CeresIncrementalBatchBackend backend(problem);
    splbatch::IncrementalProblemBatcher batcher(backend);
    auto fail = std::make_shared<bool>(false);
    auto &channel = batcher.RegisterEvaluator(PatchScalarEvaluator{fail});
    using Channel = std::decay_t<decltype(channel)>;
    auto a = std::make_shared<splbatch::IncrementalSupportDependency>();
    auto b = std::make_shared<splbatch::IncrementalSupportDependency>();
    auto c = std::make_shared<splbatch::IncrementalSupportDependency>();
    std::size_t calls = 0;
    bool fail_dependency = false;
    channel.SetDependencyResolver([&](const auto &binding) -> Channel::Dependencies
    {
      ++calls;
      if (fail_dependency) throw std::runtime_error("Injected dependency resolution failure");
      if (binding.context.preparation_offset >= 100) return {c};
      if (binding.context.preparation_offset >= 10) return {b, b}; // duplicates are legal
      return {a};
    });
    for (int i = 1; i <= 12; ++i)
      channel.AddObservation(MakeBinding(&x, 1), ScalarObservation{double(i)}, i);
    batcher.Flush();
    std::vector<ceres::ResidualBlockId> original;
    problem.GetResidualBlocks(&original);
    const auto all = [](const auto &) { return true; };
    const auto identity = [](auto, const auto &, const auto &binding, const auto &) { return binding; };
    const auto refresh = [&](double offset)
    {
      return channel.RebindInvalidatedBatches(all, identity, 0,
          [&](const auto &, auto &context) { context.preparation_offset = offset; return true; });
    };
    calls = 0;
    CopyTrackedPrepared::copies = 0;
    refresh(3);
    auto summary = channel.TakeRefreshSummary();
    Require(calls == 1 && summary.dependency_sets_checked == 1 &&
                summary.dependency_edges_added == 0 && summary.dependency_edges_removed == 0,
            "Coefficient refresh rebuilt unchanged dependency edges / resolved per row");
    Require(CopyTrackedPrepared::copies == 0, "Partial refresh copied immutable prepared payloads");
    const double cost = EvaluateCost(problem);
    for (bool dependency_failure : {false, true})
    {
      batcher.InvalidateDependency(a.get());
      *fail = !dependency_failure;
      fail_dependency = dependency_failure;
      bool threw = false;
      try { refresh(10); } catch (const std::runtime_error &) { threw = true; }
      Require(threw && EvaluateCost(problem) == cost, "Failed patch preparation published live rows");
      *fail = false; fail_dependency = false;
    }
    channel.TakeRefreshSummary();
    Require(refresh(10) == 12, "A failed patch refresh lost its pending invalidation");
    summary = channel.TakeRefreshSummary();
    Require(summary.dependency_edges_added == 1 && summary.dependency_edges_removed == 1,
            "Changed metadata identity with unchanged parameters did not update edges exactly once");
    batcher.InvalidateDependency(a.get());
    Require(refresh(10) == 0, "Retired dependency still invalidates a refreshed batch");
    batcher.InvalidateDependency(b.get());
    Require(refresh(10) == 12, "New dependency does not invalidate a refreshed batch");
    Require(batcher.Flush().replaced_batches == 0, "Partial refresh replaced Ceres residuals");
    std::vector<ceres::ResidualBlockId> current;
    problem.GetResidualBlocks(&current);
    Require(current == original && CopyTrackedPrepared::copies == 0,
            "Patch refresh changed the registered layout or copied prepared data");

    // Dirty rows span registered and appended storage. A simultaneous slow
    // rebind merges a different dependency into the fast-refreshed batch.
    channel.AddObservation(MakeBinding(&x, 1, 10), ScalarObservation{13}, 13);
    channel.AddObservation(MakeBinding(&y, 2), ScalarObservation{1}, 14);
    channel.RebindAffectedBatches(all,
        [&](auto, const auto &, const auto &, const auto &) { return MakeBinding(&x, 1, 100); }, 0,
        [&](const auto &view, auto &context)
        {
          context.preparation_offset = 10;
          return view.binding.parameters.blocks.front() == &x;
        });
    batcher.Flush();
    double expected = 0.5 * 101 * 101;
    for (int i = 1; i <= 13; ++i) expected += 0.5 * (i + 10) * (i + 10);
    Require(EvaluateCost(problem) == expected && problem.NumResidualBlocks() == 1,
            "Dirty patch publication overwrote/lost merged or appended rows");
    channel.RebindInvalidatedBatches([](const auto &) { return false; }, identity, 0);
    channel.TakeRefreshSummary();
    for (const auto &dependency : {b, c})
    {
      batcher.InvalidateDependency(dependency.get());
      channel.RebindInvalidatedBatches([](const auto &) { return false; }, identity, 0);
      Require(channel.TakeRefreshSummary().visited_batches == 1,
              "Fast refresh + slow merge dropped a dependency generation");
    }
    batcher.StageWindowCutoff(13); // only dependency c remains
    batcher.Flush();
    batcher.InvalidateDependency(b.get());
    channel.RebindInvalidatedBatches(all, identity, 0);
    Require(channel.TakeRefreshSummary().visited_batches == 0, "Expiry retained retired dependency edges");
  }

  void TestIndexedInvalidationAndSealedAppend()
  {
    for (bool sealed : {false, true})
    {
      double x = 0, y = 0;
      ceres::Problem problem;
      splbatch::CeresIncrementalBatchBackend backend(problem);
      backend.SetTimingEnabled(true);
      splbatch::IncrementalProblemBatcher batcher(backend);
      batcher.SealAfterFlush(sealed);
      auto fail = std::make_shared<bool>(false);
      auto &channel = batcher.RegisterEvaluator(ScalarEvaluator{nullptr, nullptr, fail});
      using Channel = std::decay_t<decltype(channel)>;
      auto dx = std::make_shared<splbatch::IncrementalSupportDependency>();
      auto dy = std::make_shared<splbatch::IncrementalSupportDependency>();
      channel.SetDependencyResolver([&](const auto &binding) -> Channel::Dependencies
      { return {binding.parameters.blocks[0] == &x ? dx : dy}; });
      channel.AddObservation(MakeBinding(&x, 1), ScalarObservation{1}, 1);
      channel.AddObservation(MakeBinding(&y, 2), ScalarObservation{2}, 1);
      batcher.Flush();
      std::vector<ceres::ResidualBlockId> original;
      problem.GetResidualBlocks(&original);
      const auto no_change = [](const auto &) { return false; };
      const auto identity = [](auto, const auto &, const auto &binding, const auto &) { return binding; };
      channel.RebindInvalidatedBatches(no_change, identity, 0);
      Require(channel.TakeRefreshSummary().visited_batches == 2, "New supports were not validated");
      channel.RebindInvalidatedBatches(no_change, identity, 0);
      Require(channel.TakeRefreshSummary().visited_batches == 0, "Unchanged refresh scanned all batches");

      batcher.InvalidateDependency(dy.get());
      *fail = true;
      bool threw = false;
      try { channel.RebindInvalidatedBatches([](const auto &) { return true; }, identity, 0); }
      catch (const std::runtime_error &) { threw = true; }
      Require(threw && std::abs(EvaluateCost(problem) - 2.5) < 1e-12, "Failed indexed refresh changed the live problem");
      *fail = false;
      Require(channel.RebindInvalidatedBatches([](const auto &) { return true; }, identity, 0) == 1,
              "Failed refresh lost its invalidation");
      channel.TakeRefreshSummary();

      batcher.InvalidateDependency(dx.get());
      channel.RebindInvalidatedBatches([](const auto &) { return true; }, identity, 0,
          [](const auto &, auto &context) { context.preparation_offset = 3; return true; });
      Require(channel.TakeRefreshSummary().visited_batches == 1, "Dependency invalidated an unrelated support");
      Require(batcher.Flush().replaced_batches == 0, "Numeric refresh replaced a residual");
      channel.AddObservation(MakeBinding(&x, 1, 3), ScalarObservation{3}, 2);
      const auto append = batcher.Flush();
      Require(append.built_rows == (sealed ? 1 : 2), "Append rebuilt stable rows unexpectedly");
      Require(append.added_batches == (sealed ? 1 : 0) && append.replaced_batches == (sealed ? 0 : 1),
              "Sealed append has incorrect registration counts");
      Require(append.replacement_reasons[0] == (sealed ? 0 : 1), "Append reason was not recorded");
      Require(backend.GetCeresTiming().adds == 1 && backend.GetCeresTiming().removes == (sealed ? 0 : 1),
              "Ceres timing counters do not match physical operations");
      if (sealed)
      {
        std::vector<ceres::ResidualBlockId> current;
        problem.GetResidualBlocks(&current);
        for (auto id : original) Require(std::find(current.begin(), current.end(), id) != current.end(),
                                       "Sealed append changed an old residual id");
      }
      double cost = 0; std::vector<double> g; ceres::CRSMatrix J;
      ceres::Problem::EvaluateOptions options; options.parameter_blocks = {&x, &y};
      Require(problem.Evaluate(options, &cost, nullptr, &g, &J), "Sealed evaluation failed");
      Require(std::abs(cost - 28) < 1e-12 && g.size() == 2 && std::abs(g[0] + 10) < 1e-12 &&
                  std::abs(g[1] + 2) < 1e-12, "Sealing changed cost/gradient");
      double h[2]{};
      for (std::size_t k = 0; k < J.values.size(); ++k) h[J.cols[k]] += J.values[k] * J.values[k];
      Require(h[0] == 2 && h[1] == 1, "Sealing changed the normal matrix");
      batcher.StageWindowCutoff(1);
      const auto expiry = batcher.Flush();
      Require(problem.NumResiduals() == 1 && expiry.replacement_reasons[1] == (sealed ? 0 : 1),
              "Window expiry or its replacement reason is incorrect");
      batcher.Clear();
      batcher.InvalidateDependency(dx.get());
      channel.RebindInvalidatedBatches(no_change, identity, 0);
      Require(channel.TakeRefreshSummary().visited_batches == 0, "Clear retained dependency edges");
    }
  }

} // namespace

int main()
{
  try
  {
    TestIndexedWindowCutoff();
    TestCompressedRegistration();
    TestEachObservationUsesItsOwnBindingForPreparation();
    TestCachedInsertionAndSharedStorage();
    TestSupportScopedRefresh();
    TestNewBatchAdoptsPreparedStorage();
    TestRefreshPatchesAndDependencyIdentity();
    TestIndexedInvalidationAndSealedAppend();
  }
  catch (const std::exception &error)
  {
    std::cerr << "Incremental batch check failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "PASS incremental lifecycle, refresh, dependency, expiry and compressed registration\n";
  return 0;
}
