#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <map>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <ceres/problem.h>

#include "splbatch/batch_binding.hpp"
#include "splbatch/compressed_segment_batch_cost_function.hpp"
#include "splbatch/evaluator_base.hpp"
#include "splbatch/loss_policy.hpp"
#include "splbatch/incremental_diagnostics.hpp"
#include "splbatch/segment_batch_cost_function.hpp"

namespace splbatch
{
  using IncrementalBatchId = std::uint64_t;
  using IncrementalBackendHandle = std::uint64_t;

  constexpr IncrementalBackendHandle kInvalidIncrementalBackendHandle = 0;

  /// Metadata needed by the incremental grouping layer. Application-specific
  /// information (robot ids, residual type, marginalization policy, ...) is
  /// deliberately represented by backend_tag + partition and interpreted by
  /// the selected backend rather than by the generic batch implementation.
  struct IncrementalObservationOptions
  {
    std::int64_t time_ns = 0;
    double loss_scale = 1.0;
    std::size_t partition = 0;

    /// Increment this only when evaluator semantics change in a way that is
    /// not captured by PrepareObservation(). Numerical spline coefficient
    /// refreshes with unchanged parameter support keep the same revision and
    /// update the registered batch's prepared slots in place.
    std::uint64_t binding_revision = 0;
  };

  struct IncrementalBatchKey
  {
    BatchKey binding_key;
    std::size_t partition = 0;
    std::uint64_t binding_revision = 0;

    bool operator==(const IncrementalBatchKey &other) const
    {
      return binding_key == other.binding_key &&
             partition == other.partition &&
             binding_revision == other.binding_revision;
    }
  };

  struct IncrementalBatchKeyHash
  {
    std::size_t operator()(const IncrementalBatchKey &key) const
    {
      std::size_t seed = BatchKeyHash{}(key.binding_key);
      const auto combine = [&seed](const std::size_t value)
      {
        seed ^= value + static_cast<std::size_t>(0x9e3779b9) +
                (seed << 6) + (seed >> 2);
      };
      combine(std::hash<std::size_t>{}(key.partition));
      combine(std::hash<std::uint64_t>{}(key.binding_revision));
      return seed;
    }
  };

  /// Move-only factor description handed to a registration backend. The
  /// backend becomes responsible for the CostFunction after Add/Replace.
  struct IncrementalBuiltBatch
  {
    IncrementalBatchId batch_id = 0;
    std::uint64_t backend_tag = 0;
    IncrementalBatchKey key;
    std::unique_ptr<ceres::CostFunction> cost_function;
    std::vector<double *> parameter_blocks;
    std::int64_t min_time_ns = 0;
    std::int64_t max_time_ns = 0;
    std::size_t observation_count = 0;

    IncrementalBuiltBatch() = default;
    IncrementalBuiltBatch(IncrementalBuiltBatch &&) = default;
    IncrementalBuiltBatch &operator=(IncrementalBuiltBatch &&) = default;
    IncrementalBuiltBatch(const IncrementalBuiltBatch &) = delete;
    IncrementalBuiltBatch &operator=(const IncrementalBuiltBatch &) = delete;
  };

  enum class IncrementalBatchRemovalReason
  {
    ObservationRemoved,
    WindowExpired,
    Repartitioned,
    Clear
  };

  struct IncrementalFlushSummary
  {
    std::size_t added_batches = 0;
    std::size_t replaced_batches = 0;
    std::size_t removed_batches = 0;
    std::size_t built_rows = 0;
    std::array<std::size_t, kBatchChangeCount> replacement_reasons{};
    double build_ms = 0, backend_add_ms = 0, backend_replace_ms = 0, backend_remove_ms = 0;
    double backend_finalize_ms = 0;

    IncrementalFlushSummary &operator+=(
        const IncrementalFlushSummary &other)
    {
      added_batches += other.added_batches;
      replaced_batches += other.replaced_batches;
      removed_batches += other.removed_batches;
      built_rows += other.built_rows;
      for (std::size_t i = 0; i < kBatchChangeCount; ++i)
        replacement_reasons[i] += other.replacement_reasons[i];
      build_ms += other.build_ms;
      backend_add_ms += other.backend_add_ms;
      backend_replace_ms += other.backend_replace_ms;
      backend_remove_ms += other.backend_remove_ms;
      backend_finalize_ms += other.backend_finalize_ms;
      return *this;
    }
  };

  /// Registration boundary used by both plain Ceres and application-specific
  /// persistent-problem adapters. Handles stay stable across Replace(), even
  /// though the underlying ceres::ResidualBlockId normally changes.
  class IncrementalBatchBackend
  {
  public:
    virtual ~IncrementalBatchBackend() = default;
    struct CeresTiming
    {
      double add_ms = 0, remove_ms = 0;
      std::size_t adds = 0, removes = 0;
    };
    void SetTimingEnabled(bool enabled) { timing_enabled_ = enabled; }
    bool TimingEnabled() const { return timing_enabled_; }
    void ResetCeresTiming() { ceres_timing_ = {}; }
    const CeresTiming &GetCeresTiming() const { return ceres_timing_; }

    // Optional transaction boundaries for batching application bookkeeping.
    // They do not permit concurrent Evaluate/Solve or change Ceres ownership.
    virtual void BeginCommit() {}
    virtual void EndCommit() {}

    virtual IncrementalBackendHandle Add(
        IncrementalBuiltBatch batch) = 0;

    virtual void Replace(
        IncrementalBackendHandle handle,
        IncrementalBuiltBatch batch) = 0;

    virtual void Remove(
        IncrementalBackendHandle handle,
        IncrementalBatchRemovalReason reason) = 0;
  protected:
    template <typename F> auto CeresAdd(F &&f)
    {
      IncrementalScopedTimer timer(timing_enabled_ ? &ceres_timing_.add_ms : nullptr);
      ++ceres_timing_.adds;
      return f();
    }
    template <typename F> void CeresRemove(F &&f)
    {
      IncrementalScopedTimer timer(timing_enabled_ ? &ceres_timing_.remove_ms : nullptr);
      ++ceres_timing_.removes;
      f();
    }
  private:
    bool timing_enabled_ = false;
    CeresTiming ceres_timing_;
  };

  /// Minimal backend for users that only have a persistent ceres::Problem.
  /// The Problem must use TAKE_OWNERSHIP for cost functions (Ceres' default).
  class CeresIncrementalBatchBackend final : public IncrementalBatchBackend
  {
  public:
    explicit CeresIncrementalBatchBackend(ceres::Problem &problem)
        : problem_(&problem)
    {
    }

    IncrementalBackendHandle Add(IncrementalBuiltBatch batch) override
    {
      Validate(batch);
      const IncrementalBackendHandle handle = AllocateHandle();
      const ceres::ResidualBlockId residual = AddToProblem(batch);
      residuals_.emplace(handle, residual);
      return handle;
    }

    void Replace(IncrementalBackendHandle handle,
                 IncrementalBuiltBatch batch) override
    {
      Validate(batch);
      const auto found = residuals_.find(handle);
      if (found == residuals_.end())
        throw std::out_of_range("Unknown incremental backend handle");

      CeresRemove([&] { problem_->RemoveResidualBlock(found->second); });
      found->second = AddToProblem(batch);
    }

    void Remove(IncrementalBackendHandle handle,
                IncrementalBatchRemovalReason) override
    {
      const auto found = residuals_.find(handle);
      if (found == residuals_.end())
        throw std::out_of_range("Unknown incremental backend handle");
      CeresRemove([&] { problem_->RemoveResidualBlock(found->second); });
      residuals_.erase(found);
    }

    bool Contains(const IncrementalBackendHandle handle) const
    {
      return residuals_.count(handle) != 0;
    }

    std::size_t RegisteredBatchCount() const
    {
      return residuals_.size();
    }

    ceres::ResidualBlockId ResidualBlock(
        const IncrementalBackendHandle handle) const
    {
      const auto found = residuals_.find(handle);
      if (found == residuals_.end())
        throw std::out_of_range("Unknown incremental backend handle");
      return found->second;
    }

  private:
    static void Validate(const IncrementalBuiltBatch &batch)
    {
      if (!batch.cost_function)
        throw std::invalid_argument("Incremental batch cost is null");
      if (batch.observation_count == 0)
        throw std::invalid_argument("Incremental batch is empty");
      if (batch.parameter_blocks.size() !=
          batch.cost_function->parameter_block_sizes().size())
      {
        throw std::invalid_argument(
            "Incremental batch parameter-block count is inconsistent");
      }
    }

    IncrementalBackendHandle AllocateHandle()
    {
      if (next_handle_ == kInvalidIncrementalBackendHandle)
        throw std::overflow_error("Incremental backend handle overflow");
      return next_handle_++;
    }

    ceres::ResidualBlockId AddToProblem(IncrementalBuiltBatch &batch)
    {
      ceres::CostFunction *const cost = batch.cost_function.get();
      const ceres::ResidualBlockId residual =
          CeresAdd([&] { return problem_->AddResidualBlock(
              cost, nullptr, batch.parameter_blocks); });
      batch.cost_function.release();
      return residual;
    }

    ceres::Problem *problem_ = nullptr;
    IncrementalBackendHandle next_handle_ = 1;
    std::unordered_map<IncrementalBackendHandle, ceres::ResidualBlockId>
        residuals_;
  };

  class IncrementalChannelBase
  {
  public:
    virtual ~IncrementalChannelBase() = default;

    virtual IncrementalFlushSummary Flush(
        IncrementalBatchBackend &backend) = 0;
    virtual std::size_t StageWindowCutoff(std::int64_t cutoff_ns) = 0;
    virtual void Clear(IncrementalBatchBackend &backend) = 0;
    virtual std::size_t ObservationCount() const = 0;
    virtual std::size_t BatchCount() const = 0;
    virtual std::size_t RegisteredBatchCount() const = 0;
    virtual std::size_t DirtyBatchCount() const = 0;
    virtual void InvalidateDependency(const IncrementalSupportDependency *dependency) = 0;
    virtual void SealAfterFlush(bool enabled) = 0;
    virtual IncrementalRefreshSummary TakeRefreshSummary() = 0;
  };

  template <typename Evaluator, typename LossPolicy = NoLossPolicy,
            template <typename, typename> class CostFunctionTemplate =
                SegmentBatchCostFunction>
  class IncrementalEvaluatorChannel final : public IncrementalChannelBase
  {
    static_assert(
        kIsEvaluator<Evaluator>,
        "Evaluator must derive from splbatch::EvaluatorBase");

  public:
    using Observation = typename Evaluator::Observation;
    using Binding = typename Evaluator::Binding;
    using SharedBinding = std::shared_ptr<const Binding>;
    using PreparedObservation = typename Evaluator::PreparedObservation;
    using PreparedObservationStorage =
        std::vector<PreparedObservation>;
    using SharedPreparedObservationStorage =
        std::shared_ptr<PreparedObservationStorage>;
    using CostFunction = CostFunctionTemplate<Evaluator, LossPolicy>;
    using ObservationId = std::uint64_t;
    using Dependencies = std::vector<std::shared_ptr<const IncrementalSupportDependency>>;
    // Dependencies are a property of an immutable binding generation. Values
    // derived from individual observations must be tracked separately.
    using DependencyResolver = std::function<Dependencies(const Binding &)>;

    void SetDependencyResolver(DependencyResolver resolver)
    {
      if (!observations_.empty()) throw std::logic_error("Install dependencies before ingestion");
      dependency_resolver_ = std::move(resolver);
    }
    void InvalidateDependency(const IncrementalSupportDependency *dependency) override
    {
      const auto found = dependency_batches_.find(dependency);
      if (found != dependency_batches_.end())
        invalidated_batches_.insert(found->second.begin(), found->second.end());
    }
    void SealAfterFlush(bool enabled) override { seal_after_flush_ = enabled; }
    IncrementalRefreshSummary TakeRefreshSummary() override
    { auto result = refresh_summary_; refresh_summary_ = {}; return result; }

    class BindingHandle
    {
      friend class IncrementalEvaluatorChannel;
      SharedBinding binding_;
      BatchKey key_;
    public:
      const Binding &binding() const { return *binding_; }
    };
    using SharedBindingHandle = std::shared_ptr<const BindingHandle>;

    SharedBindingHandle Bind(Binding binding) const
    {
      auto handle = std::make_shared<BindingHandle>();
      handle->key_ = binding.Key();
      handle->binding_ = std::make_shared<Binding>(std::move(binding));
      return handle;
    }

    // A bounded per-stream ingestion cache. The application explicitly checks
    // numeric semantics AND parameter lifetime/support; equal keys alone never
    // intern different preparation contexts. Flush releases all cached handles.
    template <typename Compatible, typename Factory>
    ObservationId AddWithBindingCache(Observation observation,
        IncrementalObservationOptions options, Compatible &&compatible,
        Factory &&factory)
    {
      ValidateObservationLossScale(options.loss_scale);
      auto &cached = ingestion_bindings_[options.partition];
      if (!cached.handle || cached.revision != options.binding_revision ||
          !compatible(cached.handle->binding()))
      {
        cached.handle = Bind(factory());
        cached.revision = options.binding_revision;
        cached.batch = 0;
      }
      return AddResolvedObservation(cached.handle, std::move(observation), options, cached.batch);
    }

    struct BatchView
    {
      IncrementalBatchId batch_id;
      std::uint64_t backend_tag;
      const Binding &binding;
      const BatchKey &binding_key;
      std::size_t partition;
      std::uint64_t binding_revision;
      std::int64_t min_time_ns;
      std::int64_t max_time_ns;
      std::size_t observation_count;
      bool registered;
      const Observation &representative_observation;
    };

    /// Full observation refresh used when an external spline/cache update
    /// changes not only the binding but also observation-side derived values
    /// (for example ABCD's synchronized self-IMU samples).
    struct ObservationUpdate
    {
      Observation observation;
      Binding binding;
      IncrementalObservationOptions options;
    };

    IncrementalEvaluatorChannel(
        Evaluator evaluator, LossPolicy loss_policy,
        const std::uint64_t backend_tag,
        IncrementalBatchId *next_batch_id)
        : evaluator_(
              std::make_shared<const Evaluator>(std::move(evaluator))),
          loss_policy_(std::move(loss_policy)),
          backend_tag_(backend_tag), next_batch_id_(next_batch_id)
    {
      if (next_batch_id_ == nullptr)
        throw std::invalid_argument("Incremental batch-id allocator is null");
    }

    ObservationId AddObservation(
        Binding binding, Observation observation,
        IncrementalObservationOptions options = {})
    {
      return AddObservation(Bind(std::move(binding)), std::move(observation), options);
    }

    ObservationId AddObservation(SharedBindingHandle handle, Observation observation,
                                 IncrementalObservationOptions options = {})
    {
      if (!handle) throw std::invalid_argument("Null incremental binding handle");
      ValidateObservationLossScale(options.loss_scale);
      IncrementalBatchId batch_id = 0;
      return AddResolvedObservation(handle, std::move(observation), options, batch_id);
    }

  private:
    ObservationId AddResolvedObservation(const SharedBindingHandle &handle, Observation observation,
        IncrementalObservationOptions options, IncrementalBatchId &batch_id)
    {
      ValidateObservationLossScale(options.loss_scale);
      // Preparation is observation-local.  In particular, two observations
      // can legitimately reference the same parameter blocks while having
      // been created from different (in-place updated) spline metadata.  Do
      // this once, against the observation's own binding, instead of applying
      // the first observation's binding to the whole batch during every
      // Flush().
      PreparedObservation prepared =
          evaluator_->PrepareObservation(handle->binding(), observation);
      // Invalid observations must not leave empty, never-dirty batches in the
      // support index. Resolve/cache the batch only after preparation succeeds.
      if (batch_id == 0)
        batch_id = GetOrCreateBatch(
            {handle->key_, options.partition, options.binding_revision});
      const ObservationId observation_id = AllocateObservationId();

      LiveBatch &batch = batches_.at(batch_id);
      auto slot = AppendPrepared(batch, std::move(prepared));

      ObservationEntry entry(
          std::move(observation), handle->binding_,
          slot.first, slot.second, options, batch_id, observation_id);
      const auto inserted = observations_.emplace(
          observation_id, std::move(entry));
      if (!inserted.second)
        throw std::logic_error("Duplicate incremental observation id");

      batch.observations.push_back(observation_id);
      AttachDependencies(batch, *handle->binding_);
      if (dependency_resolver_) invalidated_batches_.insert(batch.id);
      ExtendTimeRange(batch, options.time_ns);
      MarkDirty(batch, IncrementalBatchRemovalReason::ObservationRemoved, ChangeMask(BatchChange::Append));
      return observation_id;
    }

  public:
    ObservationId AddObservation(
        Binding binding, Observation observation,
        const std::int64_t time_ns, const double loss_scale = 1.0,
        const std::size_t partition = 0,
        const std::uint64_t binding_revision = 0)
    {
      return AddObservation(
          std::move(binding), std::move(observation),
          IncrementalObservationOptions{
              time_ns, loss_scale, partition, binding_revision});
    }

    bool RemoveObservation(const ObservationId observation_id)
    {
      return RemoveObservationImpl(
          observation_id,
          IncrementalBatchRemovalReason::ObservationRemoved);
    }

    /// Visit observations that the next StageWindowCutoff(cutoff_ns) will
    /// retire. Applications can use this immediately before staging to carry
    /// factor-specific information into a fixed-lag prior. The callback is
    /// invoked as
    ///
    ///   callback(observation_id, observation, binding, options)
    ///
    /// and must not mutate this channel.
    template <typename Callback>
    std::size_t VisitObservationsAtOrBefore(
        const std::int64_t cutoff_ns, Callback &&callback) const
    {
      std::vector<ObservationId> selected;
      selected.reserve(observations_.size());
      for (const auto &item : observations_)
      {
        if (item.second.options.time_ns <= cutoff_ns)
          selected.push_back(item.first);
      }
      std::sort(selected.begin(), selected.end());
      for (const ObservationId observation_id : selected)
      {
        const ObservationEntry &entry = observations_.at(observation_id);
        callback(
            observation_id, entry.observation, *entry.binding, entry.options);
      }
      return selected.size();
    }

    std::size_t StageWindowCutoff(const std::int64_t cutoff_ns) override
    {
      std::size_t expired_count = 0;
      last_window_visited_batches_ = 0;
      // The minimum-time heap skips all supports wholly newer than the cutoff.
      // Entries contain IDs, never pointers: arbitrary removal/repartitioning
      // can leave stale heap entries, which are validated lazily when popped.
      while (!expiry_index_.empty() && expiry_index_.front().first <= cutoff_ns)
      {
        std::pop_heap(expiry_index_.begin(), expiry_index_.end(), std::greater<ExpiryEntry>{});
        const auto candidate = expiry_index_.back();
        expiry_index_.pop_back();
        const auto found = batches_.find(candidate.second);
        if (found == batches_.end() || found->second.observations.empty() ||
            found->second.min_time_ns != candidate.first)
          continue;
        LiveBatch &batch = found->second;
        ++last_window_visited_batches_;
        if (batch.max_time_ns <= cutoff_ns)
        {
          expired_count += batch.observations.size();
          for (const ObservationId observation_id : batch.observations)
          {
            if (observations_.erase(observation_id) != 1)
            {
              throw std::logic_error(
                  "Incremental batch references a missing observation");
            }
          }
          batch.observations.clear();
          UnindexDependencies(batch);
          RecomputeTimeRange(batch);
          MarkDirty(batch, IncrementalBatchRemovalReason::WindowExpired);
          continue;
        }

        std::size_t retained = 0;
        const auto previous_size = batch.observations.size();
        for (const ObservationId observation_id : batch.observations)
        {
          if (observations_.at(observation_id).options.time_ns <= cutoff_ns)
          {
            observations_.erase(observation_id);
            ++expired_count;
          }
          else
          {
            batch.observations[retained++] = observation_id;
          }
        }
        if (retained != previous_size)
        {
          batch.observations.resize(retained);
          ReindexDependencies(batch);
          RecomputeTimeRange(batch);
          MarkDirty(batch, IncrementalBatchRemovalReason::WindowExpired);
        }
      }
      return expired_count;
    }

    // Diagnostic count excludes stale heap entries. Useful for checking that
    // an unchanged cutoff does not revisit every occupied support.
    std::size_t LastWindowVisitedBatches() const { return last_window_visited_batches_; }

    /// Re-resolve only observations belonging to batches selected by
    /// batch_predicate. resolver is called as
    ///
    ///   Binding resolver(observation_id, observation, old_binding, options)
    ///
    /// All resolutions are completed before membership is modified, so a
    /// resolver exception leaves the current grouping untouched.
    template <typename BatchPredicate, typename BindingResolver>
    std::size_t RebindAffectedBatches(
        BatchPredicate &&batch_predicate,
        BindingResolver &&resolver,
        const std::uint64_t new_binding_revision)
    {
      return RebindAffectedBatches(
          std::forward<BatchPredicate>(batch_predicate),
          std::forward<BindingResolver>(resolver), new_binding_revision,
          [](const BatchView &, auto &) { return false; });
    }

    /// Optional support-preserving refresh. Called once per affected batch:
    ///
    ///   bool refresh_context(const BatchView &, Binding::context &)
    ///
    /// Return true ONLY if the existing ordered parameter layout and runtime
    /// interpretation remain valid for every observation in the batch. Only
    /// observation-time preparation metadata may change. No Key construction,
    /// parameter lookup, or per-observation binding copy is needed on this path.
    /// False falls back to the ordinary per-observation resolver (split/merge).
    /// Contexts from different numeric generations are NOT implicitly interned
    /// merely because their structural keys match. The caller opts in here.
    template <typename BatchPredicate, typename BindingResolver,
              typename SupportRefresher>
    std::size_t RebindAffectedBatches(
        BatchPredicate &&batch_predicate,
        BindingResolver &&resolver,
        const std::uint64_t new_binding_revision,
        SupportRefresher &&refresh_context,
        const std::vector<IncrementalBatchId> *selected_batches = nullptr)
    {
      refresh_batches_.clear();
      resolved_refresh_.clear();
      std::size_t affected = 0;
      // Walk supports directly. Stable batches cache row pointers (unordered_map
      // rehash does not invalidate them), so numeric refresh never sorts IDs or
      // performs one hash lookup per row.
      const auto prepare_batch = [&](LiveBatch &batch)
      {
        if (batch.observations.empty()) return;
        ++refresh_summary_.visited_batches;
        const ObservationEntry &representative =
            observations_.at(batch.observations.front());
        const BatchView view{
            batch.id, backend_tag_, *representative.binding,
            batch.key.binding_key, batch.key.partition,
            batch.key.binding_revision, batch.min_time_ns,
            batch.max_time_ns, batch.observations.size(),
            batch.backend_handle.has_value(), representative.observation};
        if (!batch_predicate(view)) return;
        ++refresh_summary_.affected_batches;
        EnsureEntries(batch);
        affected += batch.entries.size();
        bool fast = false;
        if (batch.key.binding_revision == new_binding_revision)
        {
          // Recycle a numeric-context buffer only when no live row/factor or
          // external handle retains it. Never modify an immutable generation.
          if (!batch.refresh_binding || batch.refresh_binding.use_count() != 1)
            batch.refresh_binding = std::make_shared<Binding>(*representative.binding);
          else
            *batch.refresh_binding = *representative.binding;
          fast = refresh_context(view, batch.refresh_binding->context);
        }
        if (fast)
        {
          // Resolve once per binding, before publishing ANY prepared data.
          // Parameter-key equality alone does not imply dependency identity:
          // metadata can move while the Ceres parameter addresses stay fixed.
          if (dependency_resolver_)
          {
            batch.refresh_dependencies = dependency_resolver_(*batch.refresh_binding);
            for (const auto &dependency : batch.refresh_dependencies)
              if (!dependency) throw std::invalid_argument("Null support dependency");
          }
          refresh_summary_.coefficient_rows += batch.entries.size();
          batch.refresh_prepared.clear();
          batch.refresh_prepared.reserve(batch.entries.size());
          for (const ObservationEntry *entry : batch.entries)
          {
            batch.refresh_prepared.push_back(RefreshTraits::Prepare(*evaluator_,
                *entry->binding, *batch.refresh_binding, entry->observation, entry->Prepared()));
          }
          refresh_batches_.push_back(&batch);
          return;
        }
        refresh_summary_.resolved_rows += batch.entries.size();
        for (const ObservationEntry *entry : batch.entries)
        {
          Binding binding = resolver(entry->id, entry->observation,
                                     *entry->binding, entry->options);
          (void)binding.Key();
          auto prepared = evaluator_->PrepareObservation(binding, entry->observation);
          resolved_refresh_.emplace_back(entry->id, PreparedBindingUpdate{
              std::make_shared<Binding>(std::move(binding)), std::move(prepared)});
        }
      };
      if (selected_batches)
      {
        for (const auto id : *selected_batches)
        {
          const auto found = batches_.find(id);
          if (found != batches_.end()) prepare_batch(found->second);
        }
      }
      else for (auto &item : batches_) prepare_batch(item.second);

      // All preparation/resolution must succeed before any live slot changes.
      std::sort(resolved_refresh_.begin(), resolved_refresh_.end(),
                [](const auto &a, const auto &b) { return a.first < b.first; });
      ApplyBindingUpdates(resolved_refresh_, new_binding_revision);
      for (LiveBatch *batch : refresh_batches_)
      {
        bool swapped = false;
        if constexpr (!RefreshTraits::kPartial)
        {
          if (!batch->dirty && batch->registered)
          {
            batch->registered->swap(batch->refresh_prepared);
            swapped = true;
          }
        }
        for (std::size_t i = 0; i < batch->entries.size(); ++i)
        {
          auto &entry = *batch->entries[i];
          if (!swapped)
            RefreshTraits::Apply(*evaluator_, batch->refresh_prepared[i], entry.Prepared());
          entry.binding = batch->refresh_binding;
        }
        batch->previous_refresh_binding.swap(batch->refresh_binding);
        if (batch->entries_current)
          PublishRefreshDependencies(*batch);
        else
          // A slow-path split/merge may have appended rows from a different
          // binding since preparation. Preserve the union of ALL dependencies.
          ReindexDependencies(*batch);
      }
      return affected;
    }

    // Same predicate/resolver contract, but only newly staged or explicitly
    // invalidated batches are visited. Failed refreshes keep the pending set.
    template <typename Predicate, typename Resolver, typename Refresher>
    std::size_t RebindInvalidatedBatches(Predicate &&predicate, Resolver &&resolver,
        std::uint64_t revision, Refresher &&refresher)
    {
      if (!dependency_resolver_)
        return RebindAffectedBatches(std::forward<Predicate>(predicate),
            std::forward<Resolver>(resolver), revision, std::forward<Refresher>(refresher));
      std::vector<IncrementalBatchId> selected(invalidated_batches_.begin(), invalidated_batches_.end());
      std::sort(selected.begin(), selected.end());
      const auto count = RebindAffectedBatches(std::forward<Predicate>(predicate),
          std::forward<Resolver>(resolver), revision, std::forward<Refresher>(refresher), &selected);
      for (auto id : selected) invalidated_batches_.erase(id);
      return count;
    }
    template <typename Predicate, typename Resolver>
    std::size_t RebindInvalidatedBatches(Predicate &&predicate, Resolver &&resolver, std::uint64_t revision)
    {
      return RebindInvalidatedBatches(std::forward<Predicate>(predicate),
          std::forward<Resolver>(resolver), revision, [](const BatchView &, auto &) { return false; });
    }

    /// Refresh raw observations and bindings together. resolver is called as
    ///
    ///   ObservationUpdate resolver(
    ///       observation_id, observation, old_binding, old_options)
    ///
    /// This is the CT full/ABCD path for factors whose observation payload
    /// includes values sampled from another synchronized trajectory/spline.
    template <typename BatchPredicate, typename ObservationResolver>
    std::size_t RefreshAffectedBatches(
        BatchPredicate &&batch_predicate,
        ObservationResolver &&resolver)
    {
      const std::vector<ObservationId> affected =
          CollectAffectedObservations(
              std::forward<BatchPredicate>(batch_predicate));

      std::vector<std::pair<ObservationId, PreparedObservationUpdate>>
          resolved;
      resolved.reserve(affected.size());
      for (const ObservationId observation_id : affected)
      {
        const ObservationEntry &entry = observations_.at(observation_id);
        ObservationUpdate update = resolver(
            observation_id, entry.observation, *entry.binding,
            entry.options);
        ValidateObservationLossScale(update.options.loss_scale);
        (void)update.binding.Key();
        PreparedObservation prepared = evaluator_->PrepareObservation(
            update.binding, update.observation);
        resolved.emplace_back(
            observation_id,
            PreparedObservationUpdate{
                std::move(update), std::move(prepared)});
      }

      ApplyObservationUpdates(resolved);
      return resolved.size();
    }

    template <typename BindingResolver>
    std::size_t RebindAll(
        BindingResolver &&resolver,
        const std::uint64_t new_binding_revision)
    {
      return RebindAffectedBatches(
          [](const BatchView &) { return true; },
          std::forward<BindingResolver>(resolver),
          new_binding_revision);
    }

    IncrementalFlushSummary Flush(
        IncrementalBatchBackend &backend) override
    {
      ingestion_bindings_.clear();
      std::vector<IncrementalBatchId> dirty(
          dirty_batches_.begin(), dirty_batches_.end());
      std::sort(dirty.begin(), dirty.end());

      struct BuildOperation
      {
        IncrementalBatchId batch_id = 0;
        std::unique_ptr<IncrementalBuiltBatch> built;
      };
      std::vector<BuildOperation> operations;
      operations.reserve(dirty.size());
      IncrementalFlushSummary summary;

      // Build every replacement before mutating the backend. This catches
      // invalid bindings/preparation errors while the registered problem is
      // still intact.
      for (const IncrementalBatchId batch_id : dirty)
      {
        IncrementalScopedTimer timer(backend.TimingEnabled() ? &summary.build_ms : nullptr);
        const auto found = batches_.find(batch_id);
        if (found == batches_.end())
          continue;
        if (found->second.observations.empty())
          operations.push_back(BuildOperation{batch_id, nullptr});
        else
        {
          summary.built_rows += found->second.observations.size();
          operations.push_back(BuildOperation{
              batch_id,
              std::make_unique<IncrementalBuiltBatch>(
                  BuildBatch(found->second))});
        }
      }

      // Remove empty batches first. A batch that received their observations
      // can then be replaced without temporarily duplicating those rows.
      for (BuildOperation &operation : operations)
      {
        if (operation.built)
          continue;
        LiveBatch &batch = batches_.at(operation.batch_id);
        if (batch.backend_handle)
        {
          IncrementalScopedTimer timer(backend.TimingEnabled() ? &summary.backend_remove_ms : nullptr);
          backend.Remove(*batch.backend_handle, batch.empty_reason);
          ++summary.removed_batches;
        }
      }

      for (BuildOperation &operation : operations)
      {
        if (!operation.built)
          continue;
        LiveBatch &batch = batches_.at(operation.batch_id);
        if (batch.backend_handle)
        {
          IncrementalScopedTimer timer(backend.TimingEnabled() ? &summary.backend_replace_ms : nullptr);
          backend.Replace(
              *batch.backend_handle, std::move(*operation.built));
          ++summary.replaced_batches;
          for (std::size_t i = 0; i < kBatchChangeCount; ++i)
            if (batch.change_mask & (1u << i)) ++summary.replacement_reasons[i];
        }
        else
        {
          IncrementalScopedTimer timer(backend.TimingEnabled() ? &summary.backend_add_ms : nullptr);
          const IncrementalBackendHandle handle =
              backend.Add(std::move(*operation.built));
          if (handle == kInvalidIncrementalBackendHandle)
          {
            throw std::runtime_error(
                "Incremental backend returned an invalid handle");
          }
          batch.backend_handle = handle;
          ++summary.added_batches;
        }
        batch.dirty = false;
        batch.change_mask = 0;
        PublishPreparedStorage(batch);
        dirty_batches_.erase(batch.id);
      }

      for (const BuildOperation &operation : operations)
      {
        if (operation.built)
          continue;
        const auto found = batches_.find(operation.batch_id);
        if (found == batches_.end())
          continue;
        const auto indexed = batch_indices_.find(found->second.key);
        if (indexed != batch_indices_.end() &&
            indexed->second == operation.batch_id)
        {
          batch_indices_.erase(indexed);
        }
        dirty_batches_.erase(operation.batch_id);
        invalidated_batches_.erase(operation.batch_id);
        UnindexDependencies(found->second);
        batches_.erase(found);
      }
      CompactExpiryIndex();
      return summary;
    }

    void Clear(IncrementalBatchBackend &backend) override
    {
      std::vector<IncrementalBatchId> ids;
      ids.reserve(batches_.size());
      for (const auto &item : batches_)
        ids.push_back(item.first);
      std::sort(ids.begin(), ids.end());
      for (const IncrementalBatchId id : ids)
      {
        LiveBatch &batch = batches_.at(id);
        if (batch.backend_handle)
        {
          backend.Remove(
              *batch.backend_handle,
              IncrementalBatchRemovalReason::Clear);
        }
      }
      observations_.clear();
      batches_.clear();
      batch_indices_.clear();
      dirty_batches_.clear();
      ingestion_bindings_.clear();
      refresh_batches_.clear();
      resolved_refresh_.clear();
      dependency_batches_.clear();
      invalidated_batches_.clear();
      refresh_summary_ = {};
      expiry_index_.clear();
      last_window_visited_batches_ = 0;
    }

    std::size_t ObservationCount() const override
    {
      return observations_.size();
    }

    std::size_t BatchCount() const override
    {
      return batches_.size();
    }

    std::size_t RegisteredBatchCount() const override
    {
      std::size_t count = 0;
      for (const auto &item : batches_)
      {
        if (item.second.backend_handle)
          ++count;
      }
      return count;
    }

    std::size_t DirtyBatchCount() const override
    {
      return dirty_batches_.size();
    }

  private:
    struct PreparedBindingUpdate
    {
      SharedBinding binding;
      PreparedObservation prepared;
    };

    struct PreparedObservationUpdate
    {
      ObservationUpdate update;
      PreparedObservation prepared;
    };

    struct ObservationEntry
    {
      ObservationEntry(
          Observation observation_value, SharedBinding binding_value,
          SharedPreparedObservationStorage storage_value, std::size_t slot,
          const IncrementalObservationOptions &options_value,
          const IncrementalBatchId batch_id_value, ObservationId id_value)
          : observation(std::move(observation_value)),
            binding(std::move(binding_value)),
            storage(std::move(storage_value)), storage_index(slot), options(options_value),
            batch_id(batch_id_value), id(id_value)
      {
      }

      Observation observation;
      SharedBinding binding;
      // One authoritative payload, either staged or shared with a live factor.
      SharedPreparedObservationStorage storage;
      std::size_t storage_index = 0;
      PreparedObservation &Prepared() { return (*storage)[storage_index]; }
      const PreparedObservation &Prepared() const { return (*storage)[storage_index]; }
      IncrementalObservationOptions options;
      IncrementalBatchId batch_id = 0;
      ObservationId id = 0;
    };

    struct LiveBatch
    {
      LiveBatch(const IncrementalBatchId id_value,
                IncrementalBatchKey key_value)
          : id(id_value), key(std::move(key_value))
      {
      }

      IncrementalBatchId id = 0;
      IncrementalBatchKey key;
      std::vector<ObservationId> observations;
      std::vector<ObservationEntry *> entries;
      bool entries_current = false;
      Dependencies dependencies;
      const Binding *last_indexed_binding = nullptr;
      std::uint32_t change_mask = 0;
      SharedPreparedObservationStorage appended, registered, built;
      std::vector<typename detail::ObservationRefreshTraits<Evaluator>::Update> refresh_prepared;
      Dependencies refresh_dependencies;
      std::shared_ptr<Binding> refresh_binding, previous_refresh_binding;
      std::shared_ptr<const JacobianWritePlan> write_plan;
      std::int64_t min_time_ns = 0;
      std::int64_t max_time_ns = 0;
      std::optional<IncrementalBackendHandle> backend_handle;
      bool dirty = false;
      IncrementalBatchRemovalReason empty_reason =
          IncrementalBatchRemovalReason::ObservationRemoved;
    };

    using RefreshTraits = detail::ObservationRefreshTraits<Evaluator>;

    static IncrementalBatchKey MakeKey(
        const Binding &binding,
        const IncrementalObservationOptions &options)
    {
      return IncrementalBatchKey{
          binding.Key(), options.partition, options.binding_revision};
    }

    static std::uint32_t KeyChangeMask(const IncrementalBatchKey &old_key, const IncrementalBatchKey &new_key)
    {
      std::uint32_t reasons = 0;
      if (!(old_key.binding_key == new_key.binding_key)) reasons |= ChangeMask(BatchChange::Support);
      if (old_key.binding_revision != new_key.binding_revision) reasons |= ChangeMask(BatchChange::Semantics);
      if (old_key.partition != new_key.partition) reasons |= ChangeMask(BatchChange::ObservationMetadata);
      return reasons;
    }

    ObservationId AllocateObservationId()
    {
      if (next_observation_id_ == 0)
        throw std::overflow_error("Incremental observation-id overflow");
      return next_observation_id_++;
    }

    IncrementalBatchId AllocateBatchId()
    {
      if (*next_batch_id_ == 0)
        throw std::overflow_error("Incremental batch-id overflow");
      return (*next_batch_id_)++;
    }

    IncrementalBatchId GetOrCreateBatch(
        const IncrementalBatchKey &key)
    {
      const auto found = batch_indices_.find(key);
      if (found != batch_indices_.end() &&
          !(seal_after_flush_ && batches_.at(found->second).backend_handle))
        return found->second;

      const IncrementalBatchId id = AllocateBatchId();
      const auto inserted = batches_.emplace(id, LiveBatch{id, key});
      if (!inserted.second)
        throw std::logic_error("Duplicate incremental batch id");
      batch_indices_.insert_or_assign(key, id);
      return id;
    }

    void ExtendTimeRange(
        LiveBatch &batch, const std::int64_t time_ns)
    {
      if (batch.observations.size() == 1)
      {
        batch.min_time_ns = time_ns;
        batch.max_time_ns = time_ns;
        IndexExpiry(batch);
        return;
      }
      const auto previous_minimum = batch.min_time_ns;
      batch.min_time_ns = std::min(batch.min_time_ns, time_ns);
      batch.max_time_ns = std::max(batch.max_time_ns, time_ns);
      if (batch.min_time_ns != previous_minimum) IndexExpiry(batch);
    }

    void RecomputeTimeRange(LiveBatch &batch)
    {
      if (batch.observations.empty())
      {
        batch.min_time_ns = 0;
        batch.max_time_ns = 0;
        return;
      }
      std::int64_t minimum = std::numeric_limits<std::int64_t>::max();
      std::int64_t maximum = std::numeric_limits<std::int64_t>::min();
      for (const ObservationId observation_id : batch.observations)
      {
        const std::int64_t time_ns =
            observations_.at(observation_id).options.time_ns;
        minimum = std::min(minimum, time_ns);
        maximum = std::max(maximum, time_ns);
      }
      batch.min_time_ns = minimum;
      batch.max_time_ns = maximum;
      IndexExpiry(batch);
    }

    // Keep lazy deletion bounded even when callers remove/rebind out of order
    // or move the cutoff backwards. No monotonic-ingestion assumption is made.
    bool CompactExpiryIndex()
    {
      if (!batches_.empty() && expiry_index_.size() <= 2 * batches_.size() + 64)
        return false;
      expiry_index_.clear();
      for (const auto &[id, batch] : batches_)
        if (!batch.observations.empty()) expiry_index_.emplace_back(batch.min_time_ns, id);
      std::make_heap(expiry_index_.begin(), expiry_index_.end(), std::greater<ExpiryEntry>{});
      return true;
    }

    void IndexExpiry(const LiveBatch &batch)
    {
      // A rebuild already includes this batch's current minimum.
      if (CompactExpiryIndex()) return;
      expiry_index_.emplace_back(batch.min_time_ns, batch.id);
      std::push_heap(expiry_index_.begin(), expiry_index_.end(), std::greater<ExpiryEntry>{});
    }

    void MarkDirty(
        LiveBatch &batch,
        const IncrementalBatchRemovalReason empty_reason,
        std::uint32_t change_mask = 0)
    {
      batch.dirty = true;
      batch.entries_current = false;
      batch.empty_reason = empty_reason;
      if (change_mask == 0)
      {
        if (empty_reason == IncrementalBatchRemovalReason::WindowExpired)
          change_mask = ChangeMask(BatchChange::WindowExpired);
        else if (empty_reason == IncrementalBatchRemovalReason::ObservationRemoved)
          change_mask = ChangeMask(BatchChange::ObservationRemoved);
      }
      batch.change_mask |= change_mask;
      dirty_batches_.insert(batch.id);
    }

    static void RefreshPreparedInPlace(
        ObservationEntry &entry, PreparedObservation prepared)
    {
      entry.Prepared() = std::move(prepared);
    }

    bool RemoveObservationImpl(
        const ObservationId observation_id,
        const IncrementalBatchRemovalReason reason)
    {
      const auto found = observations_.find(observation_id);
      if (found == observations_.end())
        return false;

      LiveBatch &batch = batches_.at(found->second.batch_id);
      const auto position = std::find(
          batch.observations.begin(), batch.observations.end(),
          observation_id);
      if (position == batch.observations.end())
        throw std::logic_error("Incremental observation has no batch entry");
      batch.observations.erase(position);
      observations_.erase(found);
      ReindexDependencies(batch);
      RecomputeTimeRange(batch);
      MarkDirty(batch, reason);
      return true;
    }

    template <typename UpdateVector>
    std::unordered_set<IncrementalBatchId> DetachUpdatedObservations(
        const UpdateVector &updates)
    {
      std::unordered_set<ObservationId> affected;
      std::unordered_set<IncrementalBatchId> touched_batches;
      std::unordered_map<IncrementalBatchId, std::size_t> expected_per_batch;
      affected.reserve(updates.size());
      touched_batches.reserve(updates.size());
      expected_per_batch.reserve(updates.size());
      for (const auto &update : updates)
      {
        if (!affected.insert(update.first).second)
        {
          throw std::logic_error(
              "An incremental observation was updated more than once");
        }
        const IncrementalBatchId batch_id =
            observations_.at(update.first).batch_id;
        touched_batches.insert(batch_id);
        ++expected_per_batch[batch_id];
      }

      for (const IncrementalBatchId batch_id : touched_batches)
      {
        LiveBatch &batch = batches_.at(batch_id);
        const std::size_t old_size = batch.observations.size();
        batch.observations.erase(
            std::remove_if(
                batch.observations.begin(), batch.observations.end(),
                [&affected](const ObservationId observation_id)
                { return affected.count(observation_id) != 0; }),
            batch.observations.end());
        if (old_size - batch.observations.size() !=
            expected_per_batch.at(batch_id))
        {
          throw std::logic_error(
              "Incremental observation/batch membership is inconsistent");
        }
      }
      return touched_batches;
    }

    void FinishBulkUpdate(
        const std::unordered_set<IncrementalBatchId> &touched_batches)
    {
      for (const IncrementalBatchId batch_id : touched_batches)
      {
        LiveBatch &batch = batches_.at(batch_id);
        ReindexDependencies(batch);
        RecomputeTimeRange(batch);
        MarkDirty(
            batch, IncrementalBatchRemovalReason::Repartitioned);
      }
    }

    void ApplyBindingUpdates(
        std::vector<std::pair<ObservationId, PreparedBindingUpdate>> &updates,
        const std::uint64_t new_binding_revision)
    {
      std::vector<std::pair<ObservationId, PreparedBindingUpdate>>
          topology_updates;
      topology_updates.reserve(updates.size());
      std::unordered_set<IncrementalBatchId> numeric_updates;
      for (auto &update : updates)
      {
        ObservationEntry &entry = observations_.at(update.first);
        IncrementalObservationOptions new_options = entry.options;
        new_options.binding_revision = new_binding_revision;
        const IncrementalBatchKey new_key =
            MakeKey(*update.second.binding, new_options);
        if (new_key == batches_.at(entry.batch_id).key)
        {
          // CostFunction owns the same shared slot. Refresh only the
          // observation-time coefficients and captured semantic metadata;
          // Ceres' residual block and sparse layout remain untouched.
          entry.binding = std::move(update.second.binding);
          RefreshPreparedInPlace(
              entry, std::move(update.second.prepared));
          entry.options = new_options;
          if (dependency_resolver_) numeric_updates.insert(entry.batch_id);
          continue;
        }
        topology_updates.emplace_back(
            update.first, std::move(update.second));
      }
      for (auto id : numeric_updates) ReindexDependencies(batches_.at(id));

      if (topology_updates.empty())
        return;

      std::unordered_set<IncrementalBatchId> touched_batches =
          DetachUpdatedObservations(topology_updates);
      for (auto &update : topology_updates)
      {
        ObservationEntry &entry = observations_.at(update.first);
        const auto old_batch_id = entry.batch_id;
        entry.binding = std::move(update.second.binding);
        entry.options.binding_revision = new_binding_revision;
        const IncrementalBatchKey new_key =
            MakeKey(*entry.binding, entry.options);
        const auto reason = KeyChangeMask(batches_.at(old_batch_id).key, new_key);
        batches_.at(old_batch_id).change_mask |= reason;
        const IncrementalBatchId new_batch_id = GetOrCreateBatch(new_key);
        batches_.at(new_batch_id).change_mask |= reason;
        auto slot = AppendPrepared(batches_.at(new_batch_id), std::move(update.second.prepared));
        entry.storage = std::move(slot.first);
        entry.storage_index = slot.second;
        entry.batch_id = new_batch_id;
        batches_.at(new_batch_id).observations.push_back(update.first);
        touched_batches.insert(new_batch_id);
      }
      FinishBulkUpdate(touched_batches);
    }

    void ApplyObservationUpdates(
        std::vector<std::pair<ObservationId, PreparedObservationUpdate>>
            &updates)
    {
      std::vector<std::pair<ObservationId, PreparedObservationUpdate>>
          topology_updates;
      topology_updates.reserve(updates.size());
      std::unordered_set<IncrementalBatchId> numeric_updates;
      for (auto &update : updates)
      {
        ObservationEntry &entry = observations_.at(update.first);
        const IncrementalBatchKey new_key = MakeKey(
            update.second.update.binding, update.second.update.options);
        // Time/loss changes also affect backend/window metadata, so only a
        // coefficient/payload refresh is eligible for the no-Replace path.
        if (new_key == batches_.at(entry.batch_id).key &&
            update.second.update.options.loss_scale ==
                entry.options.loss_scale &&
            update.second.update.options.time_ns == entry.options.time_ns)
        {
          entry.observation =
              std::move(update.second.update.observation);
          entry.binding = std::make_shared<const Binding>(
              std::move(update.second.update.binding));
          RefreshPreparedInPlace(
              entry, std::move(update.second.prepared));
          entry.options = update.second.update.options;
          if (dependency_resolver_) numeric_updates.insert(entry.batch_id);
          continue;
        }
        topology_updates.emplace_back(
            update.first, std::move(update.second));
      }
      for (auto id : numeric_updates) ReindexDependencies(batches_.at(id));

      if (topology_updates.empty())
        return;

      std::unordered_set<IncrementalBatchId> touched_batches =
          DetachUpdatedObservations(topology_updates);
      for (auto &update : topology_updates)
      {
        ObservationEntry &entry = observations_.at(update.first);
        const auto old_batch_id = entry.batch_id;
        const auto old_options = entry.options;
        entry.observation = std::move(update.second.update.observation);
        entry.binding = std::make_shared<const Binding>(
            std::move(update.second.update.binding));
        entry.options = update.second.update.options;
        const IncrementalBatchKey new_key =
            MakeKey(*entry.binding, entry.options);
        auto reason = KeyChangeMask(batches_.at(old_batch_id).key, new_key);
        if (old_options.time_ns != entry.options.time_ns || old_options.loss_scale != entry.options.loss_scale)
          reason |= ChangeMask(BatchChange::ObservationMetadata);
        batches_.at(old_batch_id).change_mask |= reason;
        const IncrementalBatchId new_batch_id = GetOrCreateBatch(new_key);
        batches_.at(new_batch_id).change_mask |= reason;
        auto slot = AppendPrepared(batches_.at(new_batch_id), std::move(update.second.prepared));
        entry.storage = std::move(slot.first);
        entry.storage_index = slot.second;
        entry.batch_id = new_batch_id;
        batches_.at(new_batch_id).observations.push_back(update.first);
        touched_batches.insert(new_batch_id);
      }
      FinishBulkUpdate(touched_batches);
    }

    template <typename BatchPredicate>
    std::vector<ObservationId> CollectAffectedObservations(
        BatchPredicate &&batch_predicate) const
    {
      std::vector<ObservationId> affected;
      for (const auto &item : batches_)
      {
        const LiveBatch &batch = item.second;
        if (batch.observations.empty())
          continue;
        const ObservationEntry &representative =
            observations_.at(batch.observations.front());
        const BatchView view{
            batch.id, backend_tag_, *representative.binding,
            batch.key.binding_key, batch.key.partition,
            batch.key.binding_revision, batch.min_time_ns,
            batch.max_time_ns, batch.observations.size(),
            batch.backend_handle.has_value(), representative.observation};
        if (batch_predicate(view))
        {
          affected.insert(
              affected.end(), batch.observations.begin(),
              batch.observations.end());
        }
      }
      std::sort(affected.begin(), affected.end());
      affected.erase(
          std::unique(affected.begin(), affected.end()), affected.end());
      return affected;
    }

    static std::pair<SharedPreparedObservationStorage, std::size_t>
    AppendPrepared(LiveBatch &batch, PreparedObservation prepared)
    {
      if (!batch.appended) batch.appended = std::make_shared<PreparedObservationStorage>();
      if (batch.appended.use_count() == 1) batch.appended->clear();
      const auto index = batch.appended->size();
      batch.appended->push_back(std::move(prepared));
      return {batch.appended, index};
    }

    void EnsureEntries(LiveBatch &batch)
    {
      if (batch.entries_current) return;
      std::sort(batch.observations.begin(), batch.observations.end());
      batch.entries.clear();
      batch.entries.reserve(batch.observations.size());
      for (ObservationId id : batch.observations)
        batch.entries.push_back(&observations_.at(id));
      batch.entries_current = true;
    }

    std::shared_ptr<const JacobianWritePlan> WritePlan(const Binding &binding)
    {
      const auto structure = evaluator_->GetJacobianWriteStructure(binding);
      std::vector<std::size_t> key;
      key.reserve(4 + binding.parameters.sizes.size() * 5);
      key.push_back(Evaluator::kResidualDim);
      key.push_back(binding.parameters.sizes.size());
      for (auto size : binding.parameters.sizes) key.push_back(size);
      key.push_back(structure.groups.size());
      for (const auto &group : structure.groups)
      {
        key.push_back(group.residual_rows.offset);
        key.push_back(group.residual_rows.count);
        key.push_back(group.parameter_slices.size());
        for (const auto &slice : group.parameter_slices)
        {
          key.push_back(slice.block);
          key.push_back(slice.column_offset);
          key.push_back(slice.column_count);
        }
      }
      auto found = write_plans_.find(key);
      if (found != write_plans_.end()) return found->second;
      auto plan = std::make_shared<JacobianWritePlan>(
          structure, Evaluator::kResidualDim, binding.parameters.sizes);
      write_plans_.emplace(std::move(key), plan);
      return plan;
    }

    IncrementalBuiltBatch BuildBatch(LiveBatch &batch)
    {
      if (batch.observations.empty())
        throw std::logic_error("Cannot build an empty incremental batch");
      EnsureEntries(batch);
      const ObservationEntry &representative = *batch.entries.front();
      // New supports usually already have a contiguous ingestion buffer in
      // observation order. Transfer that snapshot to the factor instead of
      // copying every prepared payload once more. A split/merge or a hole
      // left by expiry still needs compaction. Never grow a registered factor's
      // storage: PublishPreparedStorage detaches an adopted append buffer.
      bool adopt = !batch.backend_handle &&
          representative.storage->size() == batch.entries.size();
      if (adopt)
        for (std::size_t i = 0; i < batch.entries.size(); ++i)
          if (batch.entries[i]->storage != representative.storage ||
              batch.entries[i]->storage_index != i)
          { adopt = false; break; }
      if (adopt)
        batch.built = representative.storage;
      else
      {
        if (!batch.built || batch.built.use_count() != 1)
          batch.built = std::make_shared<PreparedObservationStorage>();
        batch.built->clear();
        batch.built->reserve(batch.entries.size());
      }
      std::vector<double> loss_scales;
      loss_scales.reserve(batch.entries.size());
      for (const ObservationEntry *entry : batch.entries)
      {
        if (entry->batch_id != batch.id)
          throw std::logic_error("Incremental batch membership is inconsistent");
        if (!adopt) batch.built->push_back(entry->Prepared());
        loss_scales.push_back(entry->options.loss_scale);
      }

      IncrementalBuiltBatch result;
      result.batch_id = batch.id;
      result.backend_tag = backend_tag_;
      result.key = batch.key;
      result.parameter_blocks = representative.binding->parameters.blocks;
      result.min_time_ns = batch.min_time_ns;
      result.max_time_ns = batch.max_time_ns;
      result.observation_count = batch.entries.size();
      if constexpr (std::is_same_v<CostFunction, SegmentBatchCostFunction<Evaluator, LossPolicy>>)
      {
        if (!batch.write_plan) batch.write_plan = WritePlan(*representative.binding);
        result.cost_function = std::make_unique<CostFunction>(
            evaluator_, representative.binding, batch.built,
            std::move(loss_scales), loss_policy_, batch.write_plan);
      }
      else
      {
        result.cost_function = std::make_unique<CostFunction>(
            evaluator_, *representative.binding, batch.built,
            std::move(loss_scales), loss_policy_);
      }
      return result;
    }

    static void PublishPreparedStorage(LiveBatch &batch)
    {
      // Publish only after successful backend registration. Until then the
      // previously registered residual (including its row count) is unchanged.
      // BuildBatch's copy path exclusively owns its destination. Aliasing the
      // first row therefore means its adoption check already verified ALL row
      // slots, so even their shared_ptr/index publication can be skipped.
      if (batch.entries.front()->storage != batch.built)
      {
        for (std::size_t i = 0; i < batch.entries.size(); ++i)
        {
          batch.entries[i]->storage = batch.built;
          batch.entries[i]->storage_index = i;
        }
      }
      batch.registered.swap(batch.built);
      if (batch.appended == batch.registered) batch.appended.reset();
      if (batch.appended && batch.appended.use_count() == 1) batch.appended->clear();
    }

    void AttachDependencies(LiveBatch &batch, const Binding &binding)
    {
      if (!dependency_resolver_ || batch.last_indexed_binding == &binding) return;
      for (auto &dependency : dependency_resolver_(binding))
      {
        if (!dependency) throw std::invalid_argument("Null support dependency");
        if (std::find(batch.dependencies.begin(), batch.dependencies.end(), dependency) != batch.dependencies.end())
          continue;
        dependency_batches_[dependency.get()].insert(batch.id);
        batch.dependencies.push_back(std::move(dependency));
      }
      batch.last_indexed_binding = &binding;
    }
    void UnindexDependencies(LiveBatch &batch)
    {
      for (const auto &dependency : batch.dependencies)
      {
        auto found = dependency_batches_.find(dependency.get());
        if (found == dependency_batches_.end()) continue;
        found->second.erase(batch.id);
        if (found->second.empty()) dependency_batches_.erase(found);
      }
      batch.dependencies.clear();
      batch.last_indexed_binding = nullptr;
    }
    void ReindexDependencies(LiveBatch &batch)
    {
      if (!dependency_resolver_) return;
      // Keep nodes alive while replacing edges, so a registry's weak handles
      // preserve both identity and the previous semantic snapshot.
      const auto keep_alive = batch.dependencies;
      UnindexDependencies(batch);
      for (auto id : batch.observations)
      {
        const auto &entry = observations_.at(id);
        AttachDependencies(batch, *entry.binding);
      }
    }

    void PublishRefreshDependencies(LiveBatch &batch)
    {
      if (!dependency_resolver_) return;
      auto &next = batch.refresh_dependencies;
      // Tiny support sets: compare identities directly, without per-row
      // lookups, allocations, or erasing/reinserting unchanged reverse edges.
      for (auto it = batch.dependencies.begin(); it != batch.dependencies.end();)
      {
        if (std::find(next.begin(), next.end(), *it) != next.end()) { ++it; continue; }
        auto found = dependency_batches_.find(it->get());
        if (found != dependency_batches_.end())
        {
          found->second.erase(batch.id);
          if (found->second.empty()) dependency_batches_.erase(found);
        }
        it = batch.dependencies.erase(it);
        ++refresh_summary_.dependency_edges_removed;
      }
      for (auto &dependency : next)
      {
        if (std::find(batch.dependencies.begin(), batch.dependencies.end(), dependency) != batch.dependencies.end())
          continue;
        dependency_batches_[dependency.get()].insert(batch.id);
        batch.dependencies.push_back(dependency);
        ++refresh_summary_.dependency_edges_added;
      }
      batch.last_indexed_binding = batch.previous_refresh_binding.get();
      next.clear();
      ++refresh_summary_.dependency_sets_checked;
    }

    DependencyResolver dependency_resolver_;
    std::unordered_map<const IncrementalSupportDependency *, std::unordered_set<IncrementalBatchId>> dependency_batches_;
    std::unordered_set<IncrementalBatchId> invalidated_batches_;
    IncrementalRefreshSummary refresh_summary_;
    bool seal_after_flush_ = false;
    std::shared_ptr<const Evaluator> evaluator_;
    LossPolicy loss_policy_;
    std::uint64_t backend_tag_ = 0;
    IncrementalBatchId *next_batch_id_ = nullptr;
    ObservationId next_observation_id_ = 1;
    std::unordered_map<ObservationId, ObservationEntry> observations_;
    std::unordered_map<IncrementalBatchId, LiveBatch> batches_;
    using ExpiryEntry = std::pair<std::int64_t, IncrementalBatchId>;
    std::vector<ExpiryEntry> expiry_index_;
    std::size_t last_window_visited_batches_ = 0;
    std::unordered_map<
        IncrementalBatchKey, IncrementalBatchId,
        IncrementalBatchKeyHash>
        batch_indices_;
    std::unordered_set<IncrementalBatchId> dirty_batches_;
    struct IngestionBinding
    {
      SharedBindingHandle handle;
      IncrementalBatchId batch = 0;
      std::uint64_t revision = 0;
    };
    std::unordered_map<std::size_t, IngestionBinding> ingestion_bindings_;
    std::vector<LiveBatch *> refresh_batches_;
    std::vector<std::pair<ObservationId, PreparedBindingUpdate>> resolved_refresh_;
    std::map<std::vector<std::size_t>, std::shared_ptr<const JacobianWritePlan>> write_plans_;
  };

  /// Long-lived batching coordinator for persistent/sliding-window problems.
  /// It owns raw observations and batch membership, but delegates ownership and
  /// registration of live CostFunctions to IncrementalBatchBackend.
  ///
  /// This class is intentionally not thread-safe. Add/rebind/window/flush must
  /// run while the backend problem is not being evaluated by ceres::Solve.
  class IncrementalProblemBatcher
  {
  public:
    explicit IncrementalProblemBatcher(IncrementalBatchBackend &backend)
        : backend_(&backend)
    {
    }

    IncrementalProblemBatcher(const IncrementalProblemBatcher &) = delete;
    IncrementalProblemBatcher &operator=(
        const IncrementalProblemBatcher &) = delete;

    template <typename Evaluator, typename LossPolicy = NoLossPolicy>
    IncrementalEvaluatorChannel<Evaluator, LossPolicy> &RegisterEvaluator(
        Evaluator evaluator, LossPolicy loss_policy = LossPolicy{},
        const std::uint64_t backend_tag = 0)
    {
      static_assert(
          kIsEvaluator<Evaluator>,
          "Evaluator must derive from splbatch::EvaluatorBase");
      auto channel = std::make_unique<
          IncrementalEvaluatorChannel<Evaluator, LossPolicy>>(
          std::move(evaluator), std::move(loss_policy), backend_tag,
          &next_batch_id_);
      auto &result = *channel;
      result.SealAfterFlush(seal_after_flush_);
      channels_.push_back(std::move(channel));
      return result;
    }

    template <typename Evaluator, typename LossPolicy = NoLossPolicy>
    IncrementalEvaluatorChannel<
        Evaluator, LossPolicy,
        CompressedSegmentBatchCostFunction> &
    RegisterCompressedEvaluator(
        Evaluator evaluator, LossPolicy loss_policy = LossPolicy{},
        const std::uint64_t backend_tag = 0)
    {
      static_assert(
          kIsEvaluator<Evaluator>,
          "Evaluator must derive from splbatch::EvaluatorBase");
      using Channel = IncrementalEvaluatorChannel<
          Evaluator, LossPolicy,
          CompressedSegmentBatchCostFunction>;
      auto channel = std::make_unique<Channel>(
          std::move(evaluator), std::move(loss_policy), backend_tag,
          &next_batch_id_);
      auto &result = *channel;
      result.SealAfterFlush(seal_after_flush_);
      channels_.push_back(std::move(channel));
      return result;
    }

    IncrementalFlushSummary Flush()
    {
      IncrementalFlushSummary result;
      backend_->ResetCeresTiming();
      backend_->BeginCommit();
      try
      {
        for (const auto &channel : channels_)
          result += channel->Flush(*backend_);
      }
      catch (...)
      {
        backend_->EndCommit();
        throw;
      }
      {
        IncrementalScopedTimer timer(backend_->TimingEnabled() ? &result.backend_finalize_ms : nullptr);
        backend_->EndCommit();
      }
      return result;
    }

    void InvalidateDependency(const IncrementalSupportDependency *dependency)
    { for (auto &channel : channels_) channel->InvalidateDependency(dependency); }

    void SealAfterFlush(bool enabled)
    {
      seal_after_flush_ = enabled;
      for (auto &channel : channels_) channel->SealAfterFlush(enabled);
    }

    IncrementalRefreshSummary TakeRefreshSummary()
    {
      IncrementalRefreshSummary result;
      for (auto &channel : channels_) result += channel->TakeRefreshSummary();
      return result;
    }

    /// Stage removal of observations at or before the window boundary. The
    /// caller controls when Flush() changes the live problem, which lets CT
    /// systems build a marginalization prior from the old problem first.
    std::size_t StageWindowCutoff(const std::int64_t cutoff_ns)
    {
      std::size_t removed = 0;
      for (const auto &channel : channels_)
        removed += channel->StageWindowCutoff(cutoff_ns);
      return removed;
    }

    /// Convenience path for fixed-boundary windows that do not need to
    /// marginalize the outgoing residuals before removal.
    IncrementalFlushSummary AdvanceWindow(const std::int64_t cutoff_ns)
    {
      StageWindowCutoff(cutoff_ns);
      return Flush();
    }

    void Clear()
    {
      backend_->BeginCommit();
      try
      {
        for (const auto &channel : channels_)
          channel->Clear(*backend_);
      }
      catch (...)
      {
        backend_->EndCommit();
        throw;
      }
      backend_->EndCommit();
    }

    std::size_t ObservationCount() const
    {
      std::size_t count = 0;
      for (const auto &channel : channels_)
        count += channel->ObservationCount();
      return count;
    }

    std::size_t BatchCount() const
    {
      std::size_t count = 0;
      for (const auto &channel : channels_)
        count += channel->BatchCount();
      return count;
    }

    std::size_t RegisteredBatchCount() const
    {
      std::size_t count = 0;
      for (const auto &channel : channels_)
        count += channel->RegisteredBatchCount();
      return count;
    }

    std::size_t DirtyBatchCount() const
    {
      std::size_t count = 0;
      for (const auto &channel : channels_)
        count += channel->DirtyBatchCount();
      return count;
    }

  private:
    IncrementalBatchBackend *backend_ = nullptr;
    bool seal_after_flush_ = false;
    IncrementalBatchId next_batch_id_ = 1;
    std::vector<std::unique_ptr<IncrementalChannelBase>> channels_;
  };
} // namespace splbatch
