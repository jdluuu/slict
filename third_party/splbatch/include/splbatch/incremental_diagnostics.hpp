#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace splbatch
{
  // Reasons are a bit set: append and expiry can affect the same replacement.
  enum class BatchChange : std::uint32_t
  {
    Append = 1u << 0, WindowExpired = 1u << 1, ObservationRemoved = 1u << 2,
    Support = 1u << 3, Semantics = 1u << 4, ObservationMetadata = 1u << 5
  };
  constexpr std::size_t kBatchChangeCount = 6;
  constexpr std::uint32_t ChangeMask(BatchChange reason)
  { return static_cast<std::uint32_t>(reason); }

  class IncrementalScopedTimer
  {
    using Clock = std::chrono::steady_clock;
    double *milliseconds_;
    Clock::time_point start_;
  public:
    explicit IncrementalScopedTimer(double *milliseconds) : milliseconds_(milliseconds)
    { if (milliseconds_) start_ = Clock::now(); }
    ~IncrementalScopedTimer()
    {
      if (milliseconds_) *milliseconds_ +=
          std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }
  };

  // Opaque, lifetime-owned dependency identity. The application detects semantic
  // changes and notifies the batcher; splbatch has no spline/robot assumptions.
  struct IncrementalSupportDependency
  { virtual ~IncrementalSupportDependency() = default; };

  struct IncrementalRefreshSummary
  {
    std::size_t visited_batches = 0, affected_batches = 0;
    std::size_t coefficient_rows = 0, resolved_rows = 0;
    // Support-preserving refresh only; topology maintenance is counted by Flush.
    std::size_t dependency_sets_checked = 0;
    std::size_t dependency_edges_added = 0, dependency_edges_removed = 0;
    IncrementalRefreshSummary &operator+=(const IncrementalRefreshSummary &s)
    {
      visited_batches += s.visited_batches; affected_batches += s.affected_batches;
      coefficient_rows += s.coefficient_rows; resolved_rows += s.resolved_rows;
      dependency_sets_checked += s.dependency_sets_checked;
      dependency_edges_added += s.dependency_edges_added;
      dependency_edges_removed += s.dependency_edges_removed;
      return *this;
    }
  };
}
