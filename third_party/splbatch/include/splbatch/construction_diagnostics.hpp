#pragma once

// Opt-in diagnostic build only. Normal builds erase all probes, including
// observation selection and clock calls. Do not mix differently configured
// instantiations in the same process.
#ifdef SPLBATCH_CONSTRUCTION_DIAGNOSTICS
#include <array>
#include <chrono>
#include <cstdint>
#include <utility>

namespace splbatch::construction_diagnostics {
enum class Stage : std::size_t {
  Binding, Support, Prepare, BatchCreation, JacobianAllocation,
  ObservationStorage, NativeRegistration, BindingCleanup, ChannelCleanup,
  Calibration, Count
};
inline constexpr std::array<const char*, static_cast<std::size_t>(Stage::Count)> Names = {
  "binding", "support", "prepare", "batch_creation", "jacobian_allocation",
  "observation_storage", "native_registration", "binding_cleanup", "channel_cleanup",
  "calibration"
};
struct Counter {
  double exclusive_ns = 0;
  std::uint64_t calls = 0;
  double weighted_ns = 0, weighted_calls = 0, weighted_children = 0;
};
struct Profile {
  std::array<Counter, static_cast<std::size_t>(Stage::Count)> stages{};
  std::uint64_t observations = 0, sampled_observations = 0;
  int stride = 1, offset = 0;
  bool observation_selected = false;
};
class Scope;
inline thread_local Profile* active = nullptr;
inline thread_local Scope* enclosing = nullptr;
using Clock = std::chrono::steady_clock;

class Session {
 public:
  explicit Session(Profile* profile) : previous_(active), previous_scope_(enclosing) {
    active = profile; enclosing = nullptr;
  }
  ~Session() { active = previous_; enclosing = previous_scope_; }
  Session(const Session&) = delete;
 private:
  Profile* previous_;
  Scope* previous_scope_;
};

class ObservationSelection {
 public:
  ObservationSelection() : profile_(active) {
    if (!profile_) return;
    previous_ = profile_->observation_selected;
    profile_->observation_selected =
        profile_->observations++ % profile_->stride == static_cast<unsigned>(profile_->offset);
    if (profile_->observation_selected) ++profile_->sampled_observations;
  }
  ~ObservationSelection() { if (profile_) profile_->observation_selected = previous_; }
 private:
  Profile* profile_;
  bool previous_ = false;
};

// Nested scopes record exclusive intervals. Sampled parent/child scopes use
// the same observation selection and weight; batch/registration scopes are
// outside ingestion and always measured at weight one.
class Scope {
 public:
  Scope(Stage stage, bool observation) : profile_(active), stage_(stage) {
    if (!profile_ || (observation && !profile_->observation_selected)) {
      profile_ = nullptr; return;
    }
    weight_ = observation ? profile_->stride : 1;
    parent_ = enclosing; enclosing = this;
    start_ = Clock::now();
  }
  ~Scope() { Stop(); }
  void Stop() {
    if (!profile_) return;
    const auto end = Clock::now();
    const double inclusive = std::chrono::duration<double, std::nano>(end-start_).count();
    const double exclusive = inclusive - child_ns_;
    auto& counter = profile_->stages[static_cast<std::size_t>(stage_)];
    counter.exclusive_ns += exclusive; ++counter.calls;
    counter.weighted_ns += exclusive * weight_;
    counter.weighted_calls += weight_;
    counter.weighted_children += children_ * weight_;
    if (parent_) { parent_->child_ns_ += inclusive; ++parent_->children_; }
    enclosing = parent_; profile_ = nullptr;
  }
  Scope(const Scope&) = delete;
 private:
  Profile* profile_;
  Stage stage_;
  Scope* parent_ = nullptr;
  Clock::time_point start_;
  double child_ns_ = 0;
  int children_ = 0;
  int weight_ = 1;
};

template<class Function>
decltype(auto) Measure(Stage stage, bool observation, Function&& function) {
  Scope scope(stage, observation);
  return std::forward<Function>(function)();
}
}  // namespace splbatch::construction_diagnostics

#define SPLBATCH_DIAG_SCOPE(name, stage, observation) \
  ::splbatch::construction_diagnostics::Scope name( \
      ::splbatch::construction_diagnostics::Stage::stage, observation)
#define SPLBATCH_DIAG_STOP(name) name.Stop()
#define SPLBATCH_DIAG_MEASURE(stage, observation, expression) \
  ::splbatch::construction_diagnostics::Measure( \
      ::splbatch::construction_diagnostics::Stage::stage, observation, \
      [&]() { return (expression); })
#define SPLBATCH_DIAG_OBSERVATION(name) \
  ::splbatch::construction_diagnostics::ObservationSelection name
#else
#define SPLBATCH_DIAG_SCOPE(name, stage, observation)
#define SPLBATCH_DIAG_STOP(name)
#define SPLBATCH_DIAG_MEASURE(stage, observation, expression) (expression)
#define SPLBATCH_DIAG_OBSERVATION(name)
#endif
