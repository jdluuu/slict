#include <splbatch/problem_batcher.hpp>
#include <splbatch/incremental_problem_batcher.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <new>
#include <numeric>
#include <random>

// Fail each ordinary allocation in AddObservation, independently of timing.
// Failure injection is disabled during setup, evaluation and test assertions.
namespace allocation_fault {
thread_local long remaining = -1;
thread_local std::size_t calls = 0;
}
__attribute__((noinline)) void* operator new(std::size_t bytes) {
  ++allocation_fault::calls;
  if (allocation_fault::remaining == 0) {
    allocation_fault::remaining = -1;
    throw std::bad_alloc();
  }
  if (allocation_fault::remaining > 0) --allocation_fault::remaining;
  if (auto* p = std::malloc(bytes ? bytes : 1)) return p;
  throw std::bad_alloc();
}
__attribute__((noinline)) void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete[](void* p, std::size_t) noexcept { ::operator delete(p); }

namespace {
void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct Context { double offset = 0; bool fail = false; };
using Binding = splbatch::BatchBinding<Context>;
struct Evaluator : splbatch::EvaluatorBase<Evaluator, double, Binding, double, double, 1> {
  explicit Evaluator(std::size_t* count = nullptr) : count(count) {}
  double PrepareObservationImpl(const Binding& b, double observation) const {
    if (count) ++*count;
    if (b.context.fail) throw std::runtime_error("Injected preparation failure");
    return observation + b.context.offset;
  }
  double PrepareBatchImpl(const Binding& b, double const* const* parameters,
                          const splbatch::EvaluationRequest&) const {
    double sum = 0;
    for (std::size_t i = 0; i < b.parameters.blocks.size(); ++i)
      sum += parameters[i][0];
    return sum;
  }
  template <typename Output>
  bool EvaluateObservationImpl(const Binding& b, double observation, double runtime,
                               double const* const*, const Output& out) const {
    out.Residual()[0] = runtime - observation;
    for (std::size_t i = 0; i < b.parameters.blocks.size(); ++i)
      if (out.JacobianRequested(i)) out.Jacobian(i)(0, 0) = 1;
    return true;
  }
  std::size_t* count;
};
} // namespace

namespace splbatch {
template <> struct BindingPreparationReuse<::Evaluator> {
  static bool Compatible(const ::Binding& stored, const ::Context& incoming) noexcept {
    return stored.context.offset == incoming.offset && stored.context.fail == incoming.fail;
  }
};
} // namespace splbatch

namespace {
// No reuse specialization: even equal-looking contexts take the safe fallback.
struct UnadaptedEvaluator : splbatch::EvaluatorBase<UnadaptedEvaluator, double, Binding, double, double, 1> {
  double PrepareObservationImpl(const Binding& b, double o) const { return Evaluator().PrepareObservation(b, o); }
  double PrepareBatchImpl(const Binding& b, double const* const* p,
                          const splbatch::EvaluationRequest& q) const { return Evaluator().PrepareBatch(b, p, q); }
  template <typename Output>
  bool EvaluateObservationImpl(const Binding& b, double o, double r, double const* const* p,
                               const Output& out) const { return Evaluator().EvaluateObservation(b, o, r, p, out); }
};
using Input = splbatch::InlineBindingInput<Context, 16>;
Input InputFrom(const Binding& binding) {
  Input input;
  input.context = binding.context;
  for (std::size_t i = 0; i < binding.parameters.blocks.size(); ++i)
    input.AddParameter(binding.parameters.blocks[i], binding.parameters.sizes[i]);
  for (auto tag : binding.support_signature) input.support_signature.push_back(tag);
  return input;
}
Binding Bind(double* x, double offset = 0) {
  return {{offset, false}, {{x}, {1}}, {}};
}
struct Registry : splbatch::ResidualRegistry {
  struct Entry {
    std::unique_ptr<ceres::CostFunction> cost;
    std::vector<double*> parameters;
  };
  std::vector<Entry> entries;
  void AddResidualBlock(std::unique_ptr<ceres::CostFunction> cost,
                        std::vector<double*> parameters) override {
    entries.push_back({std::move(cost), std::move(parameters)});
  }
  std::vector<double> Residuals() const {
    std::vector<double> result;
    for (const auto& entry : entries) {
      const auto start = result.size();
      result.resize(start + entry.cost->num_residuals());
      Check(entry.cost->Evaluate(entry.parameters.data(), result.data() + start, nullptr),
            "Residual evaluation failed");
    }
    return result;
  }
};
void Near(const std::vector<double>& actual, const std::vector<double>& expected) {
  Check(actual.size() == expected.size(), "Residual/loss array sizes changed");
  for (std::size_t i = 0; i < actual.size(); ++i)
    Check(std::abs(actual[i] - expected[i]) < 1e-10, "Observation order/context/scale changed");
}

void OrderContextAndLifetime(bool lightweight = false) {
  for (bool shuffled : {false, true}) {
    std::vector<double> parameters(160);
    std::iota(parameters.begin(), parameters.end(), 1000.);
    std::vector<int> order(480);
    std::iota(order.begin(), order.end(), 0);
    if (shuffled) { std::mt19937 rng(42); std::shuffle(order.begin(), order.end(), rng); }
    Registry registry;
    splbatch::ProblemBatcher batcher(registry);
    std::size_t prepared = 0;
    auto& channel = batcher.RegisterEvaluator(Evaluator(&prepared));
    struct Expected { splbatch::BatchKey key; std::vector<double> residuals, scales; };
    std::vector<Expected> groups;
    for (int i : order) {
      const int support = i / 3;
      const double scale = (i % 3 + 1) * (i % 3 + 1);
      auto binding = Bind(&parameters[support], i * .125);
      // Independent reference uses the preserved public owning Key API.
      const auto key = binding.Key();
      auto found = std::find_if(groups.begin(), groups.end(),
                               [&](const auto& g) { return g.key == key; });
      if (found == groups.end()) { groups.push_back({key, {}, {}}); found = groups.end() - 1; }
      found->residuals.push_back((parameters[support] - i - binding.context.offset) * std::sqrt(scale));
      found->scales.push_back(std::sqrt(scale));
      if (lightweight) channel.AddObservation(InputFrom(binding), i, scale);
      else channel.AddObservation(binding, i, scale);
      // The channel must own the support, not retain a view into this input.
      binding.parameters.blocks.clear(); binding.parameters.sizes.clear();
      binding.support_signature = {999};
    }
    Check(prepared == order.size(), "Cache hit skipped preparation");
    Check(channel.PendingBatchCount() == groups.size(), "Grouping changed");
    batcher.Commit();
    Check(registry.entries.size() == groups.size(), "Batch count changed on commit");
    std::vector<double> expected, updated;
    for (const auto& g : groups) for (std::size_t i = 0; i < g.residuals.size(); ++i) {
      expected.push_back(g.residuals[i]); updated.push_back(g.residuals[i] + g.scales[i]);
    }
    Near(registry.Residuals(), expected);
    for (auto& p : parameters) p += 1;
    Near(registry.Residuals(), updated);
    // A new cycle at identical parameter addresses must not retain old rows.
    Registry fresh;
    splbatch::ProblemBatcher next(fresh);
    auto& next_channel = next.RegisterEvaluator(Evaluator());
    next_channel.AddObservation(Bind(&parameters[0], 7), 2);
    next.Commit(); Near(fresh.Residuals(), {parameters[0] - 9});
  }
}

void SupportAndValidation() {
  double x[4] = {10, 20, 30, 40};
  Registry registry;
  splbatch::ProblemBatcher batcher(registry);
  auto& channel = batcher.RegisterEvaluator(Evaluator());
  Binding original{{}, {{x, x + 1}, {1, 1}}, {7}};
  std::vector<Binding> valid{original};
  auto b = original; std::swap(b.parameters.blocks[0], b.parameters.blocks[1]); valid.push_back(b);
  b = original; b.parameters.sizes[0] = 2; valid.push_back(b);
  b = original; b.support_signature[0] = 8; valid.push_back(b);
  b = original; b.support_signature.push_back(9); valid.push_back(b);
  b = original; b.parameters.blocks.pop_back(); b.parameters.sizes.pop_back(); valid.push_back(b);
  b = original; b.parameters.blocks[0] = x + 2; valid.push_back(b);
  for (const auto& v : valid) {
    Check(splbatch::detail::HashBatchSupport(v) == splbatch::BatchKeyHash{}(v.Key()),
          "Non-owning hash disagrees with legacy hash");
    channel.AddObservation(v, 0);
  }
  for (const auto& v : valid) channel.AddObservation(v, 1);
  Check(channel.PendingBatchCount() == valid.size(), "Distinct supports were merged");
  channel.AddObservation(original, 2); // malformed inputs must not hit this hint
  const auto count = channel.PendingObservationCount();
  std::vector<Binding> invalid;
  b = original; b.parameters.sizes.pop_back(); invalid.push_back(b);
  b = original; b.parameters.blocks.pop_back(); invalid.push_back(b);
  b = original; b.parameters.sizes[0] = 0; invalid.push_back(b);
  b = original; b.parameters.sizes[0] = -1; invalid.push_back(b);
  b = original; b.parameters.blocks[0] = nullptr; invalid.push_back(b);
  b = original; b.parameters.blocks[0] = b.parameters.blocks[1]; invalid.push_back(b);
  for (const auto& v : invalid) {
    bool key_failed = false, add_failed = false;
    try { (void)v.Key(); } catch (const std::invalid_argument&) { key_failed = true; }
    try { channel.AddObservation(v, 0); } catch (const std::invalid_argument&) { add_failed = true; }
    Check(key_failed && add_failed, "Legacy validation was bypassed");
    Check(channel.PendingObservationCount() == count, "Invalid input appended a row");
  }
  for (double scale : {0., -1., std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    bool failed = false;
    try { channel.AddObservation(original, 0, scale); } catch (const std::invalid_argument&) { failed = true; }
    Check(failed && channel.PendingObservationCount() == count, "Invalid scale on cache hit accepted");
  }
  batcher.Commit();
  splbatch::EvaluatorChannel<Evaluator> other(Evaluator(), {});
  // Independent channel with identical addresses and different preparation.
  other.AddObservation(Bind(x, 3), 4);
  Registry other_registry; other.CommitTo(other_registry);
  Near(other_registry.Residuals(), {3});
}

void HashCollision(bool lightweight = false) {
  double x = 10;
  auto a = Bind(&x), b = a;
  const std::size_t c = 0x9e3779b9;
  // std::hash<size_t> is identity on the supported libstdc++ toolchain.
  // Construct equal hashes after two signature entries, then append identical
  // parameter layouts. This tests collisions in the full hash, not just buckets.
  const std::size_t target = c ^ (c + (c << 6) + (c >> 2));
  const std::size_t seed = c + 1;
  const std::size_t second = (target ^ seed) - c - (seed << 6) - (seed >> 2);
  a.support_signature = {0, 0}; b.support_signature = {1, second};
  Check(splbatch::detail::HashBatchSupport(a) == splbatch::detail::HashBatchSupport(b),
        "Test did not construct a full hash collision");
  Registry registry;
  splbatch::ProblemBatcher batcher(registry);
  auto& channel = batcher.RegisterEvaluator(Evaluator());
  if (lightweight) {
    channel.AddObservation(InputFrom(a), 1); channel.AddObservation(InputFrom(b), 2);
    channel.AddObservation(InputFrom(a), 3); channel.AddObservation(InputFrom(b), 4);
  } else {
    channel.AddObservation(a, 1); channel.AddObservation(b, 2);
    channel.AddObservation(a, 3); channel.AddObservation(b, 4);
  }
  Check(channel.PendingBatchCount() == 2, "Hash collision merged supports");
  batcher.Commit(); Near(registry.Residuals(), {9, 7, 8, 6});
}

void FailureRollback() {
  for (int mode = 0; mode < 4; ++mode) {
    int failures = 0;
    for (long fail_at = 0; fail_at < 64; ++fail_at) {
      double x = 10, y = 20;
      auto a = Bind(&x), b = Bind(&y);
      Registry registry;
      splbatch::ProblemBatcher batcher(registry);
      auto& channel = batcher.RegisterEvaluator(Evaluator());
      if (mode != 0) channel.AddObservation(a, 1, 4);
      if (mode == 3) channel.AddObservation(b, 2, 9);
      const auto before = channel.PendingObservationCount();
      const auto batches = channel.PendingBatchCount();
      const auto& target = mode == 2 ? b : a;
      bool failed = false;
      allocation_fault::remaining = fail_at;
      try { channel.AddObservation(target, 3, 16); }
      catch (const std::bad_alloc&) { failed = true; }
      allocation_fault::remaining = -1;
      if (failed) {
        ++failures;
        Check(channel.PendingObservationCount() == before, "Allocation failure left an observation");
        Check(channel.PendingBatchCount() == batches, "Allocation failure published a batch");
        channel.AddObservation(target, 3, 16);
      }
      batcher.Commit();
      if (mode == 0) Near(registry.Residuals(), {28});
      if (mode == 1) Near(registry.Residuals(), {18, 28});
      if (mode == 2) Near(registry.Residuals(), {18, 68});
      if (mode == 3) Near(registry.Residuals(), {18, 28, 54});
      if (!failed) break;
      Check(fail_at < 63, "Did not exhaust allocation failure points");
    }
    Check(failures > 0, "Allocation failure injection did not run");
  }
  double x = 10, y = 20;
  Registry registry;
  splbatch::ProblemBatcher batcher(registry);
  auto& channel = batcher.RegisterEvaluator(Evaluator());
  channel.AddObservation(Bind(&x), 1);
  for (auto* p : {&x, &y}) {
    auto bad = Bind(p); bad.context.fail = true;
    bool failed = false;
    try { channel.AddObservation(bad, 2); } catch (const std::runtime_error&) { failed = true; }
    Check(failed && channel.PendingBatchCount() == 1 && channel.PendingObservationCount() == 1,
          "Preparation failure changed batches");
  }
  channel.AddObservation(Bind(&x, 3), 2);
  channel.AddObservation(Bind(&y, 4), 3);
  batcher.Commit(); Near(registry.Residuals(), {9, 5, 13});
}

void AllocationFreeHit() {
  double x = 10, y = 20;
  const auto b = Bind(&x);
  splbatch::EvaluatorChannel<Evaluator> channel(Evaluator(), {});
  // After crossing a geometric vector growth boundary, the next append has
  // capacity. Support lookup and validation must not allocate an owning key.
  for (int i = 0; i < 65; ++i) channel.AddObservation(b, i);
  const auto calls = allocation_fault::calls;
  channel.AddObservation(b, 65);
  Check(allocation_fault::calls == calls, "Steady support hit allocated a key");
  channel.AddObservation(Bind(&y), 0);
  const auto table_calls = allocation_fault::calls;
  channel.AddObservation(b, 66);
  Check(allocation_fault::calls == table_calls, "Support table hit allocated a key");
}

void CompressedChannel() {
  double x = 10;
  ceres::Problem problem;
  splbatch::ProblemBatcher batcher(problem);
  auto& channel = batcher.RegisterCompressedEvaluator(Evaluator());
  for (int i = 1; i <= 3; ++i) channel.AddObservation(Bind(&x), i);
  batcher.Commit();
  double cost = 0;
  std::vector<double> gradient;
  ceres::CRSMatrix jacobian;
  Check(problem.Evaluate({}, &cost, nullptr, &gradient, &jacobian), "Compressed evaluation failed");
  double hessian = 0;
  for (double j : jacobian.values) hessian += j * j;
  Check(std::abs(cost - 97) < 1e-10 && gradient.size() == 1 &&
        std::abs(gradient[0] - 24) < 1e-10 && std::abs(hessian - 3) < 1e-10,
        "Compressed channel changed cost/gradient/Hessian");
}

void IncrementalKeyTransitions() {
  double x = 0, y = 10;
  ceres::Problem problem;
  splbatch::CeresIncrementalBatchBackend backend(problem);
  splbatch::IncrementalProblemBatcher batcher(backend);
  auto& channel = batcher.RegisterEvaluator(Evaluator());
  using Channel = std::remove_reference_t<decltype(channel)>;
  const auto first = channel.AddObservation(Bind(&x), 1, 1);
  channel.AddObservation(Bind(&x), 2, 2);
  batcher.Flush();
  const auto all = [](const auto&) { return true; };
  const auto check_cost = [&](double expected) {
    double cost = 0;
    Check(problem.Evaluate({}, &cost, nullptr, nullptr, nullptr), "Incremental evaluation failed");
    Check(std::abs(cost - expected) < 1e-10 && problem.NumResiduals() == 2,
          "Incremental update lost or retained an incorrect observation");
  };
  // Same support: refresh registered prepared slots without replacing factors.
  channel.RefreshAffectedBatches(all, [&](auto, auto observation, auto binding, auto options) {
    binding.context.offset = 1;
    return Channel::ObservationUpdate{observation, binding, options};
  });
  check_cost(6.5);
  Check(batcher.Flush().replaced_batches == 0, "Numeric refresh replaced an incremental batch");
  // One observation moves to another parameter support; the other stays put.
  channel.RefreshAffectedBatches(all, [&](auto id, auto observation, auto binding, auto options) {
    if (id == first) binding.parameters.blocks[0] = &y;
    return Channel::ObservationUpdate{observation, binding, options};
  });
  auto changes = batcher.Flush();
  Check(changes.added_batches == 1 && changes.replaced_batches == 1,
        "Support migration failed to update both incremental batches");
  check_cost(36.5);
  // Partition and semantic revision remain part of the incremental key even
  // when the ordinary support key and parameter addresses are unchanged.
  for (int transition = 0; transition < 2; ++transition) {
    channel.RefreshAffectedBatches(all, [&](auto id, auto observation, auto binding, auto options) {
      if (id == first) {
        if (transition == 0) ++options.partition;
        else ++options.binding_revision;
      }
      return Channel::ObservationUpdate{observation, binding, options};
    });
    changes = batcher.Flush();
    Check(changes.added_batches == 1 && changes.removed_batches == 1,
          "Incremental partition/revision change reused an obsolete batch");
    check_cost(36.5);
  }
  Check(channel.RemoveObservation(first), "Incremental removal failed");
  Check(batcher.Flush().removed_batches == 1, "Empty incremental batch survived removal");
  const auto replacement = channel.AddObservation(Bind(&y), 3, 3, 1., 1, 1);
  Check(replacement != first, "Incremental observation ID was reused");
  Check(batcher.Flush().added_batches == 1, "Re-add reused a deleted incremental batch");
  check_cost(29);
}

void InlineInputStorage() {
  double parameters[48];
  std::iota(std::begin(parameters), std::end(parameters), 1.);
  for (int count : {0, 1, 16, 17, 48}) {
    Input input;
    const auto before = allocation_fault::calls;
    for (int i = 0; i < count; ++i) input.AddParameter(parameters + i, 1);
    if (count <= 16) Check(before == allocation_fault::calls, "Inline input construction allocated");
    Check(input.parameters.blocks.IsInline() == (count <= 16), "Incorrect parameter spill boundary");
    for (int i = 0; i < 9; ++i) input.support_signature.push_back(i);
    Check(!input.support_signature.IsInline(), "Long signature did not spill");
    auto copy = input;
    auto moved = std::move(input);
    input.parameters.Validate();
    Check(input.parameters.blocks.empty() && input.parameters.sizes.empty(), "Moved-from input has stale lengths");
    input.AddParameter(parameters, 1); // moved-from input remains reusable
    Input assigned; assigned = copy;
    Input move_assigned; move_assigned = std::move(assigned);
    auto binding = moved.ToBinding();
    Check(splbatch::detail::SameBatchSupport(copy, binding) &&
          splbatch::detail::SameBatchSupport(move_assigned, binding), "Copy/move retained a stale self-view");
    Check(splbatch::detail::HashBatchSupport(moved) == splbatch::BatchKeyHash{}(binding.Key()),
          "Inline/spilled hash differs from owning key");
    if (count) { copy.parameters.blocks[0] = nullptr; Check(moved.parameters.blocks[0] == parameters, "Copy aliases input metadata"); }
    moved.parameters.blocks.clear(); moved.parameters.sizes.clear(); moved.support_signature.clear();
    auto empty_copy = moved;
    Check(empty_copy.parameters.blocks.empty() && empty_copy.support_signature.empty(), "Cleared spilled input copy is not empty");
    empty_copy.AddParameter(parameters, 1);
    empty_copy.parameters.Validate();

    Registry registry;
    splbatch::ProblemBatcher batcher(registry);
    auto& channel = batcher.RegisterEvaluator(Evaluator());
    channel.AddObservation(move_assigned, 1);
    channel.AddObservation(move_assigned, 2);
    batcher.Commit();
    const double sum = count * (count + 1) / 2.;
    Near(registry.Residuals(), {sum - 1, sum - 2});
  }
  // Failure in either spill allocation must not append only half a parameter.
  for (long fail_at = 0; fail_at < 2; ++fail_at) {
    splbatch::InlineBindingInput<Context, 1> input;
    input.AddParameter(parameters, 1);
    bool failed = false;
    allocation_fault::remaining = fail_at;
    try { input.AddParameter(parameters + 1, 1); } catch (const std::bad_alloc&) { failed = true; }
    allocation_fault::remaining = -1;
    Check(failed && input.parameters.blocks.size() == 1 && input.parameters.sizes.size() == 1,
          "Spill failure changed parameter lengths");
    input.AddParameter(parameters + 1, 1); input.parameters.Validate();
  }
}

void InlineValidationAndContext() {
  double x = 10, y = 20;
  Registry registry;
  splbatch::ProblemBatcher batcher(registry);
  std::size_t preparations = 0;
  auto& channel = batcher.RegisterEvaluator(Evaluator(&preparations));
  auto good = InputFrom(Bind(&x));
  channel.AddObservation(good, 1);
  std::vector<Input> invalid;
  auto bad = good; bad.parameters.sizes.pop_back(); invalid.push_back(bad);
  bad = good; bad.parameters.blocks.pop_back(); invalid.push_back(bad);
  bad = good; bad.parameters.blocks[0] = nullptr; invalid.push_back(bad);
  bad = good; bad.parameters.sizes[0] = -1; invalid.push_back(bad);
  bad = good; bad.AddParameter(&x, 1); invalid.push_back(bad);
  for (const auto& input : invalid) {
    bool failed = false;
    try { channel.AddObservation(input, 0); } catch (const std::invalid_argument&) { failed = true; }
    Check(failed && channel.PendingObservationCount() == 1 && channel.PendingBatchCount() == 1,
          "Invalid lightweight input bypassed validation");
  }
  for (double scale : {0., -1., std::numeric_limits<double>::infinity()}) {
    bool failed = false;
    try { channel.AddObservation(good, 0, scale); } catch (const std::invalid_argument&) { failed = true; }
    Check(failed, "Inline cache hit accepted invalid loss scale");
  }
  good.context.offset = 5;
  channel.AddObservation(good, 2, 4); // incompatible context: use the fresh input
  good.context.offset = 0;
  channel.AddObservation(good, 3);   // original context remains intact
  auto other = InputFrom(Bind(&y));
  other.support_signature.push_back(1);
  channel.AddObservation(other, 4);
  channel.AddObservation(good, 5);   // table hit, not last_batch
  auto broken = good; broken.context.fail = true;
  bool failed = false;
  try { channel.AddObservation(broken, 0); } catch (const std::runtime_error&) { failed = true; }
  Check(failed && channel.PendingObservationCount() == 5, "Fallback preparation failure appended a row");
  Check(preparations == 6, "A lightweight cache hit skipped preparation");
  batcher.Commit(); Near(registry.Residuals(), {9, 6, 7, 5, 16});
  failed = false;
  try { channel.AddObservation(good, 0); } catch (const std::logic_error&) { failed = true; }
  Check(failed, "Committed channel accepted lightweight input");

  Registry fallback_registry;
  splbatch::ProblemBatcher fallback(fallback_registry);
  auto& unadapted = fallback.RegisterEvaluator(UnadaptedEvaluator());
  unadapted.AddObservation(good, 1);
  good.context.offset = 2;
  unadapted.AddObservation(good, 2);
  fallback.Commit(); Near(fallback_registry.Residuals(), {9, 6});
}

void InlineAllocationAndRollback() {
  double x = 10, y = 20;
  auto a = InputFrom(Bind(&x)), b = InputFrom(Bind(&y));
  splbatch::EvaluatorChannel<Evaluator> channel(Evaluator(), {});
  for (int i = 0; i < 65; ++i) channel.AddObservation(a, i);
  auto count = allocation_fault::calls;
  Input temporary; temporary.AddParameter(&x, 1);
  channel.AddObservation(temporary, 65);
  Check(count == allocation_fault::calls, "Inline construction/compatible append allocated");
  channel.AddObservation(b, 0);
  count = allocation_fault::calls;
  channel.AddObservation(a, 66);
  Check(count == allocation_fault::calls, "Inline table hit allocated");
  for (int mode = 0; mode < 4; ++mode) {
    for (long fail_at = 0; fail_at < 64; ++fail_at) {
      Registry registry;
      splbatch::ProblemBatcher batcher(registry);
      auto& current = batcher.RegisterEvaluator(Evaluator());
      if (mode != 0) current.AddObservation(a, 1, 4);
      if (mode == 3) current.AddObservation(b, 2, 9);
      auto target = (mode == 1 || mode == 3) ? a : b;
      if (mode == 1) target.context.offset = 5;
      const auto n = current.PendingObservationCount(), batches = current.PendingBatchCount();
      bool failed = false;
      allocation_fault::remaining = fail_at;
      try { current.AddObservation(target, 3, 16); } catch (const std::bad_alloc&) { failed = true; }
      allocation_fault::remaining = -1;
      if (failed) {
        Check(current.PendingObservationCount() == n && current.PendingBatchCount() == batches,
              "Inline allocation failure published a partial observation/batch");
        current.AddObservation(target, 3, 16);
      }
      batcher.Commit();
      if (mode == 0) Near(registry.Residuals(), {68});
      if (mode == 1) Near(registry.Residuals(), {18, 8});
      if (mode == 2) Near(registry.Residuals(), {18, 68});
      if (mode == 3) Near(registry.Residuals(), {18, 28, 54});
      if (!failed) break;
      Check(fail_at < 63, "Did not exhaust inline allocation failures");
    }
  }
}
} // namespace

int main() {
  try {
    OrderContextAndLifetime(); OrderContextAndLifetime(true); SupportAndValidation();
    HashCollision(); HashCollision(true);
    FailureRollback(); AllocationFreeHit(); CompressedChannel(); IncrementalKeyTransitions();
    InlineInputStorage(); InlineValidationAndContext(); InlineAllocationAndRollback();
    std::cout << "PASS ordinary ingestion: ordering, context, support, validation, collision, rollback, allocation-free hit\n";
  } catch (const std::exception& e) {
    allocation_fault::remaining = -1;
    std::cerr << e.what() << '\n'; return 1;
  }
}
