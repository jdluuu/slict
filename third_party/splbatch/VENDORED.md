# SPLBATCH snapshot

Source: `/home/jiadong/workspace/ClampSLAM/src/ClampedSLAM/third_party/splbatch`.
Upstream: `https://github.com/jdluuu/splbatch`, commit `e2892937c356b34d2873c91cea9b044a6c9d897a` (0.1.1).
The snapshot initially copied the public headers unchanged; CMake exposes a header-only target.
See LICENSE, NOTICE.md and upstream.json for license and CT-RIO provenance.
SLICT-specific evaluators are kept outside this directory.

Local adaptation, 2026-09-07: added `residual_registry.hpp` and changed only the
registration boundary in `problem_batcher.hpp`. Existing Ceres constructors and
direct channel commits remain supported; a custom registry can own and evaluate
the same batch cost objects with SLICT's native solver. Grouping, evaluator and
batch evaluation algorithms are unchanged. `upstream.json` identifies the base
snapshot, not these local modifications. See `docs/native_batch_adaptation.md`
in the SLICT repository for the adapter and measurements.

Construction diagnostics: added `construction_diagnostics.hpp` and opt-in
probes in `problem_batcher.hpp`. `SPLBATCH_CONSTRUCTION_DIAGNOSTICS` is defined
only for a separate diagnostic library/executable; normal builds erase all
probes. Batch numerical evaluation is unchanged. See
`docs/native_batch_construction_profile.md` for measurement boundaries and probe
perturbation; diagnostic stage estimates are not uninstrumented timings.

P1 construction optimization: ordinary `EvaluatorChannel` now compares the last
support and uses a hash-to-candidate-index table without per-observation owning
keys. `batch_binding.hpp` adds stateless internal lookup helpers; its existing
Key, equality, hash and validation implementations are unchanged. Preparation
uses the incoming binding on every append, and failed appends roll back paired
storage or an unpublished batch. Incremental grouping and numerical evaluation
headers are unchanged. See `docs/splbatch_p1_results.md` for scope and results.

`tests/test_incremental.cpp` is copied unchanged from the same source tree as
the vendored headers. It is registered as a SLICT CTest alongside local ingestion
and incremental-key transition regression tests in
`tests/splbatch_ingestion_test.cpp` (the SLICT repository's tests directory).

Lightweight input adaptation: added `inline_binding_input.hpp` with owned inline
metadata and dynamic overflow, generalized the P1 comparison helper to different
input representations, and added an ordinary-channel AddObservation overload.
An explicit per-evaluator `BindingPreparationReuse` contract permits use of the
stored owning Binding only when preparation semantics match; otherwise input
context is materialized for the unchanged preparation API. The default does not
reuse. Existing owning Binding and incremental paths retain their interfaces.
SLICT opts in locally for its dt-only preparation context; no evaluator or batch
numerical code is changed. See `docs/splbatch_light_binding_results.md`.
