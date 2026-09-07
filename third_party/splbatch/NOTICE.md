# Source and license

SPLBATCH originates from the `ct_batch` implementation in CT-RIO, under
`src/crepes3/crepes/include/ct_batch`. The initial extraction renames the
namespace and include prefix from `ct_batch` to `splbatch`; the numerical
implementations are retained. `upstream.json` records the original file hashes
and the hashes after this mechanical rename. The original headers were
untracked in that working tree, so the recorded CT-RIO commit alone does not
identify their contents.

The GPL version 3 license text supplied with CT-RIO is preserved in `LICENSE`.
Retain this source notice and the license when redistributing the extracted
code. Packaging, documentation and examples added for SPLBATCH use the same
license.

`tests/test_scaled_loss.cpp` is adapted from CT-RIO's
`tests/test_ct_batch_scaled_loss.cpp`. `tests/test_incremental.cpp` extracts the
portable scalar-model tests from CT-RIO's
`tests/test_incremental_problem_batcher.cpp`. Tests of CT-RIO-specific sensor
models and its custom Ceres backend remain in that application.

This repository contains the batching infrastructure. Application-specific
LiDAR, IMU, camera and marginalization models are supplied by the calling
project through the evaluator interface.
