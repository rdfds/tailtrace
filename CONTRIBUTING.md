# Contributing

Run `pip install -e '.[train,dev]'`, `ruff check src tests`, `ruff format --check src tests`,
and `pytest -m 'not cuda and not ray and not spark'` before proposing changes. Optional
Ray/Spark tests and NVIDIA validation are separate because their runtimes are expensive
or hardware-dependent.

Changes to planning must preserve every global batch, keep all ranks nonempty, bound
local sample counts, and retain valid-token gradient scaling. Changes to timing must
disclose synchronization and instrumentation. Kernel changes require outputs and all
first-order gradients across irregular shapes and dtypes; performance alone is insufficient.

Benchmark submissions should include raw metrics, config/source hashes, device/topology,
warmup rules, independent seeds, and the exact launch commands. Distinguish observed,
synthetic, inferred, and unvalidated claims. Preserve negative results. Do not publish
fabricated measurements.
