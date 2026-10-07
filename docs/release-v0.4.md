# Version 0.4: fault-safe CPU training and bounded telemetry

The earlier recovery experiment terminated a rank after a completed save and retained
metrics only at successful job completion. Version 0.4 retains committed diagnostic
progress during jobs and verifies the optimizer state independently of that progress.

* Bounded rank journals seal durable byte frontiers; crash inspection ignores torn
  uncommitted suffixes and rejects changed committed prefixes.
* Training exchanges constant-size metadata at completion. Automatic summaries and
  paired comparisons stream metrics and compute exact percentiles using disk storage.
* Distributed checkpoints seal every storage file. Automatic recovery skips incomplete,
  corrupt, and incompatible candidates; every rank agrees before tensor loading.
* An actual 4096-byte partial DCP write followed by injected checkpoint corruption falls
  back to checkpoint 2 and reproduces the reference's full model/AdamW state bitwise.
* Two independent logical node agents reproduce single-agent two-worker state exactly.
* Fresh-process report measurements over 2,000–100,000 generated rank rows document
  memory growth and independently equal aggregate results.
* Real distributed tests exercise mismatched source/configuration, failed journal
  creation, and a rank-0 checkpoint commit error, requiring rejection on both ranks.

New commands: `fault-audit`, `node-audit`, `metrics-bench`, `inspect-run`;
new options: `train --resume-latest ROOT`, `summarize --aggregate-only`.
`metrics_flush_every=32` is the bounded default; zero defers telemetry for short
throughput diagnostics and relinquishes crash-progress and buffering guarantees.
Paired throughput baseline configs explicitly choose zero. Mid-loop flushes are
instrumentation and are recorded as such.

Training `summary.json` now contains aggregate values with `steps=[]`,
`steps_retained=false`, and an explicit quantile method. Offline summaries still retain
steps by default. The two modes use identical step/token workload hashes. New checkpoint
markers use schema 2; schema 1 remains explicitly restorable but cannot be selected
through `--resume-latest`.

Raw evidence: [checkpoint faults](../results/fault-recovery-cpu/README.md),
[logical nodes](../results/logical-nodes-cpu/README.md),
[report memory](../results/telemetry-memory-cpu/README.md).
Detailed contracts: [reliability](reliability.md).

The local validation record contains 116 passing CPU tests plus the actual Ray
two-worker integration test with per-step flushes and a sealed checkpoint. A further
published-artifact check validates the retained Ray journals and checkpoint inventory.

All new experiments use local CPU compute. Hosted CI remains intentionally skipped;
the existing badge does not certify this release. CUDA/FSDP2, physical multi-host
execution, power-loss durability, and GPU performance remain hardware validation work.
