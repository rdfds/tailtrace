# TailTrace

**An auditable scheduler and experiment lab for the slowest training rank.**

[![CPU correctness](https://github.com/rdfds/tailtrace/actions/workflows/ci.yml/badge.svg)](https://github.com/rdfds/tailtrace/actions/workflows/ci.yml)

TailTrace couples a variable-length causal transformer, global-batch-preserving workload
planning, DDP/FSDP2, a custom CUDA residual-RMSNorm operator, and trace analysis. It asks:
*does a proposed optimization reduce synchronized step latency while preserving the
training objective?* Ray runs training on a cluster; Spark aggregates rank metrics.

Version 0.2 adds heterogeneous compute calibration, capacity-aware beam scheduling,
and an exact small-batch oracle. The [384-case audit](results/scheduling-audit/README.md)
retains every input, assignment, failure, and search budget. Open its
[offline inspector](results/scheduling-audit/report.html) after cloning to explore
physical-rank layouts and proven proxy regret.

Version 0.3 adds immutable text shards, a real rank-crash recovery audit, and an
incremental scheduler. [Paired CPU measurements](results/planner-cpu/README.md) show
**2.76×–13.07× faster local search with identical assignments** on four tested sizes.
The [process recovery experiment](results/recovery-cpu/README.md) kills one rank after
a checkpoint and checks bitwise equality of the final model and full AdamW state.

Version 0.4 adds bounded durable telemetry and storage-sealed checkpoint recovery.
The [interrupted-write experiment](results/fault-recovery-cpu/README.md) recovers from
both a partial DCP write and a corrupted checkpoint, reproducing all model/AdamW state
bitwise. [Separate logical-node agents](results/logical-nodes-cpu/README.md) also match
the reference. [Measured report memory](results/telemetry-memory-cpu/README.md) remains
near **0.18 MiB of Python allocations across 2,000–100,000 generated rank rows**;
exact percentile aggregation uses disk storage. These are CPU systems results.

This is experimental systems software. CUDA and multi-node results are **not yet
validated on NVIDIA hardware**. The repository includes executable CPU experiments,
correctness tests, and hardware validation instructions.

## Why this project

Equal sample counts do not mean equal attention work. Dense padded attention costs roughly
`local_samples × local_max_sequence_length²`. TailTrace redistributes samples **within the
same shuffled global batch**, allowing unequal local counts. A global valid-token
denominator preserves the loss and averaged distributed gradient. The planner includes
the random baseline among its candidates, guaranteeing no worse predicted tail cost.
The fleet planner adds per-rank quadratic/linear/overhead terms and declared shape caps.
An independent certificate checks conservation and feasibility; the exact oracle proves
optimality only when its search completes. Predicted cost remains separate from measured
speedup. See [scheduler guarantees and calibration scope](docs/scheduler.md).

The interesting artifact is an evidence chain: workload → rank-local measurements →
trace occupancy → paired intervention report. Long NCCL duration alone does not identify
a network bottleneck; another rank may have arrived late. Cross-host clocks are never
assumed synchronized.

## Run it

The scheduling audit runs with standard Python and no training dependencies:

```bash
pip install -e .
tailtrace schedule-audit --seeds 17 23 31 43 47 59 61 73 --out runs/audit
# Open runs/audit/report.html
```

For actual training and objective correctness:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,dev]'
pytest -m 'not cuda and not ray and not spark'
tailtrace demo --out runs/demo
torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 \
  -m tailtrace train --config configs/cpu.json --out runs/ddp-cpu
tailtrace experiment --config configs/cpu-baseline.json \
  --intervention configs/interventions/balance.json \
  --seeds 17 23 31 --workers 2 --out runs/balance
```

Open `runs/balance/report.html` to inspect paired latency ratios and the seed bootstrap
interval. Each run includes raw rank metrics, config, hardware/software metadata, a
source fingerprint, and a summary. The driver records failures and refuses uncontrolled
comparisons. Use a new output directory for each run.

The offline demo needs only `pip install -e .` and labels its events **synthetic**.
Training requires the `train` extra. Ray and Spark are optional dependencies; importing
the offline tools does not load any of these runtimes.

## What is implemented

| Area | Implementation |
|---|---|
| Workload intervention | Heterogeneous compute models, capacity-aware beam construction, moves/swaps, conserved global batches |
| Scheduling audit | Independent feasibility certificates, budgeted exact minimax oracle, six-method adversarial ablations |
| Calibration | Isolated compute collection, median replicates, nonnegative fits, shape-separated held-out validation |
| Real data | UTF-8 byte shards, read-only memory maps, overlapping windows, target accounting, content identities |
| Distributed training | Causal transformer, token-weighted loss, DDP/FSDP2, bf16, activation checkpointing, torchrun/Slurm launch |
| CUDA | C++/CUDA residual-RMSNorm forward/backward, fp32 accumulation, deterministic two-pass weight reduction, current-stream support |
| Recovery | Storage-sealed DCP, compatible fallback selection, actual partial-write/corruption audits, exact model/AdamW recovery |
| Long-run telemetry | Bounded rank journals, durable byte frontiers, crash inspection, streaming summaries with exact disk-backed quantiles |
| Profiling | Kineto, NVTX, Nsight SQLite adapter, interval unions, exposed collective occupancy and copy/compute overlap |
| Experiments | Randomized paired arms, protocol/hardware/source guards, bootstrap across independent seeds, offline HTML reports |
| Cluster orchestration | Ray Train shares the training loop; separate logical torchrun agents verified on one CPU host |
| Experiment warehouse | Spark validates rank coverage and aggregates separate runs into Parquet |

## Evidence, not promises

The CPU correctness suite passes, including an actual two-rank Gloo job and exact
checkpoint recovery. The actual two-worker Ray Train CPU test also passes, including
the interpreter path with spaces. The [CPU smoke observation](results/cpu-laptop/README.md) is
inconclusive: this workload does **not** establish a performance advantage. No GPU
speedup, scaling efficiency, cost saving, or research priority is claimed.

CPU, Ray, and Spark jobs passed in [the baseline GitHub CI run](https://github.com/rdfds/tailtrace/actions/runs/37553657508).
Current CI additionally reproduces the entire oracle campaign and uploads fresh CPU
calibration evidence. CUDA/FSDP2 and multi-node execution need NVIDIA validation.
Component status and launch notes are documented in [validation](docs/validation.md).

The v0.3 and v0.4 updates are validated locally; hosted CI was intentionally skipped
to avoid spending compute minutes. The badge reflects the last hosted run.
The text and recovery experiment runs entirely on a CPU laptop:

```bash
tailtrace corpus-build examples/corpus --max-length 128 --out runs/text-corpus
tailtrace recovery-audit --config configs/cpu-recovery.json --fault-step 2 --out runs/recovery
```

The deeper failure and logical-node experiments are also entirely local:

```bash
tailtrace fault-audit --config configs/cpu-recovery.json --out runs/fault-proof
tailtrace node-audit --config configs/cpu-recovery.json --out runs/node-proof
tailtrace inspect-run runs/fault-proof/failed --out runs/fault-proof/inspection.json
tailtrace metrics-bench --out runs/report-memory
```

Local validation records **116 CPU tests plus the actual two-worker Ray Train test**.
The [retained Ray run](results/ray-streaming-cpu/README.md) exercises per-step journal
flushes and a sealed DCP save. [Raw validation logs](results/validation-v04/README.md)
record the tested commands and source identity.

Read [durability, compatibility, and instrumentation contracts](docs/reliability.md).
The bounded telemetry default marks mid-loop flushes as instrumentation. Short paired
throughput configs explicitly defer collection with `metrics_flush_every=0`.

See [reader and failure contracts](docs/corpus-and-recovery.md) for the exact tested
failure window, immutable-data rules, and shared-filesystem requirements for Ray.

The beam scheduler matches the proven proxy optimum in **358 of 358** completed feasible
searches; it finds feasible assignments in all 374 cases not proven infeasible. Ten cases
are proven infeasible and sixteen searches hit their node budget. One completed case has
17.31% regret in the single-basin beam ablation; retaining the local incumbent removes that regression. These results establish behavior under declared models, not hardware
speedups or an unrestricted optimality guarantee.

```bash
# NVIDIA machine, after correctness tests pass:
torchrun --standalone --nproc-per-node=2 -m tailtrace train \
  --config configs/gpu-ddp.json --out runs/gpu
tailtrace kernel-bench --rows 4096 --width 1024 --out runs/kernel.json
```

The [manual GPU workflow](.github/workflows/gpu.yml) runs kernel checks and compares
DDP/FSDP2 gradients and SGD updates against the global objective on two GPUs. The
[research roadmap](docs/research-roadmap.md) identifies experiments and limitations
that would turn the current lab into a stronger research contribution.

## Profiling a claim

The [profiling protocol](docs/validation.md#serious-profiling-protocol) separates
uninstrumented throughput from diagnostic traces and Nsight Compute counters.
`tailtrace analyze` reads a rank-local Chrome trace; `tailtrace nsys-import` converts
a read-only Nsight SQLite export. GPU stream overlap is counted once. NCCL waiting
is not assumed to be network transfer time. Cross-host arrival times are not compared
without clock alignment.

## Prior art and scope

[Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) already
provides distributed trace analysis. [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
and [Ray Train](https://docs.ray.io/en/latest/train/overview.html) provide sharding and
orchestration. TailTrace's proposed contribution is their integration with a constrained
workload intervention and correctness/evidence gates, not inventing these systems.
Research novelty is a hypothesis requiring broader review and real experiments.

[LB-BSP](https://arxiv.org/abs/1806.02508), [Hydraulis](https://arxiv.org/abs/2412.07894), and
[Zeppelin](https://arxiv.org/abs/2509.21841) already address heterogeneous batching or
variable-length workload imbalance. TailTrace focuses on checkable assignments,
oracle gaps, objective equivalence, and reproducible evidence. See the
[0.2 release notes](docs/release-v0.2.md) for the new interfaces and compatibility scope.
The [0.3 release notes](docs/release-v0.3.md) cover text, recovery, and planner overhead.
The [0.4 release notes](docs/release-v0.4.md) cover failure protocols and bounded reporting.

MIT licensed. See [architecture](docs/architecture.md) and [validation](docs/validation.md).
