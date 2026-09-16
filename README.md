# TailTrace

**A controlled experiment lab for the slowest rank in distributed training.**

TailTrace couples a variable-length causal transformer, global-batch-preserving workload
planning, DDP/FSDP2, a custom CUDA residual-RMSNorm operator, and trace analysis. It asks:
*does a proposed optimization reduce synchronized step latency while preserving the
training objective?* Ray runs training on a cluster; Spark aggregates rank metrics.

This is experimental systems software. CUDA and multi-node results are **not yet
validated on NVIDIA hardware**. The repository includes executable CPU experiments,
correctness tests, and hardware validation instructions.

## Why this project

Equal sample counts do not mean equal attention work. Dense padded attention costs roughly
`local_samples × local_max_sequence_length²`. TailTrace redistributes samples **within the
same shuffled global batch**, allowing unequal local counts. A global valid-token
denominator preserves the loss and averaged distributed gradient. The planner includes
the random baseline among its candidates, guaranteeing no worse predicted tail cost.
Predicted cost is a proxy, not a measured speedup or an optimality certificate.

The interesting artifact is an evidence chain: workload → rank-local measurements →
trace occupancy → paired intervention report. Long NCCL duration alone does not identify
a network bottleneck; another rank may have arrived late. Cross-host clocks are never
assumed synchronized.

## Run it

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
| Workload intervention | Deterministic minimax proxy planner with unequal local batch sizes, global sample conservation, and bounded cardinality |
| Distributed training | Causal transformer, token-weighted loss, DDP/FSDP2, bf16, activation checkpointing, torchrun/Slurm launch |
| CUDA | C++/CUDA residual-RMSNorm forward/backward, fp32 accumulation, deterministic two-pass weight reduction, current-stream support |
| Recovery | Distributed model/optimizer checkpoints, atomic completion markers, deterministic mid-epoch resume |
| Profiling | Kineto, NVTX, Nsight SQLite adapter, interval unions, exposed collective occupancy and copy/compute overlap |
| Experiments | Randomized paired arms, protocol/hardware/source guards, bootstrap across independent seeds, offline HTML reports |
| Cluster orchestration | Ray Train reuses the same loop and global planner; local object store is bounded |
| Experiment warehouse | Spark validates rank coverage and aggregates separate runs into Parquet |

## Evidence, not promises

The CPU correctness suite passes, including an actual two-rank Gloo job and exact
checkpoint recovery. The actual two-worker Ray Train CPU test also passes, including
the interpreter path with spaces. The [CPU smoke observation](results/cpu-laptop/README.md) is
inconclusive: this workload does **not** establish a performance advantage. No GPU
speedup, scaling efficiency, cost saving, or research priority is claimed.

CUDA/FSDP2 and multi-node execution need NVIDIA validation. Spark's local installation
was blocked by disk space; its test is supplied in a separate Java 17 CI job. CI has
not been executed remotely. Ray's validation status and launch notes are documented
in [validation](docs/validation.md).

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

MIT licensed. See [architecture](docs/architecture.md) and [validation](docs/validation.md).
