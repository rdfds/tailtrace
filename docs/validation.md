# Validation and reproducibility

## Status

| Component | Local evidence | Remaining validation |
|---|---|---|
| Planner, interval algebra, experiment guards | Unit tests and 384-case oracle campaign | GPU cost-model accuracy and larger batches |
| Fleet calibration / recovery | Isolated CPU collection and content guards supplied | Hardware fit stability and actual fleet performance |
| Causal transformer / token objective | CPU gradients passed | Mixed-precision accuracy on GPU |
| Real two-rank DDP | Gloo fp64 gradients + AdamW updates passed | NCCL multi-node run |
| Checkpoint recovery | Exact CPU next update and mid-epoch resume passed | FSDP2 shard recovery on GPU |
| Process crash / real text | Actual after-save and mid-write exits, corrupt-checkpoint fallback, bitwise model/AdamW proofs | Machine loss, remote storage and elastic recovery |
| Telemetry / reporting | Durable crash frontiers, bounded buffering, exact streamed quantiles, measured memory growth | Long GPU runs and remote storage behavior |
| Logical node agents | Two independent torchrun node agents, bitwise CPU state equivalence | Physical multi-host networking and NCCL |
| Incremental local search | 120 differential cases and four paired CPU timing cases | GPU training overhead/benefit |
| Kineto profiling | Actual CPU trace export and analysis passed | Real CUDA trace capture |
| Nsight SQLite adapter | Generated schema fixtures passed | Versioned NVIDIA-exported fixtures |
| CUDA norm / FSDP2 | Implemented; manual GPU gate supplied | **No NVIDIA hardware validation yet** |
| Ray orchestration | Actual two-worker CPU Ray Train test passed | Multi-node GPU Ray validation |
| Spark aggregation | Java 17 GitHub CI passed | Large warehouse / remote storage validation |

The baseline [GitHub run](https://github.com/rdfds/tailtrace/actions/runs/37553657508)
passed CPU, Ray, and Spark jobs. The current workflow also reproduces all 384 scheduling
cases and retains CPU calibration artifacts. GPU validation remains separate; a passing
CPU workflow does not validate CUDA kernels or FSDP2.

Versions 0.3 and 0.4 are validated locally and skip hosted CI to avoid paid compute minutes.
The badge and earlier hosted run do not certify this version. Raw local evidence is
in [planner measurements](../results/planner-cpu/README.md) and
[process recovery](../results/recovery-cpu/README.md); the published recovery test
reloads the retained final distributed checkpoints and reproduces the state hash.

The v0.4 reliability proofs retain raw logs and checkpoints and are independently
reloaded by `tests/test_published_reliability.py`. See [failure protocols](reliability.md). The [local validation record](../results/validation-v04/README.md)
retains the 116-test CPU suite and separate two-worker Ray test. The
[Ray run](../results/ray-streaming-cpu/README.md) enables per-step flushes and a sealed save.

## Run the local correctness suite

```bash
pip install -e '.[train,dev]'
ruff check src tests
ruff format --check src tests
pytest -m 'not cuda and not ray and not spark'
```

`tests/test_distributed.py` launches actual processes through torchrun. CPU runs on
macOS use explicit IPv4 master addressing; PyTorch's `--standalone` hostname discovery
can produce an unresolvable reverse-DNS name on some machines.

## Run a controlled experiment

```bash
tailtrace experiment --config configs/cpu-baseline.json \
  --intervention configs/interventions/balance.json \
  --seeds 17 23 31 --workers 2 --out runs/balance-cpu
```

The driver randomizes arm order within each seed, stores commands' output in logs,
records failure state, validates paired manifests, and writes JSON plus an offline
HTML report. It runs locally without cloud provisioning. For multi-node comparisons,
launch each arm with the same cluster placement and then use `tailtrace compare`.

```bash
tailtrace compare --baseline runs/base-17 runs/base-23 runs/base-31 \
  --candidate runs/test-17 runs/test-23 runs/test-31 --out runs/comparison.json
```

The checked-in [CPU smoke observation](../results/cpu-laptop/README.md) shows an
inconclusive result. It is not a CUDA performance result or a published research finding.

## CUDA correctness first

Use Linux, NVIDIA GPUs, PyTorch 2.8.0 with a compatible CUDA build, nvcc, and a C++ compiler.
Install `ninja` for JIT compilation. The CUDA toolkit must match the PyTorch build;
installation of this project does not install a toolkit or drivers.

```bash
pip install -e '.[train,dev]' ninja
pytest tests/test_cuda.py -m cuda
torchrun --standalone --nproc-per-node=2 tests/gpu_equivalence_worker.py runs/gpu-equivalence.json
tailtrace kernel-bench --rows 4096 --width 1024 --dtype float32 --out runs/kernel-fp32.json
tailtrace kernel-bench --rows 4096 --width 1024 --dtype bfloat16 --out runs/kernel-bf16.json
```

Run irregular widths, multiple leading dimensions, fp16/bf16/fp32, and non-default CUDA
streams. Run NVIDIA Compute Sanitizer on the kernel tests to detect invalid memory
access and race conditions. The manual GitHub workflow targets a self-hosted two-GPU
runner; hosted CPU runners cannot validate CUDA.

`kernel-bench` checks outputs and all first-order gradients before reporting timings.
It excludes compilation and warmup and reports forward and forward+backward distributions.
Its reference is eager PyTorch, not compiled PyTorch or every existing fused library.
Winning this microbenchmark is not sufficient to claim a training improvement.

## GPU training and multi-node execution

```bash
torchrun --standalone --nproc-per-node=2 -m tailtrace train \
  --config configs/gpu-ddp.json --out runs/ddp
torchrun --standalone --nproc-per-node=2 -m tailtrace train \
  --config configs/gpu-fsdp2.json --out runs/fsdp2
```

These example configurations have **different model sizes** and cannot be used as a
paired DDP/FSDP2 comparison. Use the same baseline configuration plus
`configs/interventions/fsdp2.json` for a controlled strategy experiment. Likewise,
`configs/gpu-baseline.json` plus `balance.json` tests the planner. Device memory demands
depend on hardware and workload; reduce dimensions if needed and disclose the change.

For two nodes with four GPUs each, set a reachable `MASTER_ADDR` and the same free
`MASTER_PORT` on both nodes. Install identical dependencies/source and mount the same
output path. Each node launches one torchrun process with its own node rank:

```bash
# NODE_RANK=0 on the first node; NODE_RANK=1 on the second.
torchrun --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  -m tailtrace train --config configs/gpu-ddp.json --out /shared/tailtrace/run-001
```

[scripts/slurm_train.sh](../scripts/slurm_train.sh) implements the equivalent Slurm
launch. It is an example for an existing allocation, not an automatic cluster purchase.
For a measured scaling curve, keep a fixed global-token workload for strong scaling;
the default batch-size setting instead scales global sample count with world size.
Report which scaling regime you use, links/topology, GPU model/count, precision, and
the exact source/config hashes.

## Serious profiling protocol

1. Establish correctness and an uninstrumented multi-seed throughput baseline.
2. Collect separate short Kineto/Nsight traces with `configs/profile-gpu.json`.
3. Attribute local compute/communication overlap; inspect delayed-rank injection by
   setting `delay_rank` and `delay_ms`. Delay injection is diagnostic, not a benchmark.
4. Test one intervention and rerun uninstrumented paired experiments.
5. Inspect memory, numerical differences, confidence intervals, and negative results.

```bash
torchrun --standalone --nproc-per-node=2 -m tailtrace train \
  --config configs/profile-gpu.json --out runs/profile
tailtrace analyze runs/profile/trace-rank0.json --rank 0 --out runs/profile/rank0-analysis.json

chmod +x scripts/nsys_rank.sh
torchrun --standalone --nproc-per-node=2 --no-python scripts/nsys_rank.sh \
  configs/profile-gpu.json runs/nsys
nsys export --type sqlite --output runs/nsys/rank0.sqlite runs/nsys/nsys-rank0.nsys-rep
tailtrace nsys-import runs/nsys/rank0.sqlite --out runs/nsys/rank0-chrome.json
tailtrace analyze runs/nsys/rank0-chrome.json --rank 0 --out runs/nsys/rank0-analysis.json
```

Prefer **either** Kineto or Nsight per diagnostic capture: set `profile=false` in a
copy of the Nsight configuration to avoid CUPTI subscriber conflicts. Keep `nvtx=true`
and `diagnostic_sync=true`. Throughput experiments should disable all three flags. For a short uninstrumented
telemetry capture, also set `metrics_flush_every=0`; this buffers the run and gives up
crash progress. The paired baseline configs set it explicitly. Bounded mid-loop flushes
are instrumentation and their wall time is recorded separately.

For individual kernels, use Nsight Compute after compiling the extension:

```bash
ncu --set full --kernel-name 'regex:.*forward_kernel.*' --launch-count 5 \
  --target-processes all -o runs/rmsnorm \
  tailtrace kernel-bench --rows 4096 --width 1024 --out runs/instrumented-kernel.json
```

Inspect register pressure, achieved occupancy, memory throughput, cache hit rates,
and warp stall reasons. Timings produced under Nsight Compute are instrumented and
must not be used as the uninstrumented benchmark. Sources:
[Nsight Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
[SQLite schema and serialized IDs](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).

## Ray and Spark

```bash
pip install -e '.[ray,dev]'
tailtrace ray-train --config configs/cpu.json --workers 2 --out runs/ray-cpu \
  --storage "$(pwd)/runs/ray-storage"
# Existing Ray cluster; train-extra dependencies must be installed on all worker nodes.
tailtrace ray-train --address auto --config configs/gpu-ddp.json --workers 8 \
  --out /shared/tailtrace/ray-run --storage /shared/ray-storage

pip install -e '.[spark,dev]'
# Spark 4 requires Java 17+; the CI job uses Java 17.
SPARK_LOCAL_IP=127.0.0.1 PYSPARK_SUBMIT_ARGS='--master local[2] pyspark-shell' \
  tailtrace spark-aggregate 'runs/*/metrics-rank*.jsonl' --out runs/warehouse
pytest tests/test_spark.py -m spark
```

The local Ray launch limits the object store to 128 MiB. Ray's default runtime environment
can fail to start Python workers when the interpreter path contains spaces. TailTrace
quotes the local `py_executable` override to address this launch issue. The actual
two-worker CPU test passed from this workspace's `cuda proj` path after that fix. Remote
cluster launches use the worker nodes' own Python environments.

Ray handles worker placement/process groups; it does not silently replace the global
batch planner with a DistributedSampler. Spark separates runs by source directory,
checks duplicate rank/step rows and global-token consistency, and partitions Parquet
by run. It aggregates records; it does not train the model. Cloud Spark users can read
the same JSONL from a supported URI. Input globs must be quoted so Spark expands them.
