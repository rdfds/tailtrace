# Research questions and engineering roadmap

The current result is a functioning experiment lab with explicit hardware gaps. It is
not yet a demonstrated novel research system or a production training platform.

## Experiments that would make it compelling

| Question | Controlled intervention | Evidence needed |
|---|---|---|
| When does unequal local cardinality reduce the distributed tail? | Random vs balanced with identical global sample IDs | Five–ten seed pairs, length regimes, compute/kernel trace attribution |
| Does a fused norm improve distributed training rather than just a microbenchmark? | Reference vs CUDA norm | Gradient tolerances, shapes/dtypes, Nsight Compute counters, training latency |
| When do sharding savings pay for extra collectives? | DDP vs FSDP2 at the same model size | Memory curve, collective overlap, 2/4/8 GPUs, same workloads |
| Are apparent communication stalls actually workload skew? | Fixed delayed rank vs no delay | Per-rank NVTX/compute/NCCL occupancy, independent throughput capture |
| Does planner overhead dominate short sequences? | Same planner at varied sequence scales | Planner wall time and end-to-end time alongside step timing |

The negative CPU smoke result is useful: a plausible quadratic cost model does not
guarantee a benefit on a small, launch/communication-sensitive workload. Larger GPUs
could show different behavior, but that must be measured.

## Subsequent engineering work

* Calibrate a cost model using isolated forward/backward timings, including linear MLP
  work and rank-dependent GPU speeds. Do not fit pure compute from NCCL-inclusive steps.
* Add real tokenized document shards and verify sample conservation across resumable
  readers. Keep the synthetic workload for deterministic diagnosis.
* Add packed/variable-length attention as a separately controlled intervention. It changes
  the padded-attention cost model; do not silently keep the current proxy.
* Verify FSDP2 mixed precision and sharded checkpoint recovery on multi-node hardware.
* Optimize the norm using warp reductions/vectorization and benchmark against compiled
  PyTorch and existing fused implementations, including adverse shapes.
* Expand the Nsight adapter with versioned real fixtures and runtime/kernel correlation
  for queue delays. Do not assume cross-host clocks share an origin.
* Implement bounded streaming metrics for long jobs and robust crash manifests.
