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
