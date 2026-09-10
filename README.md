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

