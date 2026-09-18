# Architecture and invariants

```mermaid
flowchart LR
    C[Seeded workload + config] --> P[Shared global batch plan]
    P --> R[torchrun or Ray workers]
    R --> T[DDP / FSDP2 transformer]
    T --> K[Reference or CUDA residual RMSNorm]
    T --> M[Rank metrics + provenance]
    T --> N[Kineto / NVTX / Nsight]
    N --> A[Interval union and overlap]
    M --> E[Paired seed experiment]
    A --> H[Offline evidence report]
    E --> H
    M --> S[Spark Parquet warehouse]
```

## Workload planning

For each epoch, every rank independently generates the same deterministic permutation
of sample IDs. A global batch contains `world_size × batch_size` IDs. The random,
length-clustered, and variable-cardinality greedy plans are candidates for assigning
those same IDs. Every rank receives at least one sample and at most `2 × batch_size`.
Incomplete global batches are dropped; the manifest records their count.

For a right-padded causal transformer, the attention proxy for a rank is
`B × (max(lengths) − 1)²`. Equal-cardinality planners cannot reduce the maximum proxy:
the rank with the longest sequence always executes `batch_size × longest²` work.
Unequal cardinality is the essential intervention. The greedy planner seeds each rank
with a long sequence, then assigns remaining samples using the smallest resulting
maximum proxy. The final plan minimizes `(maximum cost, total cost)` among all three
candidates. Including the random candidate proves **proxy non-regression**, not global
optimality or real-time improvement.

Padding, MLP cost, vocabulary projections, kernel selection, collective timing, and
launch overhead complicate the relationship between this proxy and wall time. The
planner is homogeneous: it does not currently calibrate different GPU speeds. Its
sample-count bound does not prove a model will fit in GPU memory.

## Objective preservation

Let `Sᵣ` be the sum of valid-token cross-entropies on rank `r`, `W` the world size,
and `T` the valid-token count across the **global** batch. Each rank backpropagates
`Lᵣ = W × Sᵣ / T`. DDP/FSDP average gradients:

`(1/W) × Σᵣ ∇Lᵣ = ∇(Σᵣ Sᵣ / T)`.

The denominator is known from the deterministic plan, so no normalization all-reduce
is needed per step. Averaging local mean losses would be wrong for unequal local
token counts. Right padding and causal attention ensure valid queries cannot attend
to future padding. Padded labels are ignored. The model has no dropout or batch
normalization, whose randomness/statistics would complicate equivalence.

Mathematical equivalence does not imply bitwise fp32 equivalence. Batch shapes change
floating-point reduction order, and AdamW can amplify errors near zero. Tests cover
fp32 gradients and real two-rank fp64 AdamW updates. The manual GPU gate covers fp32
SGD gradients/updates for both distributed strategies and both norm implementations.

## Distributed execution and recovery

`torchrun` owns rank/device setup and NCCL/Gloo process groups. Ray Train supplies an
already initialized process group; the same training loop reuses it. FSDP2 shards each
transformer block and then the root, before constructing the optimizer. Activation
checkpointing can be enabled as an explicit intervention.

Distributed Checkpoint stores canonical model and optimizer state. `complete.json` is
written after all shard writes finish; incomplete directories cannot be resumed.
Recovery requires the same world size and workload/model settings. It restores the
global step and recomputes epoch permutations. Synthetic tokens are sample-ID seeded,
so the model's deterministic execution needs no mutable dataset cursor or dropout RNG.
Elastic world-size recovery and automatic failure retries are outside the current scope.

Rank zero collects JSONL metrics and hardware metadata. A shared filesystem is required
for multi-node checkpoints and trace files. Metrics/timers are buffered for a bounded
experiment and collected at the end; this is not a streaming production trainer.

