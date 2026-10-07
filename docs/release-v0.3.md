# TailTrace 0.3: immutable text, process recovery, and planner overhead

The same global objective now trains on real UTF-8 byte documents. `corpus-build`
streams overlapping windows into memory-mapped token/index shards, seals content
identities, and accounts for omitted short tails. Training validates corpus content
on every rank; checkpoints and paired comparisons reject changed dataset identities.
The FP64 DDP equivalence test covers real documents with random, balanced, and fleet
ownership, including unequal local sample counts. Synthetic diagnostics remain available.

`recovery-audit` launches an uninterrupted two-rank CPU job, deliberately exits rank 1
after a committed save, then restarts the failed job. Final model and complete AdamW
state must be bitwise identical. The included fixture crashes inside an epoch and
crosses an epoch boundary after restart. Raw logs, commands, metrics, and state proof
are retained. Completion markers sync their data and POSIX directory entries.

The local move/swap evaluator caches rank shapes, avoids whole-assignment copies,
and evaluates only changed ranks. It preserves reference traversal, tie selection,
and floating-point sum order. Four paired CPU microbenchmarks show median runtime
ratios of 2.76×, 5.69×, 9.25×, and 13.07× with identical assignments. This measures
local search only; it does not establish a GPU training speedup. Epoch zero no longer
plans twice, and initial/epoch planning wall times are recorded separately.

The full 384-case scheduling campaign remains identical. The CPU correctness suite
and lint checks are run locally; these pushes skip hosted CI to use no paid compute.
Ray and Spark integration were validated in the earlier hosted v0.2 run; the new
Ray dataset-path resolution has not been exercised on a remote cluster. CUDA/FSDP2
and real multi-node evidence remain outstanding.

Compatibility: `dataset_path` is optional. Existing synthetic configurations continue
to run; old synthetic checkpoints have no dataset identity and load under the same
configuration. Corpus checkpoints require the same path/configuration and sealed
contents. Recovery still requires the same world size. Byte vocabularies have size 256.

Read [the full data and fault contract](corpus-and-recovery.md),
[CPU planner measurements](../results/planner-cpu/README.md), and
[the recovery proof](../results/recovery-cpu/README.md).
