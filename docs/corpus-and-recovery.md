# Immutable text and process recovery

The byte corpus adds real UTF-8 text to the same causal objective used by the synthetic
diagnostic workload. It needs no downloaded tokenizer, dataset, or service. Each source
document is validated as UTF-8; its bytes become token IDs 0–255. This is a byte language
model, not a subword model or a quality benchmark. Use text you are permitted to process.

```bash
tailtrace corpus-build examples/corpus --max-length 128 --out runs/text-corpus
tailtrace recovery-audit --config configs/cpu-recovery.json --fault-step 2 --out runs/recovery
```

The twelve included documents are original MIT-licensed systems notes with variable
lengths. With two ranks and two local samples, there are three global steps per epoch.
The crash occurs after step 2, inside the first epoch; the restarted run crosses the
next epoch and finishes at step 6.

## Reader and objective contract

`tokens.bin` holds contiguous byte windows; `index.bin` holds fixed-width little-endian
`(uint64 offset, uint32 length, uint32 source_document)` entries. Both are read-only
memory maps. The manifest seals their SHA-256 hashes, document hashes, target accounting,
and schema. Opening checks hashes and index extents before training. Ordinary file
size/mtime changes while a reader is open are rejected. This is an immutable-file
contract, not protection against an adversary preserving metadata or changing files
between validation and mapping. Treat shards as read-only during jobs.

Windows overlap by one byte to preserve every next-byte target. Short final windows
below `min_length` are omitted; the manifest accounts for dropped targets. No target
crosses a document boundary. The builder reads 64 KiB blocks; token data is not loaded
wholesale into RAM. Training loads selected sample lengths for planning and materializes
only each local batch. Each rank hashes the full shard once and scans its index on open.

Set `dataset_path` and `vocab_size=256`; `samples` selects the first indexed sample IDs.
Those IDs are shuffled deterministically by epoch, with incomplete global batches
dropped as in the synthetic workload. Model `max_length` must cover every selected
sequence; synthetic length sampling is unused. Right padding masks targets with `-100`;
every rank uses the same global valid-token denominator. FP64 two-rank tests compare
gradients and AdamW updates with a single global objective for random, balanced, and
fleet ownership of both random tokens and byte documents.

Ranks exchange content identities and validation errors before model collectives.
Checkpoints seal the dataset hash. Paired experiments require identical dataset
identities within a pair and across seeds. A changed corpus is rejected before loading
checkpoint tensors. Ray resolves the driver path to an absolute path; remote nodes must
mount the same immutable shard there. Ray uploads code, not arbitrary training data.

## Failure experiment

The audit launches three actual `torchrun` jobs on one CPU host: an uninterrupted
reference; a job whose rank 1 calls `os._exit(86)` immediately after the completed
distributed save; and a restart from that checkpoint. It requires a failed launcher,
an explicit injected-exit marker, a committed checkpoint, and no final run manifest
for the failed job. Supervisor timeouts kill the worker process group.

The audit loads final distributed checkpoints and compares every model tensor, AdamW
moment/variance/step tensor, parameter group, learning rate, and global step. It requires
bitwise equality and records a canonical state hash. Logs, exact commands, generated
configs, checkpoint commit markers, and raw rank metrics remain in the output directory.
JSON commit markers flush and sync their file and, on POSIX, parent directory before
reporting success.

This validates process failure **after a completed save**, at the same world size. It
does not test failure during writes, physical machine loss, elastic world sizes, network
partitions, CUDA, or FSDP2. The short text fixture establishes recovery correctness,
not model quality or speed.

Version 0.4 separately tests loss inside a checkpoint write and corruption of a
committed shard. It adds durable metrics and automatic verified fallback; see
[reliability contracts](reliability.md) and [the retained proof](../results/fault-recovery-cpu/README.md).
The scope above describes the original after-save experiment.

## Planner timing

Initial planning (including fleet validation) and subsequent epoch planning have
separate manifest timings.
Epoch planning is included in training-loop wall time but excluded from step timing.
Epoch zero reuses its initial plan. `planner-bench` measures the incremental move/swap
evaluator against the v0.2 full-rescoring implementation with randomized paired order,
raw repetitions, and exact assignment checks. Its scope is local search on the CPU,
not end-to-end GPU training.
