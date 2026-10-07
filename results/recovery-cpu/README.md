# Observed process failure and exact recovery

An actual two-rank Gloo job on one CPU host trains on twelve variable-length byte
documents. Rank 1 exits with code 86 immediately after completing checkpoint 2,
inside a three-step epoch. The launcher fails; a fresh job restarts at global step 2
and finishes at step 6. Its final state is compared with an uninterrupted reference.

| Check | Observed result |
|---|---|
| Uninterrupted / restarted launcher | Both exit 0 |
| Injected rank exit | Rank 1 exits 86; launcher fails |
| Complete checkpoint after crash | Step 2 marker exists |
| Completed run manifest for failed job | Absent |
| Restarted global step | 2 |
| Final global step | 6 |
| Model and AdamW tensor comparisons | 36 tensors, 40,041 elements |
| Maximum tensor difference | 0; bitwise equality |
| Parameter groups / learning rate / optimizer step | Identical |

Canonical state SHA-256:
`148c81af5c8c2957e5c5da6787b1334d1db7d671bab7f2b6a775b3dd2a0e4f86`.

[audit.json](audit.json) records exact commands, exit statuses, dataset identity,
source fingerprint, host metadata, comparison, and failure scope. The raw logs are
[whole.log](whole.log), [crash.log](crash.log), and [resume.log](resume.log). Full final
and crash-point distributed checkpoints, raw rank metrics, manifests, and the sealed
[corpus](corpus/manifest.json) are retained. Generated absolute paths describe the
machine that executed this run; use the reproduction commands with fresh local paths.
The published-checkpoint test reloads both final states and verifies this hash again.

```bash
tailtrace corpus-build examples/corpus --max-length 128 --out runs/text-corpus
tailtrace recovery-audit --config configs/cpu-recovery.json --fault-step 2 --out runs/recovery
```

The measurements establish deterministic **CPU process recovery after a completed
checkpoint**, at the same world size. They do not establish machine-loss recovery,
mid-write recovery, elastic recovery, FSDP2/CUDA correctness, model quality, or a
throughput advantage. Read [the full failure contract](../../docs/corpus-and-recovery.md).
