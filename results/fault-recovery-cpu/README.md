# Interrupted-write and corruption recovery proof

This is an observed two-rank CPU experiment, with all raw checkpoint files, rank
journals, progress markers, configs, logs, and the immutable text corpus retained.
The driver executes an uninterrupted six-step reference, a failed job, and a restart.

1. Both ranks commit checkpoints after steps 1, 2, and 3.
2. Rank 1 exits with code 88 **inside an actual PyTorch DCP shard write**, immediately
   after writing and syncing 4096 bytes for checkpoint 4. The launcher fails; there
   is no checkpoint-4 completion marker or final training manifest.
3. After that job exits, the driver flips byte zero of a committed checkpoint-3 shard.
   The audit records its original and altered bytes and SHA-256 values.
4. Automatic selection rejects incomplete checkpoint 4 and corrupt checkpoint 3,
   selects verified checkpoint 2, and records every decision.
5. Restarted training finishes at step 6. Every model tensor, AdamW moment/variance/
   step tensor, parameter group, learning rate, and global step matches the reference
   **bitwise**: 36 tensors, 40,041 elements, zero maximum error.

Durable telemetry retains the four completed training steps through zero-based step 3.
That coverage is diagnostic; it is not a committed optimizer checkpoint or a throughput
result. The optimizer restarts from completed step 2 and replays subsequent work.

Canonical final state: `148c81af5c8c2957e5c5da6787b1334d1db7d671bab7f2b6a775b3dd2a0e4f86`.

```bash
tailtrace corpus-build examples/corpus --max-length 128 --out runs/text-corpus
tailtrace fault-audit --config configs/cpu-recovery.json --out runs/fault-proof
tailtrace inspect-run runs/fault-proof/failed --out runs/fault-proof/inspection.json
pytest tests/test_published_reliability.py
```

[audit.json](audit.json) retains exact commands and source/config identities. Recorded
absolute paths describe the original run; the test reloads the retained files relative
to this repository. `corpus/` preserves the same content identity as that run.

Scope: one physical CPU host, Gloo/DDP, fp32, unchanged world size. These are actual
process/write failures and deliberately injected corruption. They do not establish
power-loss or remote-filesystem durability, elastic recovery, FSDP2, or GPU correctness.
