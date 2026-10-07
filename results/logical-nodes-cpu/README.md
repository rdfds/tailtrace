# Two logical node agents: CPU equivalence proof

The reference uses one torchrun agent with two local workers. The comparison uses
**two independently launched torchrun agents**, each with `--nnodes=2`, a distinct
`--node-rank`, and one local worker, sharing an explicit IPv4 rendezvous endpoint.
Both logical nodes run on the same physical laptop.

Observed topology:

| Global rank | Local rank | Group rank | Local world size |
|---|---|---|---|
| 0 | 0 | 0 | 1 |
| 1 | 0 | 1 | 1 |

After six byte-corpus training steps, all model and AdamW state is bitwise equal to
that of the reference: 36 tensors, 40,041 elements, zero maximum error. The canonical
state hash is `148c81af5c8c2957e5c5da6787b1334d1db7d671bab7f2b6a775b3dd2a0e4f86`.

```bash
tailtrace corpus-build examples/corpus --max-length 128 --out runs/text-corpus
tailtrace node-audit --config configs/cpu-recovery.json --out runs/node-proof
pytest tests/test_published_reliability.py
```

[audit.json](audit.json), separate node logs, final distributed checkpoints, metrics,
progress markers, and source/config metadata are retained. `corpus/` contains the
immutable input. The published test reloads and independently compares the states.

This verifies separate agent rendezvous, group/local rank mapping, and the unchanged
global objective on one CPU host. It does **not** verify physical multi-host networking,
NCCL, GPU execution, cross-host clocks, or scaling efficiency.
