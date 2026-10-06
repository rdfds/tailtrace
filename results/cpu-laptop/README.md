# CPU smoke observation — October 6, 2026

**Inconclusive. This is observed CPU data, not a GPU benchmark.**

A two-process Gloo experiment compared the random and balanced planners on the same
global batches at seeds 17, 23, and 31. Each arm trained for 20 steps, discarding the
first four as warmup. Model: two blocks, width 32, four heads, vocabulary 256; 128
synthetic samples with lengths 8–64; nominal local batch size four. One CPU thread per
rank. No profiler, NVTX, diagnostic sync, or checkpoint writes were enabled.

| Seed | Random median step (ms) | Balanced median step (ms) | Ratio (random / balanced) |
|---|---:|---:|---:|
| 17 | 3.543 | 4.358 | 0.813× |
| 23 | 4.963 | 3.455 | 1.437× |
| 31 | 3.563 | 3.956 | 0.901× |

Geometric mean ratio: **1.017×**. Seed bootstrap 95% interval: **0.813–1.437×**.
The interval includes 1. These results do not establish a planner advantage. Millisecond
CPU workloads on a shared laptop are noisy; three pairs cannot characterize all seeds
or quantify GPU behavior. The quadratic proxy and actual timing are different measures.

All six runs started from clean source revision
`4def1133b003aa958cf5989ce58f9c09f8aa5072`, with identical package source SHA-256
`34d2c7a42169b987a06cc7f4921a20834c06648945d0d908defd7cbe685f2da5`.
Arm order was deterministically randomized within each seed. Actual timestamps,
configs, software/platform versions, raw rank metrics, and run order are retained.
Hostnames are consistent pseudonyms and local paths are made relative in this public
export. Timing/token/loss values have not been changed. Logs remain in the original
ignored local run directory and are omitted from this public export.

Files: [comparison JSON](comparison.json), [offline HTML report](report.html),
[experiment metadata](experiment.json), and six directories with manifests and raw
`metrics-rank*.jsonl` records.

## Reproduce or inspect

From the repository root:

```bash
tailtrace experiment --config configs/cpu-baseline.json \
  --intervention configs/interventions/balance.json \
  --seeds 17 23 31 --workers 2 --out runs/reproduction

# Recompute this report from the retained raw measurements:
tailtrace compare \
  --baseline results/cpu-laptop/seed-17-baseline results/cpu-laptop/seed-23-baseline results/cpu-laptop/seed-31-baseline \
  --candidate results/cpu-laptop/seed-17-candidate results/cpu-laptop/seed-23-candidate results/cpu-laptop/seed-31-candidate \
  --out runs/recomputed.json
```

Reproduction timings will differ. The package source in this repository matches the
recorded SHA-256 above. A second
earlier local smoke run at a dirty development snapshot also had an interval spanning
1 (about 0.90–1.15×); it is not used for this clean-snapshot report.
