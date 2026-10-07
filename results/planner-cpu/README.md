# Observed CPU local-search overhead

The incremental evaluator performs the same strict best move/swap search as the v0.2
full-rescoring implementation. It caches rank shapes and recomputes only the two changed
ranks. Traversal order and physical-rank summation order are preserved, including ties.

Nine randomized paired repetitions per case, three search rounds, seed 17:

| Ranks | Samples | Full rescoring median | Incremental median | Median ratio |
|---:|---:|---:|---:|---:|
| 2 | 8 | 0.476 ms | 0.173 ms | 2.76× |
| 4 | 32 | 20.060 ms | 3.528 ms | 5.69× |
| 8 | 64 | 282.105 ms | 30.488 ms | 9.25× |
| 8 | 128 | 1247.168 ms | 95.449 ms | 13.07× |

[benchmark.json](benchmark.json) retains generated lengths, rank models, starting and
final assignments, randomized execution order, every raw duration, source fingerprint,
and host metadata. Each timed result is checked for exact agreement with the reference.
There are 36 paired observations, not 36 independent workloads. Ratios are descriptive
medians on one host; no population confidence interval is claimed.

```bash
tailtrace planner-bench --repeats 9 --rounds 3 --seed 17 --out runs/planner-bench
```

An additional 120 differential cases cover cardinality caps, attention caps, duplicate
maximum lengths, and non-integer coefficients. The full published 384-case oracle
campaign is reproduced separately and its complete JSON payload is unchanged.

These are **CPU local-search timings**. They exclude beam construction, oracle solving,
training, network collectives, and GPU work. The largest measured case still takes
95 ms; faster scheduling does not imply that this planner is appropriate for every
short training step. Use initial/epoch planning and training-loop wall timings in the
run manifest when assessing end-to-end overhead.
