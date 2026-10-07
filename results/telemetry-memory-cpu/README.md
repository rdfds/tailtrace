# Exact report aggregation: measured memory growth

The inputs are **deterministically generated metrics**, not training observations.
The experiment measures actual CPU report aggregation in fresh Python subprocesses.
Both modes read identical two-rank JSONL files and produce identical aggregate values
and workload hashes. Retained mode stores per-step rows; aggregate mode streams rows
and uses SQLite on temporary disk for exact, linearly interpolated quantiles.

| Global steps | Rank rows | Aggregate Python peak MiB | Retained Python peak MiB | Aggregate process peak MiB | Retained process peak MiB |
|---|---|---|---|---|---|
| 1,000 | 2,000 | 0.184 | 0.565 | 21.12 | 21.67 |
| 10,000 | 20,000 | 0.185 | 4.449 | 21.16 | 27.27 |
| 50,000 | 100,000 | 0.185 | 21.763 | 21.23 | 53.78 |

Across a 50× increase in steps, aggregate Python allocations stay near 0.18 MiB.
At 50,000 global steps, retaining rows uses 117.83× as much peak Python allocation.
The whole-process RSS reduction is smaller; Python allocation ratios must not be
presented as total RAM savings. Raw observations, input sizes/hashes, environment,
source identity, and commands are in [benchmark.json](benchmark.json).

```bash
tailtrace metrics-bench --sizes 1000 10000 50000 --out runs/report-memory
tailtrace summarize runs/ddp-cpu --aggregate-only --out runs/ddp-cpu/aggregate.json
```

The command regenerates every input row from its deterministic formula. Generated
input files are left locally in the experiment output; their hashes are retained here
without committing redundant large fixtures. The published test regenerates those
files and verifies the recorded hashes and aggregates.

`tracemalloc` excludes SQLite C allocations and OS page cache. Fresh-process peak RSS
includes the interpreter and SQLite. SQLite has a 1 MiB page-cache target, uses disk
for temporary sorting, and needs disk space proportional to step count. This is a
memory-growth experiment with single observations, not a latency speedup, a universal
RAM upper bound, or a GPU result.
