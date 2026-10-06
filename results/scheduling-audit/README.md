# Small-batch oracle audit

This is an **analytic proxy experiment**, not a GPU benchmark. The committed
`campaign.json` contains all 384 cases and six scheduling methods. `report.html` is
a self-contained interactive inspector. The manifest pins the generating source and
command; hostname is pseudonymized. Inputs and assignments are deterministic.

The campaign crosses six length families, four rank fleets, two batch sizes/topologies,
and eight seeds. Two-rank batches contain six samples; three-rank batches contain twelve.
The rank fleets include homogeneous costs, speed skew, quadratic/linear crossovers,
and tight padded-token/sample caps. Seeds 47, 59, 61, and 73 extend the initial tuning
seeds; the same workload families remain visible, so this is not a blind benchmark.

| Method | Feasible assignments / 384 | Cases with proven optimum | Matches optimum | Worst proven regret |
|---|---:|---:|---:|---:|
| Random equal counts | 312 | 296 | 32 | 1897.72% |
| Legacy balanced | 307 | 291 | 109 | 465.56% |
| Fleet greedy | 349 | 333 | 160 | 218.46% |
| Fleet greedy + local moves/swaps | 349 | 333 | 269 | 141.40% |
| Fleet beam, single search basin | 374 | 358 | 357 | 17.31% |
| Fleet beam + retained local incumbent | 374 | 358 | 358 | 0.00% |

The oracle completed 358 feasible searches, proved ten cases infeasible, and exhausted
its 100,000-node budget on sixteen cases. Beam construction found feasible assignments
in all 374 cases not proven infeasible. This does **not** prove feasibility completeness
in general or optimality in the sixteen unfinished searches. Methods have different
coverage, so their regret columns describe different feasible subsets.

The single-basin beam ablation regresses on `log_uniform/tight_capacity/w3/seed31`:
its estimated makespan is 17.31% above the exact optimum of 12,548 proxy units, while
local search from the greedy construction reaches that optimum. A better construction
can enter a worse local-search basin. Retaining both independently refined paths removes
this regression and guarantees no worse proxy objective than local-only search.

Zero proven regret on these small completed cases is not a universal guarantee. The
sixteen budget-limited searches have no proven optimality result. The oracle bound applies
only to this monotone padded compute model and declared constraints.

```bash
pip install -e .
tailtrace schedule-audit --seeds 17 23 31 43 47 59 61 73 \
  --node-budget 100000 --out runs/scheduling-audit
```

The generated campaign payload should match the committed JSON exactly. The manifest's
clock, platform, and source revision may differ. Hardware timing, GPU memory feasibility,
and multi-node scaling require separate experiments. See [scheduler design](../../docs/scheduler.md).
