# Scheduling with checkable evidence

TailTrace optimizes a declared rank-local compute model while preserving each shuffled
global batch. It does not repartition attention across ranks or alter the optimizer's
global token objective. This deliberately narrow scope makes assignments auditable.

For rank `r`, a group of `B` samples with maximum next-token width `T` has estimated
cost `a[r] × B × T² + b[r] × B × T + c[r]`. Coefficients are nonnegative. Every rank
must receive at least one sample. Optional caps bound local sample count, padded tokens,
and padded attention cells. These are declared shape constraints, not measured GPU
memory limits; parameter, optimizer, allocator, and collective storage also matter.

## Planner and oracle

The fleet planner retains a feasible equal-count random assignment, constructs greedy
alternatives, and for at most 32 samples / eight ranks runs a width-64 constructive beam.
Partial beam states with the same ordered `(count, maximum width)` are cost-equivalent
under this model and can be merged. Physical rank identities never rotate. The planner
then accepts strict minimax improvements through sample moves and pair swaps, breaking
ties by total estimated work. It caches epoch plans outside timed training steps.

Including a feasible random baseline guarantees estimated makespan non-regression against
that baseline. It does not guarantee a feasible assignment will be found whenever one
exists, global optimality, or faster GPU execution. Larger batches use greedy construction
and local improvement without beam expansion; evaluate their planning overhead separately.

The small-batch oracle enumerates assignments to labeled ranks, rejecting capacity and
nonempty violations and pruning partial objectives dominated by a feasible incumbent.
It supports at most 18 samples / four ranks and a node budget. Completed searches return
`optimal` or `infeasible`; truncated searches return `budget_exhausted`, including any
incumbent and a conservative analytic lower bound. Never describe a truncated incumbent
as optimal. The certificate independently checks sample multisets, tokens, and caps.

The analytic bound relaxes padding and assignment coupling. Each sample requires at least
its cheapest feasible singleton cost; the makespan also exceeds average unpadded work
plus all unavoidable rank overheads. Both are lower bounds because coefficients are
nonnegative and every rank is nonempty. Independent brute-force tests check the oracle
and these bounds with heterogeneous coefficients and capacity limits.

## Calibration and scope

```bash
torchrun --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29500 \
  -m tailtrace calibrate --config configs/cpu-baseline.json \
  --batches 1 2 4 8 --lengths 8 16 32 64 --repeats 7 --out runs/calibration
```

The collector measures an unwrapped model's forward/backward computation, including the
configured activation recomputation, after per-shape warmup. It excludes optimizer work,
input construction, and collectives. CUDA uses events and CPU uses wall time. Each rank
collects independently, then gathers observations. Do not regress NCCL-inclusive step
time against a compute proxy: waiting for peers is part of that time and changes with
the assignment itself.

Median replicate times reduce isolated outlier influence. A three-feature nonnegative
least-squares fit enumerates active coefficient subsets; ill-conditioned designs are
rejected. Entire shapes are held out with a deterministic split. The exported profile
retains raw observations, shape domains, coefficients, held-out errors, hardware,
workload settings, and source identity. A small error on these shapes does not establish
general accuracy, causal speedup, or calibration stability under contention.

Set `planner=fleet` and `fleet_path=runs/calibration/fleet.json` in a copy of the same
configuration. Training rejects source/hardware/workload changes and shape extrapolation.
All ranks must load the same content hash; checkpoints and seed-paired experiments pin
that hash. Do not modify a profile between seed pairs. Inspect fit errors before choosing
an intervention. The `configs/fleet-declared.json` example is explicitly a declared proxy.

Observed unsharded fits are rejected for FSDP2 because resharding and mixed-precision
parameter policies alter execution. Declared fleet proxies can still drive exploratory
FSDP2 assignments; they require a separate hardware validation experiment. Ray uses the
same loop. Fleet files must be accessible at the same shared absolute path on all workers;
the Ray package upload does not transfer arbitrary calibration files.

## Reproduce the adversarial audit

```bash
pip install -e .
tailtrace schedule-audit --seeds 17 23 31 43 47 59 61 73 \
  --node-budget 100000 --out runs/scheduling-audit
```

This requires only Python. Open `report.html` to inspect regret, assignments, coefficients,
and search budgets. `campaign.json` retains every input and outcome, including unavailable
heuristics, proven infeasibility, and unfinished searches. Regret means
`candidate estimated makespan / proven optimal estimated makespan − 1`. It is neither a
throughput measurement nor a claim about the optimal real GPU schedule.

## Related systems and contribution boundary

[LB-BSP](https://arxiv.org/abs/1806.02508) already adapts worker batch sizes to heterogeneous
processing capabilities. [Hydraulis](https://arxiv.org/abs/2412.07894) jointly considers
parallel strategies and variable-length data assignment. [Zeppelin](https://arxiv.org/abs/2509.21841)
addresses variable-length workloads through attention partitioning, routing, and layout
remapping. TailTrace does not claim to introduce load balancing or unequal local batches.

Its engineering contribution is a compact chain from bounded cost assumptions to
certified assignments, exact-oracle regret, token-objective correctness, content-pinned
recovery, and paired observed evidence. Establishing a research contribution still needs
broader workload families, stronger comparator systems, and actual GPU/multi-node results.
