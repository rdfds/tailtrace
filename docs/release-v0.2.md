# TailTrace 0.2: auditable fleet scheduling

This release adds heterogeneous rank costs, declared shape constraints, isolated
forward/backward calibration, a bounded beam scheduler, and an exact small-batch oracle.
The shared training loop retains global samples and the token-weighted objective under
unequal local batch sizes. Profile content hashes guard recovery and paired comparisons.

The [384-case audit](../results/scheduling-audit/README.md) includes raw inputs, six
scheduler ablations, feasibility certificates, search budgets, and an interactive offline
inspector. Beam scheduling reaches the proven optimum in 358 of 358 completed feasible
searches. Ten cases are proven infeasible; sixteen searches are unfinished. The single-basin beam ablation retains a 17.31% regression;
the default planner avoids it by preserving its local-search incumbent. These are analytic scheduling results, not GPU performance results.

```bash
pip install -e .
tailtrace schedule-audit --out runs/audit
# Actual isolated calibration needs the training extra:
pip install -e '.[train,dev]'
tailtrace calibrate --config configs/cpu-baseline.json --out runs/calibration
```

The default audit uses four seeds; use the eight-seed command in the result README for
the exact published payload. A fleet profile contains one ordered cost model per physical
rank. `configs/cpu-fleet.json` is a two-rank example with declared coefficients, not an
observed hardware calibration. See [scheduler design](scheduler.md) before using a fit.

Existing random and balanced configuration files remain valid. New configurations expose
`fleet_path`; outputs include fleet identity and cost scope. Checkpoints bind profile
content. The `v0.1.0` source tag preserves the original CPU smoke experiment's package
fingerprint; that observation validates the baseline implementation, not this release.

NVIDIA CUDA kernels, FSDP2/NCCL execution, and multi-node performance remain unverified.
Observed isolated unsharded fits are rejected for FSDP2 because they omit sharding effects.
Declared proxies remain available for exploratory FSDP2 schedules. Claims about production
speedups, GPU-memory safety, or research priority require further evidence.
