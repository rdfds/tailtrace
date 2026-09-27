# Validation and reproducibility

## Status

| Component | Local evidence | Remaining validation |
|---|---|---|
| Planner, interval algebra, experiment guards | Unit tests passed | Larger heterogeneous workloads |
| Causal transformer / token objective | CPU gradients passed | Mixed-precision accuracy on GPU |
| Real two-rank DDP | Gloo fp64 gradients + AdamW updates passed | NCCL multi-node run |
| Checkpoint recovery | Exact CPU next update and mid-epoch resume passed | FSDP2 shard recovery on GPU |
| Kineto profiling | Actual CPU trace export and analysis passed | Real CUDA trace capture |
| Nsight SQLite adapter | Generated schema fixtures passed | Versioned NVIDIA-exported fixtures |
| CUDA norm / FSDP2 | Implemented; manual GPU gate supplied | **No NVIDIA hardware validation yet** |
| Ray orchestration | Actual two-worker CPU Ray Train test passed | Multi-node GPU Ray validation |
| Spark aggregation | CI test supplied | Local install blocked by disk space |

CI workflows are configuration, not evidence of a successful remote CI run. No remote
repository is configured in the initial workspace.

## Run the local correctness suite

```bash
pip install -e '.[train,dev]'
ruff check src tests
ruff format --check src tests
pytest -m 'not cuda and not ray and not spark'
```

`tests/test_distributed.py` launches actual processes through torchrun. CPU runs on
macOS use explicit IPv4 master addressing; PyTorch's `--standalone` hostname discovery
can produce an unresolvable reverse-DNS name on some machines.

## Run a controlled experiment

```bash
tailtrace experiment --config configs/cpu-baseline.json \
  --intervention configs/interventions/balance.json \
  --seeds 17 23 31 --workers 2 --out runs/balance-cpu
```

The driver randomizes arm order within each seed, stores commands' output in logs,
records failure state, validates paired manifests, and writes JSON plus an offline
HTML report. It runs locally without cloud provisioning. For multi-node comparisons,
launch each arm with the same cluster placement and then use `tailtrace compare`.

```bash
tailtrace compare --baseline runs/base-17 runs/base-23 runs/base-31 \
  --candidate runs/test-17 runs/test-23 runs/test-31 --out runs/comparison.json
```

The checked-in [CPU smoke observation](../results/cpu-laptop/README.md) shows an
inconclusive result. It is not a CUDA performance result or a published research finding.

