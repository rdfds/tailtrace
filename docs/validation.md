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

