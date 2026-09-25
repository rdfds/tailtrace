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

