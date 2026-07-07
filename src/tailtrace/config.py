from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 17
    samples: int = 256
    min_length: int = 16
    max_length: int = 128
    batch_size: int = 4
    steps: int = 20
    warmup: int = 3
    width: int = 64
    heads: int = 4
    layers: int = 2
    vocab_size: int = 256
    lr: float = 0.001
    strategy: str = "ddp"
    planner: str = "balanced"
    kernel: str = "reference"
    device: str = "cpu"
    precision: str = "fp32"
    bucket_cap_mb: int = 25
    activation_checkpointing: bool = False
    profile: bool = False
    nvtx: bool = False
    profile_steps: int = 3
    diagnostic_sync: bool = False
    checkpoint_every: int = 0
    threads: int = 1
    delay_rank: int = -1
    delay_ms: float = 0.0

