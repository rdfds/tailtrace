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
    fleet_path: str | None = None
    dataset_path: str | None = None
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

    def __post_init__(self):
        for name in ("activation_checkpointing", "profile", "nvtx", "diagnostic_sync"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("seed", "warmup", "checkpoint_every", "delay_rank"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        for name in ("lr", "delay_ms"):
            if (
                isinstance(getattr(self, name), bool)
                or not isinstance(getattr(self, name), (int, float))
                or not math.isfinite(getattr(self, name))
            ):
                raise ValueError(f"{name} must be a finite number")
        for name in (
            "samples",
            "min_length",
            "max_length",
            "batch_size",
            "steps",
            "width",
            "heads",
            "layers",
            "vocab_size",
            "bucket_cap_mb",
            "profile_steps",
            "threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_length < 2 or self.max_length < self.min_length:
            raise ValueError("require 2 <= min_length <= max_length")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if not 0 <= self.warmup < self.steps or self.checkpoint_every < 0:
            raise ValueError("require 0 <= warmup < steps and checkpoint_every >= 0")
        if self.lr <= 0 or self.delay_ms < 0:
            raise ValueError("lr must be positive and delay_ms nonnegative")
        for field, choices in {
            "strategy": {"single", "ddp", "fsdp2"},
            "planner": {"random", "balanced", "fleet"},
            "kernel": {"reference", "cuda"},
            "device": {"cpu", "cuda"},
            "precision": {"fp32", "bf16"},
        }.items():
            if getattr(self, field) not in choices:
                raise ValueError(f"{field} must be one of {sorted(choices)}")
        if self.strategy == "fsdp2" and self.device != "cuda":
            raise ValueError("FSDP2 requires CUDA in this project")
        if self.kernel == "cuda" and self.device != "cuda":
            raise ValueError("the CUDA kernel requires device=cuda")
        if self.fleet_path is not None and (
            not isinstance(self.fleet_path, str) or not self.fleet_path
        ):
            raise ValueError("fleet_path must be a nonempty string")
        if (self.planner == "fleet") != (self.fleet_path is not None):
            raise ValueError("planner=fleet and fleet_path must be supplied together")
        if self.dataset_path is not None:
            if not isinstance(self.dataset_path, str) or not self.dataset_path:
                raise ValueError("dataset_path must be a nonempty string")
            if self.vocab_size != 256:
                raise ValueError("byte corpora require vocab_size=256")

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
