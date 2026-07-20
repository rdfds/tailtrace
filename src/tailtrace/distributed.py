from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup(config):
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        # Ray restricts CUDA_VISIBLE_DEVICES per worker; torchrun exposes all local GPUs.
        device_index = 0 if torch.cuda.device_count() == 1 else local_rank
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)
    else:
        device = torch.device("cpu")
    if config.strategy == "single" and world != 1:
        raise ValueError("strategy=single cannot run with multiple ranks")
    owned = False
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=5)
        )
        owned = True
    if dist.is_initialized():
        world, rank = dist.get_world_size(), dist.get_rank()
    else:
        rank = 0
    if config.strategy == "fsdp2" and not dist.is_initialized():
        raise ValueError("launch FSDP2 with torchrun and at least two ranks")
    return device, rank, world, owned


