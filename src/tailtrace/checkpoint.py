from __future__ import annotations

import json
from pathlib import Path

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from tailtrace.evidence import atomic_json


def save(path, model, optimizer, config, world, step, rank):
    path = Path(path)
    if (path / "complete.json").exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=path)
    # Metadata is the commit marker, written only after every rank's shards are durable.
    if world > 1:
        dist.barrier()
    if rank == 0:
        atomic_json(
            path / "complete.json",
            {"schema_version": 1, "step": step, "world_size": world, "config": config.to_dict()},
        )
    if world > 1:
        dist.barrier()


