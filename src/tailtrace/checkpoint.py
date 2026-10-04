from __future__ import annotations

import json
from pathlib import Path

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from tailtrace.evidence import atomic_json


def save(path, model, optimizer, config, world, step, rank, fleet_sha256=None):
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
            {
                "schema_version": 1,
                "step": step,
                "world_size": world,
                "config": config.to_dict(),
                "fleet_sha256": fleet_sha256,
            },
        )
    if world > 1:
        dist.barrier()


def restore(path, model, optimizer, config, world, fleet_sha256=None):
    path = Path(path)
    if not (path / "complete.json").exists():
        raise ValueError("checkpoint has no complete.json commit marker")
    metadata = json.loads((path / "complete.json").read_text())
    if metadata["world_size"] != world:
        raise ValueError("checkpoint recovery currently requires the same world size")
    if metadata.get("fleet_sha256") != fleet_sha256:
        raise ValueError("checkpoint fleet profile content changed")
    mutable = {
        "steps",
        "warmup",
        "profile",
        "nvtx",
        "profile_steps",
        "checkpoint_every",
        "diagnostic_sync",
    }
    for key, value in metadata["config"].items():
        if key not in mutable and config.to_dict().get(key) != value:
            raise ValueError(f"checkpoint configuration changed: {key}")
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(
        model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"]
    )
    # The model has no stochastic layers. Data and epoch permutations are sample-ID seeded;
    # resuming a global step exactly restores the next batch without RNG state snapshots.
    return metadata["step"]
