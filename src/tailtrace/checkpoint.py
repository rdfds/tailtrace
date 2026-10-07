from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from tailtrace.checkpoint_index import check_compatible, seal, verify
from tailtrace.evidence import atomic_json


def save(path, model, optimizer, config, world, step, rank, fleet_sha256=None, dataset_sha256=None):
    path = Path(path)
    if (path / "complete.json").exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    model_state, optim_state = get_state_dict(model, optimizer)
    dcp.save(
        {"model": model_state, "optimizer": optim_state},
        storage_writer=dcp.FileSystemWriter(path, overwrite=False),
    )
    # Metadata is the commit marker, written only after every rank's shards are durable.
    if world > 1:
        dist.barrier()
    error = None
    if rank == 0:
        try:
            atomic_json(
                path / "complete.json",
                seal(
                    path,
                    {
                        "step": step,
                        "world_size": world,
                        "torch_version": str(torch.__version__),
                        "config": config.to_dict(),
                        "fleet_sha256": fleet_sha256,
                        "dataset_sha256": dataset_sha256,
                    },
                ),
            )
        except (ValueError, OSError) as exc:
            error = str(exc)
    errors = [error]
    if world > 1:
        dist.broadcast_object_list(errors, src=0)
    if errors[0]:
        raise ValueError("checkpoint commit failed: " + errors[0])


def restore(path, model, optimizer, config, world, fleet_sha256=None, dataset_sha256=None):
    path = Path(path)
    metadata, error = None, None
    try:
        metadata = verify(path)
        check_compatible(
            metadata,
            config,
            world,
            fleet_sha256,
            dataset_sha256,
            require_remaining=False,
            torch_version=str(torch.__version__),
        )
    except (ValueError, KeyError, TypeError, OSError) as exc:
        error = str(exc)
    status = {"error": error, "identity": metadata.get("sha256") if metadata else None}
    statuses = [status]
    if world > 1 and dist.is_initialized():
        statuses = [None] * world
        dist.all_gather_object(statuses, status)
    errors = [s["error"] for s in statuses if s["error"]]
    if errors:
        raise ValueError("checkpoint preflight failed: " + "; ".join(errors))
    if len({s["identity"] for s in statuses}) != 1:
        raise ValueError("ranks loaded different checkpoint contents")
    model_state, optim_state = get_state_dict(model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(
        model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"]
    )
    # The model has no stochastic layers. Data and epoch permutations are sample-ID seeded;
    # resuming a global step exactly restores the next batch without RNG state snapshots.
    return metadata["step"]
