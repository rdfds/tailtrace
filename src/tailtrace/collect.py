"""Rank-local forward/backward calibration, with collectives outside timed regions."""

from __future__ import annotations

import contextlib
import random
import time
from pathlib import Path


def collect(config, batches, lengths, repeats, warmup, out):
    import torch
    import torch.distributed as dist

    from tailtrace.calibration import fit_rank
    from tailtrace.distributed import checkpoint_activations, setup
    from tailtrace.evidence import atomic_json, provenance
    from tailtrace.fleet import workload_signature
    from tailtrace.model import CausalTransformer, make_batch, token_loss

    if repeats < 3 or warmup < 1:
        raise ValueError("require >=3 repeats and >=1 warmup per shape")
    if len(set(batches)) != len(batches) or len(set(lengths)) != len(lengths):
        raise ValueError("duplicate calibration shapes")
    if any(b < 1 for b in batches) or any(n < 2 or n > config.max_length for n in lengths):
        raise ValueError("invalid calibration shape")
    if len(batches) * len(lengths) < 8:
        raise ValueError("require at least eight distinct shapes")
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"calibration output already exists: {out}")
    device, rank, world, owned = setup(config)
    try:
        torch.set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        # Deliberately unwrapped: DDP/NCCL wait is not an isolated compute target.
        model = checkpoint_activations(CausalTransformer(config).to(device), config)
        shapes = [(b, n) for b in batches for n in lengths]
        random.Random(config.seed).shuffle(shapes)
        rows = []
        for batch, length in shapes:
            x, y = make_batch(
                list(range(batch)), [length] * batch, config.vocab_size, config.seed, device
            )
            for repeat in range(-warmup, repeats):
                model.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                begin = time.perf_counter()
                amp = (
                    torch.autocast(device.type, dtype=torch.bfloat16)
                    if config.precision == "bf16"
                    else contextlib.nullcontext()
                )
                with amp:
                    loss, _ = token_loss(model(x), y, batch * (length - 1), 1)
                loss.backward()
                if device.type == "cuda":
                    end.record()
                    end.synchronize()
                    ms = start.elapsed_time(end)
                else:
                    ms = (time.perf_counter() - begin) * 1000
                if repeat >= 0:
                    rows.append({"batch": batch, "width": length - 1, "repeat": repeat, "ms": ms})
        hardware = {
            **provenance(),
            "rank": rank,
            "device": str(device),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda,
        }
        # No collective inside a measured shape. All ranks independently finish first.
        gathered = [None] * world
        payload = {"rows": rows, "hardware": hardware}
        if world > 1:
            dist.all_gather_object(gathered, payload)
        else:
            gathered = [payload]
        if rank == 0:
            out.mkdir(parents=True)
            raw = {
                "schema_version": 1,
                "evidence": "observed_isolated",
                "scope": "forward_backward_no_collectives",
                "timing": "cuda_event" if device.type == "cuda" else "cpu_wall",
                "config": config.to_dict(),
                "batches": batches,
                "lengths": lengths,
                "repeats": repeats,
                "warmup_per_shape": warmup,
                "ranks": gathered,
            }
            atomic_json(out / "observations.json", raw)
            # Retain observations even if a rank's fit is nonidentifiable or poor.
            profile = {
                "schema_version": 1,
                "evidence": "observed_isolated",
                "scope": raw["scope"],
                "units": "ms",
                "workload": workload_signature(config),
                "hardware": [p["hardware"] for p in gathered],
                "ranks": [fit_rank(p["rows"], config.seed) for p in gathered],
            }
            atomic_json(out / "fleet.json", profile)
            return profile
        return {}
    finally:
        if owned:
            dist.destroy_process_group()
