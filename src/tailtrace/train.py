from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from tailtrace.config import TrainConfig
from tailtrace.distributed import setup, wrap_model
from tailtrace.evidence import atomic_json, provenance
from tailtrace.model import CausalTransformer, make_batch, token_loss
from tailtrace.planner import make_lengths, plan_epoch
from tailtrace.profiling import region


def run(config: TrainConfig, out: str | Path, resume: str | None = None) -> dict:
    out = Path(out).resolve()
    # Reusing output mixes stale ranks/traces with a new run. All ranks make the same check.
    if (out / "manifest.json").exists():
        raise FileExistsError(f"run already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    device, rank, world, owned = setup(config)
    try:
        return _run(config, out, resume, device, rank, world)
    finally:
        if owned:
            dist.destroy_process_group()


def _run(config, out, resume, device, rank, world):
    origin = provenance()
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    lengths = make_lengths(config.samples, config.min_length, config.max_length, config.seed)
    fleet = None
    if config.fleet_path:
        from tailtrace.fleet import FleetProfile

        fleet = FleetProfile.load(config.fleet_path)
        fleet.check_workload(config, world)
    plans = plan_epoch(lengths, world, config.batch_size, config.seed, config.planner, fleet)
    if not plans:
        raise ValueError("samples must cover at least one complete global batch")
    model = wrap_model(CausalTransformer(config).to(device), config, device, world)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    start_step = 0
    if resume:
        from tailtrace.checkpoint import restore

        start_step = restore(
            resume, model, optimizer, config, world, fleet.sha256 if fleet else None
        )
        if start_step >= config.steps:
            raise ValueError("resume checkpoint must precede configured steps")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        schedule=torch.profiler.schedule(
            wait=config.warmup, warmup=1, active=config.profile_steps, repeat=1
        ),
        on_trace_ready=lambda p: p.export_chrome_trace(str(out / f"trace-rank{rank}.json")),
    )
    rows, losses, timers = [], [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if world > 1:
        dist.barrier()
    wall_start = time.perf_counter()
    current_epoch = -1
    nvtx = config.nvtx and device.type == "cuda"
    with profiler if config.profile else contextlib.nullcontext():
        for step in range(start_step, config.steps):
            # Re-shuffle at each epoch; both planners see exactly the same global batches.
            epoch, offset = divmod(step, len(plans))
            if epoch != current_epoch:
                plans = plan_epoch(
                    lengths, world, config.batch_size, config.seed + epoch, config.planner, fleet
                )
                current_epoch = epoch
            plan = plans[offset]
            ids = plan.ranks[rank]
            begin = time.perf_counter()
            gpu_pair = None
            if device.type == "cuda":
                gpu_pair = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                gpu_pair[0].record()
            with region(f"tailtrace/step/{step}", nvtx):
                with region("tailtrace/input", nvtx):
                    if rank == config.delay_rank:
                        time.sleep(config.delay_ms / 1000)
                    tokens, target = make_batch(
                        ids, lengths, config.vocab_size, config.seed, device
                    )
                optimizer.zero_grad(set_to_none=True)
                amp = (
                    torch.autocast(device.type, dtype=torch.bfloat16)
                    if config.precision == "bf16"
                    else contextlib.nullcontext()
                )
                with region("tailtrace/forward", nvtx), amp:
                    logits = model(tokens)
                    loss, loss_sum = token_loss(logits, target, plan.tokens, world)
                with region("tailtrace/backward", nvtx):
                    loss.backward()
                with region("tailtrace/optimizer", nvtx):
                    optimizer.step()
                if config.diagnostic_sync and device.type == "cuda":
                    torch.cuda.synchronize()
            if gpu_pair:
                gpu_pair[1].record()
            enqueue_ms = (time.perf_counter() - begin) * 1000
            rows.append(
                {
                    "schema_version": 1,
                    "rank": rank,
                    "step": step,
                    "epoch": epoch,
                    "warmup": step < config.warmup,
                    "local_samples": len(ids),
                    "local_tokens": sum(lengths[i] - 1 for i in ids),
                    "global_tokens": plan.tokens,
                    "padded_tokens": len(ids) * tokens.shape[1],
                    "predicted_cost": plan.costs[rank],
                    "host_enqueue_ms": enqueue_ms,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated() if gpu_pair else 0,
                }
            )
            losses.append(loss_sum)
            timers.append(gpu_pair)
            if config.profile:
                profiler.step()
            if config.checkpoint_every and (step + 1) % config.checkpoint_every == 0:
                from tailtrace.checkpoint import save

                save(
                    out / f"checkpoint-{step + 1}",
                    model,
                    optimizer,
                    config,
                    world,
                    step + 1,
                    rank,
                    fleet.sha256 if fleet else None,
                )
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    for row, loss_sum, timer in zip(rows, losses, timers, strict=True):
        row["step_ms"] = timer[0].elapsed_time(timer[1]) if timer else row["host_enqueue_ms"]
        row["loss_sum"] = loss_sum.item()
    hardware = {
        **origin,
        "rank": rank,
        "torch": str(torch.__version__),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "backend": dist.get_backend() if world > 1 else None,
    }
    payload = {"rows": rows, "hardware": hardware, "wall_ms": wall_ms}
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    if rank == 0:
        for p in gathered:
            path = out / f"metrics-rank{p['hardware']['rank']}.jsonl"
            path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in p["rows"]))
        manifest = {
            "schema_version": 1,
            "evidence": "observed",
            "config": config.to_dict(),
            "world_size": world,
            "hardware": [p["hardware"] for p in gathered],
            "start_step": start_step,
            "resume": resume,
            "training_loop_wall_ms": max(p["wall_ms"] for p in gathered),
            "dropped_samples_per_epoch": config.samples % (world * config.batch_size),
            "timing": "cuda_event" if device.type == "cuda" else "cpu_wall",
            "instrumented": config.profile or config.diagnostic_sync or config.nvtx,
            "fleet": fleet.summary() if fleet else None,
            "cost_scope": "isolated_compute_estimate" if fleet else "padded_attention_proxy",
        }
        atomic_json(out / "manifest.json", manifest)
        from tailtrace.report import summarize_run

        summary = summarize_run(out)
        atomic_json(out / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return summary
    return {}
