from __future__ import annotations

import html
import json
import math
import random
import statistics
from pathlib import Path


def percentile(values, q):
    xs = sorted(values)
    if not xs:
        raise ValueError("empty distribution")
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize_run(directory: str | Path) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    world = manifest["world_size"]
    paths = sorted(directory.glob("metrics-rank*.jsonl"))
    if len(paths) != world:
        raise ValueError(f"expected {world} rank files, found {len(paths)}")
    grouped = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rank, step = row["rank"], row["step"]
            if rank not in range(world) or (step, rank) in grouped:
                raise ValueError("duplicate or invalid rank/step")
            if not math.isfinite(row["step_ms"]) or row["step_ms"] <= 0:
                raise ValueError("step_ms must be finite and positive")
            grouped[step, rank] = row
    expected_steps = set(range(manifest.get("start_step", 0), manifest["config"]["steps"]))
    if {s for s, _ in grouped} != expected_steps:
        raise ValueError("missing or unexpected steps")
    steps = []
    for step in sorted(expected_steps):
        if any((step, rank) not in grouped for rank in range(world)):
            raise ValueError(f"incomplete rank coverage at step {step}")
        rows = [grouped[step, r] for r in range(world)]
        tokens = rows[0]["global_tokens"]
        if (
            tokens <= 0
            or any(r["global_tokens"] != tokens for r in rows)
            or sum(r["local_tokens"] for r in rows) != tokens
        ):
            raise ValueError("inconsistent global token denominator")
        if any(r["warmup"] != (step < manifest["config"]["warmup"]) for r in rows):
            raise ValueError("inconsistent warmup flags")
        if step < manifest["config"]["warmup"]:
            continue
        times = [r["step_ms"] for r in rows]
        costs = [r["predicted_cost"] for r in rows]
        steps.append(
            {
                "step": step,
                "step_ms": max(times),
                "tokens": tokens,
                "loss": sum(r["loss_sum"] for r in rows) / tokens,
                "rank_spread_ms": max(times) - min(times),
                "proxy_imbalance": max(costs) / statistics.mean(costs),
                "padding_fraction": 1 - tokens / sum(r["padded_tokens"] for r in rows),
            }
        )
    if not steps:
        raise ValueError("no measured steps remain")
    durations = [s["step_ms"] for s in steps]
    return {
        "schema_version": 1,
        "evidence": manifest["evidence"],
        "run": str(directory),
        "world_size": world,
        "timing": manifest["timing"],
        "instrumented": manifest["instrumented"],
        "measured_steps": len(steps),
        "median_step_ms": statistics.median(durations),
        "p95_step_ms": percentile(durations, 0.95),
        "tokens_per_second": sum(s["tokens"] for s in steps) / sum(durations) * 1000,
        "mean_padding_fraction": statistics.mean(s["padding_fraction"] for s in steps),
        "max_peak_memory_bytes": max(r["peak_memory_bytes"] for r in grouped.values()),
        "steps": steps,
        "limitations": [
            "Step latency is max rank-local duration, not a clock-aligned global critical path.",
            "Throughput excludes warmup, checkpoint I/O, and post-run collection; wall time is in manifest.",
            "Synthetic token data exercises systems behavior; loss is not a language quality benchmark.",
        ],
    }


INTERVENTIONS = {
    "planner",
    "kernel",
    "bucket_cap_mb",
    "activation_checkpointing",
    "strategy",
    "precision",
}


