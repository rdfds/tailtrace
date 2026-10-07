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
    "fleet_path",
    "planner",
    "kernel",
    "bucket_cap_mb",
    "activation_checkpointing",
    "strategy",
    "precision",
}


def compare_runs(baselines: list[str], candidates: list[str], bootstrap: int = 4000) -> dict:
    if len(baselines) != len(candidates) or not baselines:
        raise ValueError("provide the same nonzero number of baseline and candidate runs")
    if bootstrap < 100:
        raise ValueError("bootstrap must be >= 100")
    pairs, seeds = [], set()
    protocol = None
    for base, candidate in zip(baselines, candidates, strict=True):
        a = json.loads((Path(base) / "manifest.json").read_text())
        b = json.loads((Path(candidate) / "manifest.json").read_text())
        if a["evidence"] != "observed" or b["evidence"] != "observed":
            raise ValueError("comparison requires observed runs, not synthetic evidence")
        if a["world_size"] != b["world_size"] or a["timing"] != b["timing"]:
            raise ValueError("world size and timing method must match")
        for key in ("start_step", "resume", "instrumented"):
            if a.get(key) != b.get(key):
                raise ValueError(f"mismatched protocol: {key}")
        if set(a["config"]) != set(b["config"]):
            raise ValueError("configuration schema differs")
        differences = {k for k in a["config"] if a["config"][k] != b["config"].get(k)}
        if not differences or differences - INTERVENTIONS:
            raise ValueError(f"uncontrolled or absent intervention: {sorted(differences)}")
        for m in (a, b):
            if m["config"].get("planner") == "fleet" and not m.get("fleet", {}).get("sha256"):
                raise ValueError("fleet runs require a recorded profile content hash")
            if m["config"].get("dataset_path") and not (m.get("dataset") or {}).get("sha256"):
                raise ValueError("corpus runs require a recorded dataset content hash")
        if a.get("dataset") != b.get("dataset"):
            raise ValueError("paired runs loaded different dataset contents")
        pair_protocol = [
            {
                "config": {k: v for k, v in m["config"].items() if k != "seed"},
                "fleet": m.get("fleet"),
                "dataset": m.get("dataset"),
            }
            for m in (a, b)
        ]
        if protocol is not None and protocol != pair_protocol:
            raise ValueError("experiment settings differ across seed pairs")
        protocol = pair_protocol
        hardware_keys = (
            "hostname",
            "platform",
            "torch",
            "gpu",
            "cuda_runtime",
            "backend",
            "source_sha256",
        )

        def hardware(m, keys=hardware_keys):
            return [
                [r.get(k) for k in keys] for r in sorted(m["hardware"], key=lambda r: r["rank"])
            ]

        if hardware(a) != hardware(b):
            raise ValueError("hardware/software topology differs between paired runs")
        seed = a["config"]["seed"]
        if seed in seeds:
            raise ValueError("independent pairs must use distinct seeds")
        seeds.add(seed)
        sa, sb = summarize_run(base), summarize_run(candidate)
        if [(s["step"], s["tokens"]) for s in sa["steps"]] != [
            (s["step"], s["tokens"]) for s in sb["steps"]
        ]:
            raise ValueError("step and token workloads differ")
        pairs.append(
            {
                "seed": seed,
                "baseline": base,
                "candidate": candidate,
                "interventions": sorted(differences),
                "speedup": sa["median_step_ms"] / sb["median_step_ms"],
                "baseline_median_ms": sa["median_step_ms"],
                "candidate_median_ms": sb["median_step_ms"],
                "baseline_tokens_per_second": sa["tokens_per_second"],
                "candidate_tokens_per_second": sb["tokens_per_second"],
                "baseline_fleet": a.get("fleet"),
                "candidate_fleet": b.get("fleet"),
            }
        )
    if len({tuple(p["interventions"]) for p in pairs}) != 1:
        raise ValueError("all seed pairs must test the same interventions")
    logs = [math.log(p["speedup"]) for p in pairs]
    estimate = math.exp(statistics.mean(logs))
    ci = None
    if len(pairs) >= 3:
        rng = random.Random(19)
        samples = [
            math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(bootstrap)
        ]
        ci = [percentile(samples, 0.025), percentile(samples, 0.975)]
    return {
        "schema_version": 1,
        "evidence": "observed",
        "pairs": pairs,
        "geometric_mean_speedup": estimate,
        "seed_bootstrap_95_ci": ci,
        "independent_pairs": len(pairs),
        "instrumented": a["instrumented"],
        "interpretation": "insufficient independent pairs for an interval"
        if ci is None
        else "candidate faster in sampled runs"
        if ci[0] > 1
        else "candidate slower in sampled runs"
        if ci[1] < 1
        else "interval includes no improvement",
        "limitations": [
            "Bootstrap resamples run seeds, not correlated training steps.",
            "Order, thermal drift, and contention require randomized alternating runs.",
            "A speedup is not evidence of equivalent gradients; run correctness tests separately.",
            "Multiple changed settings estimate the joint intervention, not individual causes.",
        ],
    }


def write_html(data: dict, path: str | Path):
    """Offline report; no third-party scripts, CDNs, or trace data uploads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for key in (
        "evidence",
        "timing",
        "world_size",
        "median_step_ms",
        "p95_step_ms",
        "tokens_per_second",
        "geometric_mean_speedup",
        "independent_pairs",
        "interpretation",
    ):
        if key in data:
            value = data[key]
            cards.append(
                f'<div class="card"><small>{html.escape(key.replace("_", " "))}</small><strong>{html.escape(f"{value:.3f}" if isinstance(value, float) else str(value))}</strong></div>'
            )
    rows = data.get("steps", data.get("pairs", []))
    columns = list(rows[0]) if rows else []
    table = "<tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in columns) + "</tr>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{html.escape(str(row[k]))}</td>" for k in columns) + "</tr>"
    limitations = "".join(f"<li>{html.escape(s)}</li>" for s in data.get("limitations", []))
    path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>TailTrace evidence report</title>
<style>body{{margin:0;background:#101722;color:#e7edf5;font:16px system-ui;padding:48px;max-width:1200px;margin:auto}}
h1{{font-size:42px;margin-bottom:8px}}p,small{{color:#aebed0}}.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:28px 0}}
.card{{background:#1a2636;border:1px solid #314359;border-radius:12px;padding:20px;min-width:160px}}
small,strong{{display:block}}strong{{font-size:24px;margin-top:10px}}table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{text-align:left;border-bottom:1px solid #314359;padding:10px}}.scroll{{overflow:auto}}li{{margin:10px 0}}
pre{{background:#1a2636;padding:20px;overflow:auto}}@media(max-width:600px){{body{{padding:20px}}}}</style>
<h1>TailTrace</h1><p>Evidence for distributed training tail latency · schema 1</p>
<div class="cards">{"".join(cards)}</div><h2>Measurement limits</h2><ul>{limitations}</ul>
<h2>Evidence rows</h2><div class="scroll"><table>{table}</table></div>
<details><summary>Full reproducible payload</summary><pre>{html.escape(json.dumps(data, indent=2))}</pre></details></html>""")
