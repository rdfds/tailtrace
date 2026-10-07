"""Local memory experiment on explicitly generated metric rows, not training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

from tailtrace.corpus import file_hash
from tailtrace.evidence import atomic_json, provenance
from tailtrace.report import summarize_run


def make_fixture(root, steps):
    root = Path(root)
    root.mkdir(parents=True)
    atomic_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "evidence": "synthetic",
            "world_size": 2,
            "timing": "generated_fixture",
            "instrumented": False,
            "config": {"steps": steps, "warmup": 0},
        },
    )
    for rank in range(2):
        with (root / f"metrics-rank{rank}.jsonl").open("w") as stream:
            for step in range(steps):
                stream.write(
                    json.dumps(
                        {
                            "rank": rank,
                            "step": step,
                            "warmup": False,
                            "local_tokens": 4,
                            "global_tokens": 8,
                            "padded_tokens": 5,
                            "step_ms": 1 + ((step * 17 + rank * 3) % 101) / 7,
                            "loss_sum": 8,
                            "predicted_cost": 4 + rank,
                            "peak_memory_bytes": 0,
                        }
                    )
                    + "\n"
                )


def measure(root, retain_steps):
    tracemalloc.start()
    begin = time.perf_counter()
    summary = summarize_run(root, retain_steps=retain_steps)
    elapsed_ms = (time.perf_counter() - begin) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss if sys.platform == "darwin" else rss * 1024)
    except ImportError:
        rss_bytes = None
    return {
        "retain_steps": retain_steps,
        "python_peak_bytes": peak,
        "process_peak_rss_bytes": rss_bytes,
        "wall_ms": elapsed_ms,
        "retained_rows": len(summary["steps"]),
        "aggregates": {
            key: summary[key]
            for key in (
                "measured_steps",
                "median_step_ms",
                "p95_step_ms",
                "tokens_per_second",
                "mean_padding_fraction",
                "max_peak_memory_bytes",
                "workload_sha256",
            )
        },
    }


def benchmark(out, sizes=(1000, 10000, 50000)):
    if not sizes or any(type(n) is not int or n < 1 for n in sizes):
        raise ValueError("sizes must be positive integers")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for size in sizes:
        fixture = out / f"fixture-{size}"
        make_fixture(fixture, size)
        observations = []
        for retain_steps in (False, True):
            command = [sys.executable, "-m", "tailtrace.metrics_bench", "--worker", str(fixture)]
            if retain_steps:
                command.append("--retain-steps")
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=120, env={**os.environ}
            )
            if result.returncode:
                raise RuntimeError(result.stderr)
            observation = json.loads(result.stdout)
            observations.append({"command": command, **observation})
        if observations[0]["aggregates"] != observations[1]["aggregates"]:
            raise AssertionError("streaming and retained reports disagree")
        cases.append(
            {
                "global_steps": size,
                "rank_rows": size * 2,
                "input_bytes": sum(p.stat().st_size for p in fixture.glob("*.jsonl")),
                "input_files": [
                    {"path": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)}
                    for p in sorted(fixture.glob("*.jsonl"))
                ],
                "observations": observations,
                "python_peak_ratio": observations[1]["python_peak_bytes"]
                / observations[0]["python_peak_bytes"],
            }
        )
    data = {
        "schema_version": 1,
        "evidence": "observed_cpu",
        "input_evidence": "synthetic",
        "scope": "report_aggregation_memory",
        "provenance": provenance(),
        "cases": cases,
        "limitations": [
            "Generated metric rows are not observed training or GPU performance.",
            "tracemalloc measures Python allocations, excluding SQLite C allocations and OS cache.",
            "Process peak RSS includes interpreter imports and SQLite, measured in fresh subprocesses.",
            "Aggregate mode uses disk space proportional to steps and a 1 MiB SQLite cache target.",
            "Retained mode intentionally stores per-step rows for offline inspection.",
            "Single observations characterize memory growth, not latency speedup or a universal RAM bound.",
        ],
    }
    atomic_json(out / "benchmark.json", data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--retain-steps", action="store_true")
    args = parser.parse_args()
    print(json.dumps(measure(args.worker, args.retain_steps)))
