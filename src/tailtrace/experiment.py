from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
from pathlib import Path

from tailtrace.config import TrainConfig
from tailtrace.evidence import atomic_json, provenance
from tailtrace.report import INTERVENTIONS, compare_runs, write_html


def run_experiment(
    config_path: str, intervention_path: str, seeds: list[int], workers: int, out: str
):
    if workers < 1 or len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("workers must be positive and seeds nonempty/distinct")
    config = TrainConfig.load(config_path)
    changes = json.loads(Path(intervention_path).read_text())
    if not changes or set(changes) - INTERVENTIONS:
        raise ValueError(f"intervention keys must be in {sorted(INTERVENTIONS)}")
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    records = {
        "schema_version": 1,
        "status": "running",
        "provenance": provenance(),
        "workers": workers,
        "seeds": seeds,
        "intervention": changes,
        "runs": [],
    }
    atomic_json(root / "experiment.json", records)
    baselines, candidates = [], []
    try:
        for seed in seeds:
            a = TrainConfig(**{**config.to_dict(), "seed": seed})
            b = TrainConfig(**{**a.to_dict(), **changes})
            order = ["baseline", "candidate"]
            random.Random(seed).shuffle(order)
            paths = {name: root / f"seed-{seed}-{name}" for name in order}
            for name in order:
                run_dir = paths[name]
                config_file = root / f"seed-{seed}-{name}.json"
                atomic_json(config_file, (a if name == "baseline" else b).to_dict())
                command = [
                    sys.executable,
                    "-m",
                    "tailtrace",
                    "train",
                    "--config",
                    str(config_file),
                    "--out",
                    str(run_dir),
                ]
                if workers > 1:
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        port = sock.getsockname()[1]
                    command = [
                        sys.executable,
                        "-m",
                        "torch.distributed.run",
                        "--nnodes=1",
                        f"--nproc-per-node={workers}",
                        "--master-addr=127.0.0.1",
                        f"--master-port={port}",
                        "-m",
                        "tailtrace",
                        *command[3:],
                    ]
                print(f"seed={seed} arm={name} workers={workers}", flush=True)
                log = root / f"seed-{seed}-{name}.log"
                with log.open("w") as stream:
                    result = subprocess.run(
                        command,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        env={**os.environ, "OMP_NUM_THREADS": str(config.threads)},
                        timeout=3600,
                    )
                records["runs"].append(
                    {
                        "seed": seed,
                        "arm": name,
                        "returncode": result.returncode,
                        "path": str(run_dir),
                        "log": str(log),
                    }
                )
                atomic_json(root / "experiment.json", records)
                if result.returncode:
                    raise RuntimeError(f"training failed; inspect {log}")
            baselines.append(str(paths["baseline"]))
            candidates.append(str(paths["candidate"]))
        comparison = compare_runs(baselines, candidates)
        atomic_json(root / "comparison.json", comparison)
        write_html(comparison, root / "report.html")
        records["status"] = "complete"
        atomic_json(root / "experiment.json", records)
        return comparison
    except BaseException:
        records["status"] = "failed"
        atomic_json(root / "experiment.json", records)
        raise
