"""Separate two-node torchrun agents on localhost; this is not multi-host evidence."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from tailtrace.evidence import atomic_json, provenance
from tailtrace.recovery import compare_states, launch, load_state


def launch_nodes(config_path, out, log_root, timeout=120):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    processes, logs, commands = [], [], []
    try:
        for node in range(2):
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nnodes=2",
                "--nproc-per-node=1",
                f"--node-rank={node}",
                "--master-addr=127.0.0.1",
                f"--master-port={port}",
                "-m",
                "tailtrace",
                "train",
                "--config",
                str(config_path),
                "--out",
                str(out),
            ]
            log = (Path(log_root) / f"node-{node}.log").open("x")
            logs.append(log)
            commands.append(command)
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1"},
                )
            )
        deadline = time.monotonic() + timeout
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("logical node failed; inspect node logs")
            if time.monotonic() >= deadline:
                raise TimeoutError("logical nodes timed out; inspect node logs")
            time.sleep(0.1)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("logical node failed; inspect node logs")
        return [
            {
                "logical_node": i,
                "command": command,
                "returncode": processes[i].returncode,
                "log": f"node-{i}.log",
            }
            for i, command in enumerate(commands)
        ]
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs:
            log.close()


def audit_nodes(config, out):
    if config.device != "cpu" or config.strategy != "ddp" or config.precision != "fp32":
        raise ValueError("logical-node audit requires CPU fp32 DDP")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = replace(
        config,
        checkpoint_every=config.steps,
        profile=False,
        nvtx=False,
        diagnostic_sync=False,
        metrics_flush_every=2,
        dataset_path=str(Path(config.dataset_path).resolve()) if config.dataset_path else None,
        fleet_path=str(Path(config.fleet_path).resolve()) if config.fleet_path else None,
    )
    atomic_json(out / "config.json", config.to_dict())
    reference = launch(out / "config.json", out / "single_agent", out / "single-agent.log")
    if reference["returncode"]:
        raise RuntimeError("single-agent reference failed; inspect single-agent.log")
    nodes = launch_nodes(out / "config.json", out / "two_agents", out)
    manifest = json.loads((out / "two_agents/manifest.json").read_text())
    hardware = manifest["hardware"]
    topology = [
        (h["rank"], h["local_rank"], h["group_rank"], h["local_world_size"]) for h in hardware
    ]
    if topology != [(0, 0, 0, 1), (1, 0, 1, 1)] or len({h["hostname"] for h in hardware}) != 1:
        raise AssertionError(
            "expected two logical groups, each with local rank zero, on one physical host"
        )
    dataset = (manifest.get("dataset") or {}).get("sha256")
    fleet = (manifest.get("fleet") or {}).get("sha256")
    comparison = compare_states(
        load_state(out / "single_agent" / f"checkpoint-{config.steps}", config, dataset, fleet),
        load_state(out / "two_agents" / f"checkpoint-{config.steps}", config, dataset, fleet),
    )
    data = {
        "schema_version": 1,
        "evidence": "observed_cpu",
        "scope": "two_logical_node_rendezvous",
        "physical_hosts": 1,
        "logical_nodes": 2,
        "world_size": 2,
        "topology": topology,
        "comparison": comparison,
        "reference": reference,
        "agents": nodes,
        "hardware": provenance(),
        "limitations": [
            "both logical nodes run on one physical CPU host",
            "no multi-host networking, clock alignment, NCCL, GPU, or scaling claim",
        ],
    }
    atomic_json(out / "audit.json", data)
    return data
