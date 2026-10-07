"""Real two-rank CPU process-failure and deterministic checkpoint recovery audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from tailtrace.config import TrainConfig
from tailtrace.evidence import atomic_json, provenance


def launch(
    config_path,
    out,
    log_path,
    resume=None,
    fault_step=None,
    timeout=120,
    resume_root=None,
    fault_mode="after_save",
):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        "-m",
        "tailtrace.recovery",
        "--config",
        str(config_path),
        "--out",
        str(out),
    ]
    if resume:
        command += ["--resume", str(resume)]
    if fault_step is not None:
        command += ["--fault-step", str(fault_step), "--fault-mode", fault_mode]
    if resume_root:
        command += ["--resume-latest", str(resume_root)]
    with Path(log_path).open("x") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError(f"recovery worker timed out; inspect {log_path}") from None
    return {"command": command, "returncode": code, "log": Path(log_path).name}


def load_state(path, config, dataset_hash, fleet_hash):
    import torch

    from tailtrace.checkpoint import restore
    from tailtrace.model import CausalTransformer

    model = CausalTransformer(config)
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr)
    step = restore(path, model, optim, config, 2, fleet_hash, dataset_hash)
    return {"model": model.state_dict(), "optimizer": optim.state_dict(), "step": step}


def compare_states(reference, resumed):
    import torch

    digest = hashlib.sha256()
    tensors, elements, maximum = 0, 0, 0.0

    def visit(a, b, path):
        nonlocal tensors, elements, maximum
        digest.update(path.encode() + b"\0")
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape:
                raise AssertionError(f"state tensor metadata differs: {path}")
            if not torch.equal(a, b):
                raise AssertionError(
                    f"state tensor differs: {path}; max error {(a - b).abs().max().item()}"
                )
            tensors += 1
            elements += a.numel()
            maximum = max(maximum, (a - b).abs().max().item() if a.numel() else 0)
            digest.update(str(a.dtype).encode() + str(tuple(a.shape)).encode())
            digest.update(
                a.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise AssertionError(f"state mapping differs: {path}")
            for key in sorted(a, key=str):
                visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise AssertionError(f"state sequence differs: {path}")
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                visit(x, y, f"{path}/{i}")
        else:
            if type(a) is not type(b) or a != b:
                raise AssertionError(f"state value differs: {path}")
            digest.update(json.dumps(a, sort_keys=True).encode())

    visit(reference, resumed, "state")
    return {
        "bitwise_equal": True,
        "tensor_count": tensors,
        "tensor_elements": elements,
        "max_tensor_error": maximum,
        "canonical_state_sha256": digest.hexdigest(),
    }


def audit(config, out, fault_step=2):
    if config.device != "cpu" or config.strategy != "ddp":
        raise ValueError("free recovery audit requires device=cpu and strategy=ddp")
    if not 1 <= fault_step < config.steps:
        raise ValueError("fault step must precede the final training step")
    if config.samples < 2 * config.batch_size:
        raise ValueError("samples must cover two ranks")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = replace(
        config,
        profile=False,
        nvtx=False,
        diagnostic_sync=False,
        dataset_path=str(Path(config.dataset_path).resolve()) if config.dataset_path else None,
        fleet_path=str(Path(config.fleet_path).resolve()) if config.fleet_path else None,
    )
    whole = replace(config, checkpoint_every=config.steps)
    crashed = replace(config, checkpoint_every=fault_step)
    atomic_json(out / "whole-config.json", whole.to_dict())
    atomic_json(out / "crash-config.json", crashed.to_dict())
    runs = {}
    runs["uninterrupted"] = launch(out / "whole-config.json", out / "whole", out / "whole.log")
    if runs["uninterrupted"]["returncode"]:
        raise RuntimeError(f"uninterrupted reference failed; inspect {out / 'whole.log'}")
    runs["crashed"] = launch(
        out / "crash-config.json", out / "crashed", out / "crash.log", fault_step=fault_step
    )
    checkpoint = out / "crashed" / f"checkpoint-{fault_step}"
    if (
        not runs["crashed"]["returncode"]
        or not (checkpoint / "complete.json").is_file()
        or (out / "crashed/manifest.json").exists()
        or "TAILTRACE_INJECTED_EXIT rank=1 code=86" not in (out / "crash.log").read_text()
    ):
        raise AssertionError("crash did not leave the expected committed checkpoint and failed job")
    runs["resumed"] = launch(
        out / "whole-config.json", out / "resumed", out / "resume.log", resume=checkpoint
    )
    if runs["resumed"]["returncode"]:
        raise RuntimeError(f"restart failed; inspect {out / 'resume.log'}")
    resumed_manifest = json.loads((out / "resumed/manifest.json").read_text())
    if resumed_manifest["start_step"] != fault_step:
        raise AssertionError("restart used the wrong global step")
    metadata = json.loads((checkpoint / "complete.json").read_text())
    dataset_hash, fleet_hash = metadata.get("dataset_sha256"), metadata.get("fleet_sha256")
    comparison = compare_states(
        load_state(out / "whole" / f"checkpoint-{config.steps}", whole, dataset_hash, fleet_hash),
        load_state(out / "resumed" / f"checkpoint-{config.steps}", whole, dataset_hash, fleet_hash),
    )
    data = {
        "schema_version": 1,
        "evidence": "observed_cpu",
        "scope": "two_rank_process_recovery",
        "world_size": 2,
        "physical_hosts": 1,
        "fault_rank": 1,
        "fault_step": fault_step,
        "fault": "os._exit(86) after committed distributed checkpoint",
        "final_step": config.steps,
        "dataset_sha256": dataset_hash,
        "hardware": provenance(),
        "runs": runs,
        "comparison": comparison,
        "limitations": [
            "same world size",
            "process failure after completed save",
            "no mid-write, machine-loss, network-partition, or GPU recovery claim",
        ],
    }
    atomic_json(out / "audit.json", data)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--resume-latest")
    parser.add_argument("--fault-step", type=int)
    parser.add_argument(
        "--fault-mode", choices=["after_save", "partial_write"], default="after_save"
    )
    args = parser.parse_args()
    if args.fault_step is not None and args.fault_mode == "partial_write":
        from tailtrace.faults import install_partial_writer

        install_partial_writer(args.fault_step)
    elif args.fault_step is not None:
        import tailtrace.checkpoint as checkpoint

        original = checkpoint.save

        def save_then_exit(path, model, optim, config, world, step, rank, *identities):
            original(path, model, optim, config, world, step, rank, *identities)
            if step == args.fault_step and rank == 1:
                print("TAILTRACE_INJECTED_EXIT rank=1 code=86", flush=True)
                os._exit(86)

        checkpoint.save = save_then_exit
    from tailtrace.train import run

    run(TrainConfig.load(args.config), args.out, args.resume, args.resume_latest)


if __name__ == "__main__":
    main()
