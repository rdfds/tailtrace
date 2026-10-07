"""Faults inside actual DCP writes and recovery from a corrupted newest checkpoint."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import replace
from pathlib import Path

from tailtrace.checkpoint_index import verify
from tailtrace.corpus import file_hash
from tailtrace.evidence import atomic_json, provenance
from tailtrace.journal import inspect_run
from tailtrace.recovery import compare_states, launch, load_state


def install_partial_writer(step, limit=4096):
    import torch.distributed.checkpoint as dcp

    original = dcp.FileSystemWriter

    class PartialStream:
        def __init__(self, stream):
            self.stream, self.written = stream, 0

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def write(self, data):
            remaining = limit - self.written
            count = self.stream.write(data[:remaining])
            self.written += count
            if self.written >= limit:
                self.stream.flush()
                os.fsync(self.stream.fileno())
                print(
                    f"TAILTRACE_PARTIAL_WRITE rank=1 step={step} bytes={limit} code=88", flush=True
                )
                os._exit(88)
            return count

    class FaultWriter(original):
        def write_data(self, plan, planner):
            if int(os.environ["RANK"]) == 1 and Path(self.path).name == f"checkpoint-{step}":
                create_stream = self.fs.create_stream

                @contextlib.contextmanager
                def partial_stream(path, mode):
                    with create_stream(path, mode) as stream:
                        yield PartialStream(stream) if str(path).endswith(".distcp") else stream

                self.fs.create_stream = partial_stream
            return super().write_data(plan, planner)

    dcp.FileSystemWriter = FaultWriter


def audit_faults(config, out):
    if config.device != "cpu" or config.strategy != "ddp" or config.precision != "fp32":
        raise ValueError("fault audit requires CPU fp32 DDP")
    if config.steps < 6:
        raise ValueError("fault audit requires at least six steps")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = replace(
        config,
        profile=False,
        nvtx=False,
        diagnostic_sync=False,
        metrics_flush_every=1,
        dataset_path=str(Path(config.dataset_path).resolve()) if config.dataset_path else None,
        fleet_path=str(Path(config.fleet_path).resolve()) if config.fleet_path else None,
    )
    whole, crash = (
        replace(config, checkpoint_every=config.steps),
        replace(config, checkpoint_every=1),
    )
    atomic_json(out / "whole-config.json", whole.to_dict())
    atomic_json(out / "crash-config.json", crash.to_dict())
    runs = {}
    runs["uninterrupted"] = launch(out / "whole-config.json", out / "whole", out / "whole.log")
    if runs["uninterrupted"]["returncode"]:
        raise RuntimeError(f"reference failed; inspect {out / 'whole.log'}")
    runs["partial_write"] = launch(
        out / "crash-config.json",
        out / "failed",
        out / "partial-write.log",
        fault_step=4,
        fault_mode="partial_write",
    )
    failed = out / "failed"
    incomplete = failed / "checkpoint-4"
    marker = "TAILTRACE_PARTIAL_WRITE rank=1 step=4 bytes=4096 code=88"
    if (
        not runs["partial_write"]["returncode"]
        or (incomplete / "complete.json").exists()
        or marker not in (out / "partial-write.log").read_text()
        or not any(p.stat().st_size == 4096 for p in incomplete.glob("*.distcp"))
    ):
        raise AssertionError("expected partial DCP shard and failed process were not observed")
    before = verify(failed / "checkpoint-3", require_sealed=True)
    shard = next(p for p in (failed / "checkpoint-3").glob("*.distcp") if p.is_file())
    before_hash = file_hash(shard)
    with shard.open("r+b") as stream:
        old = stream.read(1)
        stream.seek(0)
        stream.write(bytes([old[0] ^ 1]))
        stream.flush()
        os.fsync(stream.fileno())
    corruption = {
        "path": str(shard.relative_to(out)),
        "offset": 0,
        "before_byte": old[0],
        "after_byte": old[0] ^ 1,
        "before_sha256": before_hash,
        "after_sha256": file_hash(shard),
        "committed_step": before["step"],
    }
    telemetry = inspect_run(failed)
    if telemetry["status"] != "partial" or telemetry["durable_through_step"] != 3:
        raise AssertionError("bounded telemetry did not retain the four completed training steps")
    runs["resumed"] = launch(
        out / "whole-config.json", out / "resumed", out / "resume.log", resume_root=failed
    )
    if runs["resumed"]["returncode"]:
        raise RuntimeError(f"fallback recovery failed; inspect {out / 'resume.log'}")
    manifest = json.loads((out / "resumed/manifest.json").read_text())
    selection = manifest["checkpoint_selection"]
    if manifest["start_step"] != 2 or selection["step"] != 2:
        raise AssertionError(
            "automatic recovery did not skip both corrupt and incomplete checkpoints"
        )
    state = verify(failed / "checkpoint-2")
    comparison = compare_states(
        load_state(
            out / "whole" / f"checkpoint-{config.steps}",
            whole,
            state.get("dataset_sha256"),
            state.get("fleet_sha256"),
        ),
        load_state(
            out / "resumed" / f"checkpoint-{config.steps}",
            whole,
            state.get("dataset_sha256"),
            state.get("fleet_sha256"),
        ),
    )
    data = {
        "schema_version": 1,
        "evidence": "observed_cpu",
        "scope": "partial_write_and_corrupt_checkpoint_recovery",
        "world_size": 2,
        "physical_hosts": 1,
        "partial_write_step": 4,
        "exitcode": 88,
        "partial_shard_bytes": 4096,
        "corruption": corruption,
        "selection": selection,
        "telemetry": telemetry,
        "comparison": comparison,
        "runs": runs,
        "hardware": provenance(),
        "limitations": [
            "one physical CPU host",
            "same world size",
            "fault injection aborts one rank after 4096 bytes of an actual DCP shard write",
            "corruption is deliberately injected after the failed job exits",
            "no power-loss, remote-filesystem durability, elastic, FSDP2, or GPU claim",
        ],
    }
    atomic_json(out / "audit.json", data)
    return data
