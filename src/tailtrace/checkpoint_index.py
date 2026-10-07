"""Content verification and compatible committed-checkpoint selection, without torch."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from tailtrace.corpus import content_hash, file_hash

MUTABLE = {
    "steps",
    "warmup",
    "profile",
    "nvtx",
    "profile_steps",
    "checkpoint_every",
    "diagnostic_sync",
    "metrics_flush_every",
}
SHARD = re.compile(r"__\d+_\d+\.distcp\Z")


def model_contract_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in (
        "model.py",
        "distributed.py",
        "kernels/__init__.py",
        "kernels/bindings.cpp",
        "kernels/residual_rmsnorm.cu",
    ):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes())
    return digest.hexdigest()


def storage_files(root):
    return sorted(
        p for p in Path(root).iterdir() if p.name == ".metadata" or SHARD.fullmatch(p.name)
    )


def seal(root, metadata):
    paths = storage_files(root)
    if not any(p.name == ".metadata" for p in paths) or not any(
        SHARD.fullmatch(p.name) for p in paths
    ):
        raise ValueError("checkpoint storage is missing metadata or shards")
    data = {
        **metadata,
        "schema_version": 2,
        "model_contract_sha256": model_contract_hash(),
        "files": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in paths
        ],
    }
    data["sha256"] = content_hash(data)
    return data


def verify(root, require_sealed=False):
    root = Path(root)
    if not (root / "complete.json").is_file():
        raise ValueError("checkpoint has no complete.json commit marker")
    data = json.loads((root / "complete.json").read_text())
    if data.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported checkpoint schema")
    for name in ("step", "world_size"):
        if (
            isinstance(data.get(name), bool)
            or not isinstance(data.get(name), int)
            or data[name] < 1
        ):
            raise ValueError(f"checkpoint {name} must be a positive integer")
    if not isinstance(data.get("config"), dict):
        raise ValueError("checkpoint config must be a mapping")
    if data["schema_version"] == 1:
        if require_sealed:
            raise ValueError("legacy checkpoint has no storage content seal")
        return data
    if data.get("sha256") != content_hash({k: v for k, v in data.items() if k != "sha256"}):
        raise ValueError("checkpoint marker content changed")
    entries = data.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("checkpoint storage inventory is empty")
    names = []
    for entry in entries:
        name = entry["path"]
        if not isinstance(name, str) or (name != ".metadata" and not SHARD.fullmatch(name)):
            raise ValueError("invalid checkpoint storage path")
        names.append(name)
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or file_hash(path) != entry["sha256"]
        ):
            raise ValueError(f"checkpoint storage content changed: {name}")
    if (
        len(set(names)) != len(names)
        or ".metadata" not in names
        or not any(SHARD.fullmatch(name) for name in names)
        or sorted(names) != [p.name for p in storage_files(root)]
    ):
        raise ValueError("checkpoint storage inventory differs")
    return data


def check_compatible(
    data,
    config,
    world,
    fleet_sha256=None,
    dataset_sha256=None,
    require_remaining=True,
    torch_version=None,
):
    if torch_version and data.get("torch_version") and data["torch_version"] != torch_version:
        raise ValueError("checkpoint torch version changed")
    if data["world_size"] != world:
        raise ValueError("checkpoint recovery currently requires the same world size")
    if data.get("fleet_sha256") != fleet_sha256:
        raise ValueError("checkpoint fleet profile content changed")
    if data.get("dataset_sha256") != dataset_sha256:
        raise ValueError("checkpoint dataset content changed")
    if data.get("model_contract_sha256") and data["model_contract_sha256"] != model_contract_hash():
        raise ValueError("checkpoint model/objective implementation changed")
    for key, value in data["config"].items():
        if key not in MUTABLE and config.to_dict().get(key) != value:
            raise ValueError(f"checkpoint configuration changed: {key}")
    if require_remaining and data["step"] >= config.steps:
        raise ValueError("resume checkpoint must precede configured steps")


def select(root, config, world, fleet_sha256=None, dataset_sha256=None, torch_version=None):
    """Choose the largest verified compatible step; record every rejected candidate."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("checkpoint root is missing")
    valid, inspected = [], []
    for path in sorted(root.glob("checkpoint-*")):
        if not path.is_dir():
            continue
        try:
            data = verify(path, require_sealed=True)
            check_compatible(
                data, config, world, fleet_sha256, dataset_sha256, torch_version=torch_version
            )
            if path.name != f"checkpoint-{data['step']}":
                raise ValueError("checkpoint directory name disagrees with committed step")
            valid.append((data["step"], str(path), data["sha256"]))
            inspected.append({"path": path.name, "step": data["step"], "status": "eligible"})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            inspected.append({"path": path.name, "status": "rejected", "reason": str(exc)})
    if not valid:
        raise ValueError("no verified compatible checkpoint: " + json.dumps(inspected))
    step, path, sha256 = max(valid)
    return {"path": path, "step": step, "sha256": sha256, "candidates": inspected}
