"""Portable cost profiles with explicit measurement scope and content identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tailtrace.cost import RankCost, group_shape


def workload_signature(config):
    # Device identity is stored per rank. Hardware migration needs fresh calibration.
    return {
        key: getattr(config, key)
        for key in (
            "width",
            "heads",
            "layers",
            "vocab_size",
            "max_length",
            "kernel",
            "precision",
            "activation_checkpointing",
            "device",
            "threads",
        )
    }


class FleetProfile:
    def __init__(self, data):
        if data.get("schema_version") != 1 or data.get("evidence") not in {
            "declared",
            "observed_isolated",
        }:
            raise ValueError("unsupported fleet profile schema or evidence")
        if data.get("units") not in {"proxy", "ms"}:
            raise ValueError("fleet units must be proxy or ms")
        if data["evidence"] == "observed_isolated" and (
            data["units"] != "ms"
            or data.get("scope") != "forward_backward_no_collectives"
            or not data.get("workload")
            or not data.get("hardware")
        ):
            raise ValueError("observed profiles require isolated scope, workload, and hardware")
        if not isinstance(data.get("ranks"), list) or not data["ranks"]:
            raise ValueError("fleet needs a nonempty ordered rank list")
        self.models = tuple(RankCost(**rank["model"]) for rank in data["ranks"])
        if data["evidence"] == "observed_isolated" and len(data["hardware"]) != len(self.models):
            raise ValueError("observed profile hardware must cover every rank")
        self.data = json.loads(json.dumps(data, allow_nan=False))
        self.sha256 = hashlib.sha256(
            json.dumps(self.data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def check_workload(self, config, world):
        if len(self.models) != world:
            raise ValueError("fleet rank count must match the process group")
        if self.data["evidence"] == "observed_isolated" and self.data[
            "workload"
        ] != workload_signature(config):
            raise ValueError("calibrated workload changed; collect a new profile")
        if self.data["evidence"] == "observed_isolated" and config.strategy == "fsdp2":
            raise ValueError(
                "isolated unsharded calibration does not model FSDP2 resharding; use a declared proxy"
            )

    def check_domains(self, groups, lengths):
        for group, rank in zip(groups, self.data["ranks"], strict=True):
            domain = rank.get("domain")
            if domain is None and self.data["evidence"] == "observed_isolated":
                raise ValueError("observed profile has no calibration domain")
            if domain:
                batch, width = group_shape(group, lengths)
                if (
                    not domain["min_batch"] <= batch <= domain["max_batch"]
                    or not domain["min_width"] <= width <= domain["max_width"]
                ):
                    raise ValueError("planned shape lies outside the calibration domain")

    def check_hardware(self, rank, current):
        if self.data["evidence"] == "observed_isolated":
            recorded = self.data["hardware"][rank]
            for key in (
                "source_sha256",
                "hostname",
                "platform",
                "torch",
                "gpu",
                "cuda_runtime",
                "device",
            ):
                if key not in recorded or recorded[key] != current.get(key):
                    raise ValueError(f"calibrated rank {rank} hardware/software changed: {key}")

    def summary(self):
        return {
            "sha256": self.sha256,
            "evidence": self.data["evidence"],
            "units": self.data["units"],
            "scope": self.data.get("scope", "declared_compute_proxy"),
            "ranks": self.data["ranks"],
        }
