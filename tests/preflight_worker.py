"""Real distributed failure guards; each rank must report the expected rejection."""

import os
import sys
from dataclasses import replace
from pathlib import Path

from tailtrace import train
from tailtrace.config import TrainConfig
from tailtrace.evidence import atomic_json

mode, directory = sys.argv[1], Path(sys.argv[2])
rank = int(os.environ["RANK"])
config = TrainConfig(
    samples=8,
    batch_size=2,
    steps=2,
    warmup=0,
    width=16,
    heads=2,
    layers=1,
    vocab_size=32,
    min_length=2,
    max_length=8,
    checkpoint_every=1,
    metrics_flush_every=1,
)
expected = "rank source, configuration, or torch version differs"
if mode == "config" and rank == 1:
    config = replace(config, dataset_path=str(directory / "missing-corpus"), vocab_size=256)
if mode == "source":
    original = train.provenance

    def different_source():
        data = original()
        if rank == 1:
            data["source_sha256"] = "different-worker-source"
        return data

    train.provenance = different_source
if mode == "journal":
    expected = "journal preflight failed"
    original_journal = train.RankJournal

    def broken_journal(*args, **kwargs):
        if rank == 1:
            raise OSError("injected rank-1 journal open failure")
        return original_journal(*args, **kwargs)

    train.RankJournal = broken_journal
if mode == "commit":
    from tailtrace import checkpoint

    expected = "checkpoint commit failed"

    def broken_seal(*args, **kwargs):
        raise OSError("injected rank-0 checkpoint commit failure")

    if rank == 0:
        checkpoint.seal = broken_seal
try:
    train.run(config, directory / "run")
except ValueError as exc:
    if expected not in str(exc):
        raise
    atomic_json(
        directory / f"proof-rank{rank}.json", {"mode": mode, "rank": rank, "error": str(exc)}
    )
else:
    raise AssertionError("expected distributed rejection did not occur")
