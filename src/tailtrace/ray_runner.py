from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from uuid import uuid4

from tailtrace.config import TrainConfig


def worker(payload):
    from ray import train

    from tailtrace.train import run

    context = train.get_context()
    os.environ.update(
        WORLD_SIZE=str(context.get_world_size()),
        RANK=str(context.get_world_rank()),
        LOCAL_RANK=str(context.get_local_rank()),
    )
    summary = run(TrainConfig(**payload["config"]), payload["out"])
    # All workers report the same number of times; Ray records rank-zero metrics.
    train.report({k: v for k, v in summary.items() if isinstance(v, (int, float))})


