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


def launch(args):
    import ray
    from ray.train import FailureConfig, RunConfig, ScalingConfig
    from ray.train.torch import TorchConfig, TorchTrainer

    config = TrainConfig.load(args.config)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if config.strategy == "single" and args.workers != 1:
        raise ValueError("single strategy requires one Ray worker")
    # Upload the package (including CUDA sources) rather than the entire checkout/venv.
    init_args = {"runtime_env": {"py_modules": [str(Path(__file__).parent)]}}
    if args.address:
        init_args["address"] = args.address
    else:
        # Ray's RuntimeEnvContext embeds py_executable in a shell command. Its default
        # is unquoted on POSIX, so an interpreter path with spaces fails at exec.
        # Restrict this override to local launches: remote nodes choose their own Python.
        init_args["runtime_env"]["py_executable"] = shlex.quote(sys.executable)
        init_args["num_cpus"] = args.workers * config.threads + 1
        init_args["include_dashboard"] = False
        init_args["object_store_memory"] = 128 * 1024 * 1024
        if config.device == "cpu":
            init_args["num_gpus"] = 0
    ray.init(**init_args)
    try:
        trainer = TorchTrainer(
            worker,
            train_loop_config={"config": config.to_dict(), "out": str(Path(args.out).resolve())},
            scaling_config=ScalingConfig(
                num_workers=args.workers,
                use_gpu=config.device == "cuda",
                resources_per_worker={"CPU": config.threads},
            ),
            torch_config=TorchConfig(backend="nccl" if config.device == "cuda" else "gloo"),
            run_config=RunConfig(
                storage_path=args.storage,
                name=f"tailtrace-{uuid4().hex[:12]}",
                failure_config=FailureConfig(max_failures=0),
            ),
        )
        result = trainer.fit()
        print(result.metrics)
        return result
    finally:
        ray.shutdown()
