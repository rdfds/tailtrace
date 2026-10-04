"""Executed by torchrun: actual DDP vs a single-process global objective."""

import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tailtrace.config import TrainConfig
from tailtrace.evidence import atomic_json
from tailtrace.fleet import FleetProfile
from tailtrace.model import CausalTransformer, make_batch, token_loss
from tailtrace.planner import plan_epoch


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cpu")
    config = TrainConfig(width=16, heads=2, layers=1, max_length=24, vocab_size=32)
    lengths = [24, 3, 3, 3, 3, 3, 3, 3, 24, 12, 8, 9, 3, 5, 4, 16]
    errors = {}
    try:
        fleet = FleetProfile(
            {
                "schema_version": 1,
                "evidence": "declared",
                "units": "proxy",
                "ranks": [
                    {"model": {"quadratic": 1, "max_samples": 7}},
                    {"model": {"quadratic": 4, "max_samples": 7}},
                ],
            }
        )
        for mode in ("random", "balanced", "fleet"):
            torch.manual_seed(91)
            # FP64 separates objective correctness from fp32 cancellation in AdamW's
            # near-zero gradients. FP32 gradient equivalence is tested independently.
            reference = CausalTransformer(config).double()
            local = CausalTransformer(config).double()
            local.load_state_dict(reference.state_dict())
            model = DistributedDataParallel(local)
            reference_optim = torch.optim.AdamW(reference.parameters(), lr=0.001)
            optim = torch.optim.AdamW(model.parameters(), lr=0.001)
            plans = plan_epoch(lengths, world, 4, 3, mode, fleet if mode == "fleet" else None)
            for plan in plans:
                reference_optim.zero_grad(set_to_none=True)
                optim.zero_grad(set_to_none=True)
                ids = [i for group in plan.ranks for i in group]
                x, y = make_batch(ids, lengths, config.vocab_size, config.seed, device)
                loss, _ = token_loss(reference(x), y, plan.tokens, 1)
                loss.backward()
                x, y = make_batch(plan.ranks[rank], lengths, config.vocab_size, config.seed, device)
                loss, _ = token_loss(model(x), y, plan.tokens, world)
                loss.backward()
                for a, b in zip(model.module.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-11)
                optim.step()
                reference_optim.step()
            errors[mode] = max(
                (a - b).abs().max().item()
                for a, b in zip(model.module.parameters(), reference.parameters(), strict=True)
            )
            for a, b in zip(model.module.parameters(), reference.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
        if rank == 0:
            atomic_json(
                sys.argv[1],
                {
                    "world_size": world,
                    "dtype": "float64",
                    "max_parameter_error": errors,
                    "gradient_and_adamw_equivalence": "passed",
                },
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
