"""Manual GPU gate: DDP and FSDP2 with eager and custom norms vs global SGD."""

import sys
from dataclasses import replace

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor

from tailtrace.config import TrainConfig
from tailtrace.distributed import setup, wrap_model
from tailtrace.evidence import atomic_json
from tailtrace.model import CausalTransformer, make_batch, token_loss
from tailtrace.planner import plan_epoch


def main():
    config = TrainConfig(
        device="cuda",
        strategy="ddp",
        width=32,
        heads=4,
        layers=2,
        min_length=2,
        max_length=24,
        vocab_size=64,
    )
    device, rank, world, owned = setup(config)
    if world != 2:
        raise ValueError("GPU validation requires exactly two ranks")
    torch.backends.cuda.matmul.allow_tf32 = False
    lengths = [24, 3, 3, 3, 3, 3, 3, 3, 24, 12, 8, 9, 3, 5, 4, 16]
    errors = {}
    try:
        for strategy in ("ddp", "fsdp2"):
            for kernel in ("reference", "cuda"):
                cfg = replace(config, strategy=strategy, kernel=kernel)
                torch.manual_seed(81)
                reference = CausalTransformer(config).to(device)
                local = CausalTransformer(cfg).to(device)
                local.load_state_dict(reference.state_dict())
                model = wrap_model(local, cfg, device, world)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
                for plan in plan_epoch(lengths, world, 4, 3):
                    ref_optimizer.zero_grad(set_to_none=True)
                    optimizer.zero_grad(set_to_none=True)
                    ids = [i for group in plan.ranks for i in group]
                    x, y = make_batch(ids, lengths, cfg.vocab_size, cfg.seed, device)
                    loss, _ = token_loss(reference(x), y, plan.tokens, 1)
                    loss.backward()
                    x, y = make_batch(plan.ranks[rank], lengths, cfg.vocab_size, cfg.seed, device)
                    loss, _ = token_loss(model(x), y, plan.tokens, world)
                    loss.backward()
                    for a, b in zip(local.parameters(), reference.parameters(), strict=True):
                        grad = a.grad.full_tensor() if isinstance(a.grad, DTensor) else a.grad
                        torch.testing.assert_close(grad, b.grad, rtol=3e-4, atol=5e-6)
                    optimizer.step()
                    ref_optimizer.step()
                state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True))
                errors[f"{strategy}/{kernel}"] = max(
                    (state[k] - v).abs().max().item() for k, v in reference.state_dict().items()
                )
                for k, v in reference.state_dict().items():
                    torch.testing.assert_close(state[k], v, rtol=3e-4, atol=5e-6)
        if rank == 0:
            atomic_json(
                sys.argv[1],
                {
                    "world_size": world,
                    "dtype": "float32",
                    "max_parameter_error": errors,
                    "gradient_and_sgd_equivalence": "passed",
                },
            )
    finally:
        if owned:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
