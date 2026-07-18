import pytest

torch = pytest.importorskip("torch")

from tailtrace.config import TrainConfig
from tailtrace.model import CausalTransformer, make_batch, residual_rmsnorm, token_loss
from tailtrace.planner import plan_epoch


def test_rmsnorm_gradcheck():
    # The reference intentionally accumulates in fp32. Use direct fp64 algebra for gradcheck.
    x, r = (torch.randn(3, 7, dtype=torch.double, requires_grad=True) for _ in range(2))
    w = torch.randn(7, dtype=torch.double, requires_grad=True)

    def fn(x, r, w):
        return (x + r) * ((x + r).square().mean(-1, keepdim=True) + 1e-6).rsqrt() * w

    assert torch.autograd.gradcheck(fn, (x, r, w))
    actual = residual_rmsnorm(x.float(), r.float(), w.float())
    torch.testing.assert_close(actual.double(), fn(x, r, w), rtol=1e-5, atol=1e-6)


def test_variable_rank_loss_matches_global_objective():
    torch.set_num_threads(1)
    config = TrainConfig(width=16, heads=2, layers=1, max_length=24, vocab_size=32)
    torch.manual_seed(5)
    model = CausalTransformer(config)
    lengths = [24, 3, 3, 3, 3, 3, 3, 3]
    plan = plan_epoch(lengths, 2, 4, 1)[0]
    device = torch.device("cpu")
    global_ids = [i for group in plan.ranks for i in group]
    x, y = make_batch(global_ids, lengths, config.vocab_size, config.seed, device)
    loss, _ = token_loss(model(x), y, plan.tokens, 1)
    loss.backward()
    reference = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for ids in plan.ranks:
        x, y = make_batch(ids, lengths, config.vocab_size, config.seed, device)
        loss, _ = token_loss(model(x), y, plan.tokens, 2)
        (loss / 2).backward()
    for p, expected in zip(model.parameters(), reference, strict=True):
        torch.testing.assert_close(p.grad, expected, rtol=1e-4, atol=2e-6)
