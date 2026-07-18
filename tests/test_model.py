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


