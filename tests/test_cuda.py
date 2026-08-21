import pytest

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires NVIDIA CUDA"),
]

from tailtrace.kernels import fused_residual_rmsnorm
from tailtrace.model import residual_rmsnorm


@pytest.mark.parametrize("shape", [(1, 1), (7, 31), (4, 257), (2, 3, 1024), (513, 65)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_and_all_gradients(shape, dtype):
    torch.manual_seed(9)
    x, r = (torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True) for _ in range(2))
    w = torch.randn(shape[-1], device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)
    ref = residual_rmsnorm(x, r, w)
    ref_grads = torch.autograd.grad(ref, (x, r, w), dy)
    actual = fused_residual_rmsnorm(x, r, w)
    actual_grads = torch.autograd.grad(actual, (x, r, w), dy)
    tol = 3e-5 if dtype == torch.float32 else 0.04 if dtype == torch.bfloat16 else 0.005
    torch.testing.assert_close(actual, ref, rtol=tol, atol=tol)
    for a, b in zip(actual_grads, ref_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)


