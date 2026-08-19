#include <torch/extension.h>
#include <cmath>

std::vector<torch::Tensor> rms_forward_cuda(torch::Tensor x, torch::Tensor r,
                                         torch::Tensor w, double eps);
std::vector<torch::Tensor> rms_backward_cuda(torch::Tensor dy, torch::Tensor x,
                                          torch::Tensor r, torch::Tensor w, torch::Tensor inv);

void validate(torch::Tensor x, torch::Tensor r, torch::Tensor w) {
  TORCH_CHECK(x.is_cuda() && r.is_cuda() && w.is_cuda(), "all inputs must be CUDA tensors");
  TORCH_CHECK(x.device() == r.device() && x.device() == w.device(), "device mismatch");
  TORCH_CHECK(x.scalar_type() == r.scalar_type() && x.scalar_type() == w.scalar_type(), "dtype mismatch");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
              x.scalar_type() == at::kBFloat16, "supported dtypes: fp32/fp16/bf16");
  TORCH_CHECK(x.dim() >= 1 && x.numel() > 0 && x.sizes() == r.sizes(), "invalid residual shape");
  TORCH_CHECK(w.dim() == 1 && w.numel() == x.size(-1), "weight must match final dimension");
  TORCH_CHECK(x.is_contiguous() && r.is_contiguous() && w.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(x.size(-1) <= 65536, "width exceeds supported limit 65536");
  TORCH_CHECK(x.numel() / x.size(-1) <= 2147483647LL, "too many rows");
}

std::vector<torch::Tensor> forward(torch::Tensor x, torch::Tensor r, torch::Tensor w, double eps) {
  validate(x, r, w);
  TORCH_CHECK(std::isfinite(eps) && eps > 0, "eps must be finite and positive");
  return rms_forward_cuda(x, r, w, eps);
}

std::vector<torch::Tensor> backward(torch::Tensor dy, torch::Tensor x, torch::Tensor r,
                                  torch::Tensor w, torch::Tensor inv) {
  validate(x, r, w);
  TORCH_CHECK(dy.is_cuda() && dy.device() == x.device() && dy.sizes() == x.sizes() &&
              dy.scalar_type() == x.scalar_type() && dy.is_contiguous(), "invalid output gradient");
  TORCH_CHECK(inv.is_cuda() && inv.device() == x.device() && inv.scalar_type() == at::kFloat &&
              inv.is_contiguous() && inv.numel() == x.numel() / x.size(-1), "invalid inverse RMS");
  return rms_backward_cuda(dy, x, r, w, inv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward);
  m.def("backward", &backward);
}
