#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Fixed power-of-two block, strided rows, fp32 accumulation, no global atomics.
constexpr int THREADS = 256;

__device__ float block_sum(float value) {
  __shared__ float scratch[THREADS];
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (int stride = THREADS / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    __syncthreads();
  }
  return scratch[0];
}

template <typename T>
__global__ void forward_kernel(const T* x, const T* r, const T* w, T* y, float* inv,
                               int width, float eps) {
  const int row = blockIdx.x;
  const int64_t base = int64_t(row) * width;
  float total = 0;
  for (int col = threadIdx.x; col < width; col += THREADS) {
    float z = float(x[base + col]) + float(r[base + col]);
    total += z * z;
  }
  float scale = rsqrtf(block_sum(total) / width + eps);
  if (threadIdx.x == 0) inv[row] = scale;
  for (int col = threadIdx.x; col < width; col += THREADS) {
    float z = float(x[base + col]) + float(r[base + col]);
    y[base + col] = T(z * scale * float(w[col]));
  }
}

template <typename T>
__global__ void dx_kernel(const T* dy, const T* x, const T* r, const T* w,
                          const float* inv, T* dx, int width) {
  const int row = blockIdx.x;
  const int64_t base = int64_t(row) * width;
  float dot = 0;
  for (int col = threadIdx.x; col < width; col += THREADS) {
    float z = float(x[base + col]) + float(r[base + col]);
    dot += float(dy[base + col]) * float(w[col]) * z;
  }
  float scale = inv[row];
  float correction = block_sum(dot) * scale * scale / width;
  for (int col = threadIdx.x; col < width; col += THREADS) {
    float z = float(x[base + col]) + float(r[base + col]);
    dx[base + col] = T(scale * (float(dy[base + col]) * float(w[col]) - z * correction));
  }
}

// Two-pass deterministic weight reduction avoids rows*width temporary storage and
// floating-point atomic contention. A tile contains up to 256 rows.
template <typename T>
__global__ void dw_partial_kernel(const T* dy, const T* x, const T* r, const float* inv,
                                 float* partial, int rows, int width) {
  const int col = blockIdx.x * THREADS + threadIdx.x;
  const int tile = blockIdx.y;
  if (col >= width) return;
  float sum = 0;
  int end = min(rows, (tile + 1) * 256);
  for (int row = tile * 256; row < end; ++row) {
    int64_t i = int64_t(row) * width + col;
    sum += float(dy[i]) * (float(x[i]) + float(r[i])) * inv[row];
  }
  partial[int64_t(tile) * width + col] = sum;
}

std::vector<torch::Tensor> rms_forward_cuda(torch::Tensor x, torch::Tensor r,
                                         torch::Tensor w, double eps) {
  c10::cuda::CUDAGuard guard(x.device());
  int width = x.size(-1), rows = x.numel() / width;
  auto y = torch::empty_like(x);
  auto inv = torch::empty({rows}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "tailtrace_rms_forward", [&] {
    forward_kernel<scalar_t><<<rows, THREADS, 0, stream>>>(x.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(),
        w.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), inv.data_ptr<float>(), width, float(eps));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {y, inv};
}

