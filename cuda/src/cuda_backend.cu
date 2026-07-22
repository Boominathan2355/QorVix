// Real CUDA backend — compiled by nvcc only when QORVIX_ENABLE_CUDA is on and a toolkit is
// found (see cuda/CMakeLists.txt). The CPU stub (cuda_backend_stub.cpp) provides the same
// symbols otherwise; the two are never compiled together.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "qorvix/cuda/backend.hpp"
#include "qorvix/cuda/gpu_memory.hpp"
#include "qorvix/memory/disk_allocator.hpp"
#include "qorvix/memory/slab_allocator.hpp"

namespace qorvix::cuda {

namespace {

std::string cudaErr(cudaError_t e) { return cudaGetErrorString(e); }

// Elementwise scale — the minimal kernel that proves nvcc built device code and a launch works.
__global__ void scaleKernel(float* data, float factor, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] * factor;
}

// Native quantized GEMV: out[r] = dot(dequant(Q8_0 row r), x). One WARP per output row; the 32
// lanes cooperatively process each 32-element block — lane `l` reads quant `l` (so the warp's
// reads of a block's 32 int8 quants and of x are coalesced, one transaction each) and the fp16
// scale is broadcast from lane 0. A warp-shuffle reduction sums each block; results accumulate
// across blocks. The weight stays Q8_0 in VRAM. This replaces the earlier block-per-row kernel,
// whose per-thread strided 34-byte reads were uncoalesced (~6% of peak bandwidth on a T4).
constexpr int kQ8Block = 32;
constexpr int kQ8Bytes = 34;      // fp16 scale (2) + 32 int8
constexpr int kWarpsPerBlock = 8;  // 256 threads/block

__global__ void qmatmulQ8_0Kernel(float* __restrict__ out, const std::uint8_t* __restrict__ W,
                                  const float* __restrict__ x, int rows, int cols) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int nBlocks = cols / kQ8Block;
  const std::size_t rowBytes = static_cast<std::size_t>(nBlocks) * kQ8Bytes;
  const std::uint8_t* rowPtr = W + static_cast<std::size_t>(row) * rowBytes;

  float sum = 0.0f;
  for (int b = 0; b < nBlocks; ++b) {
    const std::uint8_t* blk = rowPtr + static_cast<std::size_t>(b) * kQ8Bytes;
    float d = 0.0f;
    if (lane == 0) {
      const unsigned short hbits = blk[0] | (static_cast<unsigned short>(blk[1]) << 8);
      d = __half2float(__ushort_as_half(hbits));
    }
    d = __shfl_sync(0xffffffffu, d, 0);  // broadcast the scale to the warp
    const signed char q = static_cast<signed char>(blk[2 + lane]);       // coalesced across lanes
    const float xv = x[static_cast<std::size_t>(b) * kQ8Block + lane];   // coalesced across lanes
    sum += d * static_cast<float>(q) * xv;
  }

  // Warp-shuffle reduction (no shared memory needed).
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) out[row] = sum;
}

// Launch geometry for the warp-per-row kernel.
inline int qmatmulGridBlocks(int rows) { return (rows + kWarpsPerBlock - 1) / kWarpsPerBlock; }

// ---- forward-pass elementwise / norm kernels (building blocks for on-GPU inference) --------
// These match the CPU reference ops (runtime/ops.cpp) so the GPU forward pass reproduces the
// validated math. Each is verified against a host reference in opsSelfTest().

// RMSNorm over one vector: out[i] = x[i] * rsqrt(mean(x^2)+eps) * w[i]. One block.
__global__ void rmsnormKernel(float* __restrict__ out, const float* __restrict__ x,
                              const float* __restrict__ w, int n, float eps) {
  __shared__ float red[256];
  __shared__ float scale;
  float local = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local += x[i] * x[i];
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) scale = rsqrtf(red[0] / n + eps);
  __syncthreads();
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * scale * w[i];
}

// RoPE (NeoX: rotate split-half pairs (i, i+head_dim/2)) over a [n_heads*head_dim] vector.
__global__ void ropeNeoxKernel(float* __restrict__ vec, int n_heads, int head_dim, int pos,
                               float freqBase) {
  const int half = head_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per (head, pair)
  if (idx >= n_heads * half) return;
  const int h = idx / half;
  const int i = idx % half;
  float* v = vec + static_cast<std::size_t>(h) * head_dim;
  const float theta = pos * powf(freqBase, -2.0f * i / head_dim);
  const float c = cosf(theta), s = sinf(theta);
  const float a = v[i], b = v[i + half];
  v[i] = a * c - b * s;
  v[i + half] = a * s + b * c;
}

// SwiGLU: out[i] = silu(gate[i]) * up[i], silu(z) = z * sigmoid(z).
__global__ void swigluKernel(float* __restrict__ out, const float* __restrict__ gate,
                             const float* __restrict__ up, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = gate[i];
  out[i] = (g / (1.0f + expf(-g))) * up[i];
}

// Residual add: out[i] += x[i].
__global__ void addKernel(float* __restrict__ out, const float* __restrict__ x, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] += x[i];
}

// Single-query (decode-step) attention with a cached K/V that live in VRAM. One block per query
// head (blockDim == headDim); GQA maps query head h -> kv head h/(nHeads/nKv). Computes
// softmax(qK^T / sqrt(headDim)) V over cached positions 0..seqLen-1. Correct-and-clear (per-t
// block reduce + serial softmax); FlashAttention-style tiling is the later optimization.
// Shared memory: (seqLen + headDim) floats.
__global__ void attentionDecodeKernel(float* __restrict__ out, const float* __restrict__ q,
                                      const float* __restrict__ K, const float* __restrict__ V,
                                      int nHeads, int nKv, int headDim, int seqLen, int kvDim) {
  const int h = blockIdx.x;
  if (h >= nHeads) return;
  const int kvHead = h / (nHeads / nKv);
  const int tid = threadIdx.x;  // 0..headDim-1
  const float* qh = q + static_cast<std::size_t>(h) * headDim;
  const float invSqrt = rsqrtf(static_cast<float>(headDim));

  extern __shared__ float sh[];
  float* scores = sh;               // seqLen
  float* red = sh + seqLen;         // headDim (reduction scratch)

  // scores[t] = (qh . K[t][kvHead]) * invSqrt
  for (int t = 0; t < seqLen; ++t) {
    const float* kt = K + static_cast<std::size_t>(t) * kvDim + kvHead * headDim;
    red[tid] = qh[tid] * kt[tid];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) red[tid] += red[tid + s];
      __syncthreads();
    }
    if (tid == 0) scores[t] = red[0] * invSqrt;
    __syncthreads();
  }

  // softmax over scores (serial on lane 0 — seqLen is small per decode step)
  if (tid == 0) {
    float mx = scores[0];
    for (int t = 1; t < seqLen; ++t) mx = fmaxf(mx, scores[t]);
    float sum = 0.0f;
    for (int t = 0; t < seqLen; ++t) {
      scores[t] = expf(scores[t] - mx);
      sum += scores[t];
    }
    const float inv = 1.0f / sum;
    for (int t = 0; t < seqLen; ++t) scores[t] *= inv;
  }
  __syncthreads();

  // out[h][d] = sum_t scores[t] * V[t][kvHead][d]
  float acc = 0.0f;
  for (int t = 0; t < seqLen; ++t) {
    const float* vt = V + static_cast<std::size_t>(t) * kvDim + kvHead * headDim;
    acc += scores[t] * vt[tid];
  }
  out[static_cast<std::size_t>(h) * headDim + tid] = acc;
}

}  // namespace

bool builtWithCuda() noexcept { return true; }

int deviceCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
  return count;
}

std::vector<DeviceInfo> enumerateDevices() {
  std::vector<DeviceInfo> out;
  const int count = deviceCount();
  for (int i = 0; i < count; ++i) {
    if (auto info = deviceInfo(i)) out.push_back(*info);
  }
  return out;
}

std::optional<DeviceInfo> deviceInfo(int index) {
  if (index < 0 || index >= deviceCount()) return std::nullopt;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) return std::nullopt;

  DeviceInfo info;
  info.index = index;
  info.name = prop.name;
  info.totalGlobalMem = prop.totalGlobalMem;
  info.computeMajor = prop.major;
  info.computeMinor = prop.minor;
  info.multiProcessorCount = prop.multiProcessorCount;

  // Free memory is queried per active device.
  int previous = 0;
  cudaGetDevice(&previous);
  if (cudaSetDevice(index) == cudaSuccess) {
    std::size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) == cudaSuccess) info.freeMem = freeBytes;
    cudaSetDevice(previous);
  }
  return info;
}

bool setDevice(int index) { return cudaSetDevice(index) == cudaSuccess; }

SelfTestResult selfTest() {
  if (deviceCount() <= 0) return {false, false, "no CUDA device present"};

  constexpr int n = 1024;
  std::vector<float> host(n), out(n);
  for (int i = 0; i < n; ++i) host[i] = static_cast<float>(i);

  float* d = nullptr;
  cudaError_t e = cudaMalloc(&d, n * sizeof(float));
  if (e != cudaSuccess) return {true, false, "cudaMalloc: " + cudaErr(e)};

  cudaMemcpy(d, host.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  scaleKernel<<<blocks, threads>>>(d, 2.0f, n);
  e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    cudaFree(d);
    return {true, false, "kernel launch: " + cudaErr(e)};
  }
  cudaMemcpy(out.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d);

  for (int i = 0; i < n; ++i) {
    if (out[i] != host[i] * 2.0f) return {true, false, "result verification failed"};
  }
  return {true, true, "scale kernel host<->device round-trip verified over 1024 floats"};
}

SelfTestResult gemmSelfTest() {
  if (deviceCount() <= 0) return {false, false, "no CUDA device present"};

  constexpr int n = 4;
  constexpr int elems = n * n;
  // Column-major. A = identity, so C = A*B must equal B.
  std::vector<float> A(elems, 0.0f), B(elems), C(elems, 0.0f);
  for (int i = 0; i < n; ++i) A[i * n + i] = 1.0f;
  for (int i = 0; i < elems; ++i) B[i] = static_cast<float>(i + 1);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  if (cudaMalloc(&dA, elems * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&dB, elems * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&dC, elems * sizeof(float)) != cudaSuccess) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return {true, false, "cudaMalloc failed for GEMM operands"};
  }
  cudaMemcpy(dA, A.data(), elems * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), elems * sizeof(float), cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return {true, false, "cublasCreate failed"};
  }
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA, n,
                                        dB, n, &beta, dC, n);
  cublasDestroy(handle);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return {true, false, "cublasSgemm failed"};
  }
  cudaMemcpy(C.data(), dC, elems * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  for (int i = 0; i < elems; ++i) {
    if (std::fabs(C[i] - B[i]) > 1e-3f) return {true, false, "GEMM result mismatch"};
  }
  return {true, true, "cuBLAS SGEMM (A=I) verified on a 4x4 matrix"};
}

SelfTestResult qmatmulSelfTest() {
  if (deviceCount() <= 0) return {false, false, "no CUDA device present"};

  auto encodeHalf = [](float f) -> unsigned short {
    __half h = __float2half(f);
    unsigned short bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
  };
  auto decodeHalf = [](unsigned short bits) -> float {
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
  };

  // ---- correctness on a small Q8_0 matrix vs a host reference ----
  const int rows = 64, cols = 256;
  const int nBlocks = cols / kQ8Block;
  const std::size_t rowBytes = static_cast<std::size_t>(nBlocks) * kQ8Bytes;
  std::vector<std::uint8_t> W(static_cast<std::size_t>(rows) * rowBytes, 0);
  std::vector<float> x(cols), ref(rows, 0.0f), gpu(rows, 0.0f);

  for (int i = 0; i < cols; ++i) x[i] = 0.1f * ((i % 5) - 2);
  for (int r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (int b = 0; b < nBlocks; ++b) {
      std::uint8_t* blk = W.data() + static_cast<std::size_t>(r) * rowBytes + static_cast<std::size_t>(b) * kQ8Bytes;
      const unsigned short hs = encodeHalf(0.02f * ((r % 7) + 1));
      blk[0] = static_cast<std::uint8_t>(hs & 0xFF);
      blk[1] = static_cast<std::uint8_t>(hs >> 8);
      const float d = decodeHalf(hs);  // use the rounded scale so the reference matches the kernel
      for (int i = 0; i < kQ8Block; ++i) {
        const signed char q = static_cast<signed char>(((r + b + i) % 15) - 7);
        blk[2 + i] = static_cast<std::uint8_t>(q);
        acc += d * static_cast<float>(q) * x[b * kQ8Block + i];
      }
    }
    ref[r] = static_cast<float>(acc);
  }

  std::uint8_t* dW = nullptr;
  float* dX = nullptr;
  float* dOut = nullptr;
  if (cudaMalloc(&dW, W.size()) != cudaSuccess || cudaMalloc(&dX, cols * sizeof(float)) != cudaSuccess ||
      cudaMalloc(&dOut, rows * sizeof(float)) != cudaSuccess) {
    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dOut);
    return {true, false, "cudaMalloc failed for qmatmul operands"};
  }
  cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dX, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

  qmatmulQ8_0Kernel<<<qmatmulGridBlocks(rows), kWarpsPerBlock * 32>>>(dOut, dW, dX, rows, cols);
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dOut);
    return {true, false, "qmatmul kernel launch: " + cudaErr(e)};
  }
  cudaMemcpy(gpu.data(), dOut, rows * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dW);
  cudaFree(dX);
  cudaFree(dOut);

  float maxErr = 0.0f;
  for (int r = 0; r < rows; ++r) maxErr = std::max(maxErr, std::fabs(gpu[r] - ref[r]));
  if (maxErr > 1e-2f) {
    return {true, false, "GPU qmatmul disagrees with host reference (max err " +
                             std::to_string(maxErr) + ")"};
  }

  // ---- throughput on a large Q8_0 matrix (content irrelevant here) ----
  const int R = 4096, C = 4096;
  const std::size_t bigRowBytes = static_cast<std::size_t>(C / kQ8Block) * kQ8Bytes;
  const std::size_t bigW = static_cast<std::size_t>(R) * bigRowBytes;
  std::uint8_t* dbW = nullptr;
  float *dbX = nullptr, *dbOut = nullptr;
  std::string timing;
  if (cudaMalloc(&dbW, bigW) == cudaSuccess && cudaMalloc(&dbX, C * sizeof(float)) == cudaSuccess &&
      cudaMalloc(&dbOut, R * sizeof(float)) == cudaSuccess) {
    cudaMemset(dbW, 1, bigW);
    cudaMemset(dbX, 0, C * sizeof(float));
    const int grid = qmatmulGridBlocks(R), threads = kWarpsPerBlock * 32;
    qmatmulQ8_0Kernel<<<grid, threads>>>(dbOut, dbW, dbX, R, C);  // warm up
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    const int iters = 50;
    cudaEventRecord(t0);
    for (int it = 0; it < iters; ++it) qmatmulQ8_0Kernel<<<grid, threads>>>(dbOut, dbW, dbX, R, C);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    const double sec = ms / 1000.0;
    const double gflops = 2.0 * R * C * iters / sec / 1e9;
    const double gbps = static_cast<double>(bigW) * iters / sec / 1e9;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "; %dx%d Q8_0 GEMV: %.0f GFLOP/s, %.0f GB/s", R, C, gflops, gbps);
    timing = buf;
  }
  cudaFree(dbW);
  cudaFree(dbX);
  cudaFree(dbOut);

  return {true, true, "GPU Q8_0 matmul matches host reference (max err " + std::to_string(maxErr) +
                          ")" + timing};
}

SelfTestResult opsSelfTest() {
  if (deviceCount() <= 0) return {false, false, "no CUDA device present"};

  auto upload = [](const std::vector<float>& v) {
    float* d = nullptr;
    cudaMalloc(&d, v.size() * sizeof(float));
    cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
  };
  auto download = [](float* d, std::vector<float>& v) {
    cudaMemcpy(v.data(), d, v.size() * sizeof(float), cudaMemcpyDeviceToHost);
  };
  float maxErr = 0.0f;
  auto track = [&](const std::vector<float>& a, const std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) maxErr = std::max(maxErr, std::fabs(a[i] - b[i]));
  };

  // RMSNorm.
  {
    const int n = 512;
    std::vector<float> x(n), w(n), ref(n), gpu(n);
    for (int i = 0; i < n; ++i) {
      x[i] = 0.01f * ((i % 13) - 6);
      w[i] = 0.5f + 0.001f * i;
    }
    double sq = 0.0;
    for (float v : x) sq += static_cast<double>(v) * v;
    const float scale = 1.0f / std::sqrt(static_cast<float>(sq / n) + 1e-5f);
    for (int i = 0; i < n; ++i) ref[i] = x[i] * scale * w[i];
    float *dx = upload(x), *dw = upload(w), *dout = upload(gpu);
    rmsnormKernel<<<1, 256>>>(dout, dx, dw, n, 1e-5f);
    cudaDeviceSynchronize();
    download(dout, gpu);
    cudaFree(dx);
    cudaFree(dw);
    cudaFree(dout);
    track(ref, gpu);
  }
  // RoPE (NeoX).
  {
    const int nH = 4, hd = 8, half = hd / 2, pos = 3, n = nH * hd;
    const float base = 10000.0f;
    std::vector<float> v(n), ref(n), gpu(n);
    for (int i = 0; i < n; ++i) v[i] = 0.1f * ((i % 7) - 3);
    ref = v;
    for (int h = 0; h < nH; ++h)
      for (int i = 0; i < half; ++i) {
        const float theta = pos * std::pow(base, -2.0f * i / hd);
        const float c = std::cos(theta), s = std::sin(theta);
        const float a = ref[h * hd + i], b = ref[h * hd + i + half];
        ref[h * hd + i] = a * c - b * s;
        ref[h * hd + i + half] = a * s + b * c;
      }
    float* dv = upload(v);
    ropeNeoxKernel<<<(nH * half + 63) / 64, 64>>>(dv, nH, hd, pos, base);
    cudaDeviceSynchronize();
    download(dv, gpu);
    cudaFree(dv);
    track(ref, gpu);
  }
  // SwiGLU.
  {
    const int n = 1000;
    std::vector<float> g(n), u(n), ref(n), gpu(n);
    for (int i = 0; i < n; ++i) {
      g[i] = 0.02f * ((i % 11) - 5);
      u[i] = 0.03f * ((i % 9) - 4);
      ref[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
    }
    float *dg = upload(g), *du = upload(u), *dout = upload(gpu);
    swigluKernel<<<(n + 255) / 256, 256>>>(dout, dg, du, n);
    cudaDeviceSynchronize();
    download(dout, gpu);
    cudaFree(dg);
    cudaFree(du);
    cudaFree(dout);
    track(ref, gpu);
  }
  // Residual add.
  {
    const int n = 1000;
    std::vector<float> a(n), b(n), ref(n), gpu(n);
    for (int i = 0; i < n; ++i) {
      a[i] = 0.1f * i;
      b[i] = -0.05f * i;
      ref[i] = a[i] + b[i];
    }
    float *da = upload(a), *db = upload(b);
    addKernel<<<(n + 255) / 256, 256>>>(da, db, n);
    cudaDeviceSynchronize();
    download(da, gpu);
    cudaFree(da);
    cudaFree(db);
    track(ref, gpu);
  }

  const bool ok = maxErr < 1e-3f;
  return {true, ok,
          (ok ? "rmsnorm/rope/swiglu/add match CPU (max err " : "GPU ops disagree with CPU (max err ") +
              std::to_string(maxErr) + ")"};
}

SelfTestResult attentionSelfTest() {
  if (deviceCount() <= 0) return {false, false, "no CUDA device present"};

  // GQA: 4 query heads, 2 kv heads, head_dim 8, 5 cached positions.
  const int nHeads = 4, nKv = 2, headDim = 8, seqLen = 5;
  const int kvDim = nKv * headDim;
  const int group = nHeads / nKv;
  std::vector<float> q(static_cast<std::size_t>(nHeads) * headDim);
  std::vector<float> K(static_cast<std::size_t>(seqLen) * kvDim);
  std::vector<float> V(static_cast<std::size_t>(seqLen) * kvDim);
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.05f * ((i % 11) - 5);
  for (std::size_t i = 0; i < K.size(); ++i) K[i] = 0.03f * ((i % 13) - 6);
  for (std::size_t i = 0; i < V.size(); ++i) V[i] = 0.04f * ((i % 7) - 3);

  // CPU reference.
  std::vector<float> ref(static_cast<std::size_t>(nHeads) * headDim, 0.0f), gpu(ref.size());
  const float invSqrt = 1.0f / std::sqrt(static_cast<float>(headDim));
  for (int h = 0; h < nHeads; ++h) {
    const int kvHead = h / group;
    std::vector<float> sc(seqLen);
    for (int t = 0; t < seqLen; ++t) {
      float dot = 0.0f;
      for (int d = 0; d < headDim; ++d)
        dot += q[h * headDim + d] * K[t * kvDim + kvHead * headDim + d];
      sc[t] = dot * invSqrt;
    }
    float mx = sc[0];
    for (int t = 1; t < seqLen; ++t) mx = std::max(mx, sc[t]);
    float sum = 0.0f;
    for (int t = 0; t < seqLen; ++t) {
      sc[t] = std::exp(sc[t] - mx);
      sum += sc[t];
    }
    for (int d = 0; d < headDim; ++d) {
      float acc = 0.0f;
      for (int t = 0; t < seqLen; ++t)
        acc += (sc[t] / sum) * V[t * kvDim + kvHead * headDim + d];
      ref[h * headDim + d] = acc;
    }
  }

  float *dq = nullptr, *dK = nullptr, *dV = nullptr, *dout = nullptr;
  cudaMalloc(&dq, q.size() * sizeof(float));
  cudaMalloc(&dK, K.size() * sizeof(float));
  cudaMalloc(&dV, V.size() * sizeof(float));
  cudaMalloc(&dout, gpu.size() * sizeof(float));
  cudaMemcpy(dq, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, K.data(), K.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, V.data(), V.size() * sizeof(float), cudaMemcpyHostToDevice);

  const std::size_t shBytes = (seqLen + headDim) * sizeof(float);
  attentionDecodeKernel<<<nHeads, headDim, shBytes>>>(dout, dq, dK, dV, nHeads, nKv, headDim,
                                                      seqLen, kvDim);
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  cudaMemcpy(gpu.data(), dout, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dq);
  cudaFree(dK);
  cudaFree(dV);
  cudaFree(dout);
  if (e != cudaSuccess) return {true, false, "attention kernel launch: " + cudaErr(e)};

  float maxErr = 0.0f;
  for (std::size_t i = 0; i < ref.size(); ++i) maxErr = std::max(maxErr, std::fabs(gpu[i] - ref[i]));
  const bool ok = maxErr < 1e-3f;
  return {true, ok,
          (ok ? "GQA decode attention matches CPU (max err " : "attention disagrees with CPU (max err ") +
              std::to_string(maxErr) + ")"};
}

// ---- memory integration --------------------------------------------------------------------

namespace {

class CudaSlabAllocator final : public memory::ISlabAllocator {
 public:
  void* allocate(std::size_t bytes) override {
    if (bytes == 0) return nullptr;
    void* p = nullptr;
    return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
  }
  void deallocate(void* ptr, std::size_t /*bytes*/) noexcept override {
    if (ptr) cudaFree(ptr);
  }
  memory::Tier tier() const noexcept override { return memory::Tier::GpuVram; }
};

class CudaTransferEngine final : public memory::ITransferEngine {
 public:
  void copy(void* dst, memory::Tier dstTier, const void* src, memory::Tier srcTier,
            std::size_t bytes) override {
    const bool dstGpu = dstTier == memory::Tier::GpuVram;
    const bool srcGpu = srcTier == memory::Tier::GpuVram;
    if (!dstGpu && !srcGpu) {
      std::memcpy(dst, src, bytes);
      return;
    }
    const cudaMemcpyKind kind = (dstGpu && srcGpu) ? cudaMemcpyDeviceToDevice
                                : dstGpu           ? cudaMemcpyHostToDevice
                                                   : cudaMemcpyDeviceToHost;
    cudaMemcpy(dst, src, bytes, kind);
  }
};

}  // namespace

std::unique_ptr<memory::ISlabAllocator> makeGpuSlabAllocator() {
  if (deviceCount() <= 0) return nullptr;
  return std::make_unique<CudaSlabAllocator>();
}

std::unique_ptr<memory::ITransferEngine> makeCudaTransferEngine() {
  return std::make_unique<CudaTransferEngine>();
}

std::unique_ptr<memory::MemoryManager> makeGpuMemoryManager(std::size_t vramBudgetBytes,
                                                            std::size_t ramBudgetBytes,
                                                            std::size_t diskBudgetBytes,
                                                            const std::filesystem::path& spoolDir) {
  auto gpu = makeGpuSlabAllocator();
  if (!gpu) return nullptr;

  std::map<memory::Tier, memory::MemoryManager::TierSpec> tiers;
  tiers[memory::Tier::GpuVram] = {std::move(gpu),
                                  memory::PoolConfig{.budgetBytes = vramBudgetBytes}};
  tiers[memory::Tier::HostRam] = {std::make_unique<memory::HostSlabAllocator>(),
                                  memory::PoolConfig{.budgetBytes = ramBudgetBytes}};
  tiers[memory::Tier::DiskNvme] = {std::make_unique<memory::DiskSlabAllocator>(spoolDir),
                                   memory::PoolConfig{.budgetBytes = diskBudgetBytes}};
  return std::make_unique<memory::MemoryManager>(std::move(tiers), makeCudaTransferEngine());
}

}  // namespace qorvix::cuda
