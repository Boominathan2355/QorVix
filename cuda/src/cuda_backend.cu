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
#include "qorvix/cuda/gpu_model.hpp"
#include "qorvix/memory/disk_allocator.hpp"
#include "qorvix/memory/slab_allocator.hpp"

#include <memory>

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

// ---- K-quant matmul kernels (Q4_K, Q6_K) — the types real GGUF models actually use -----------
// Warp-per-row GEMV that dequantizes 256-element super-blocks on the fly, mirroring the CPU
// dequant (runtime/dequant.cpp). Correctness-first (per-element dequant + warp reduce); the same
// tuning that the Q8_0 kernel wants applies later.
constexpr int kQKK = 256;      // K-quant super-block
constexpr int kQ4KBytes = 144;  // Q4_K: d(2) + dmin(2) + scales[12] + qs[128]
constexpr int kQ6KBytes = 210;  // Q6_K: ql[128] + qh[64] + scales[16] + d(2)

// 6-bit (scale, min) unpack for sub-block j of a Q4_K/Q5_K scales array. Verbatim from ggml.
__device__ inline void getScaleMinK4(int j, const unsigned char* q, unsigned char& d,
                                     unsigned char& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

__global__ void qmatmulQ4_KKernel(float* __restrict__ out, const std::uint8_t* __restrict__ W,
                                  const float* __restrict__ x, int rows, int cols) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const int nSB = cols / kQKK;
  const std::size_t rowBytes = static_cast<std::size_t>(nSB) * kQ4KBytes;
  const std::uint8_t* rowPtr = W + static_cast<std::size_t>(row) * rowBytes;

  float sum = 0.0f;
  for (int sb = 0; sb < nSB; ++sb) {
    const std::uint8_t* blk = rowPtr + static_cast<std::size_t>(sb) * kQ4KBytes;
    const float d = __half2float(__ushort_as_half(blk[0] | (static_cast<unsigned short>(blk[1]) << 8)));
    const float dmin = __half2float(__ushort_as_half(blk[2] | (static_cast<unsigned short>(blk[3]) << 8)));
    const std::uint8_t* scales = blk + 4;
    const std::uint8_t* qs = blk + 16;
    const float* xb = x + static_cast<std::size_t>(sb) * kQKK;
    for (int i = lane; i < kQKK; i += 32) {
      const int s = i / 32, chunk = i / 64, local = i % 64;
      const std::uint8_t qbyte = qs[chunk * 32 + (local & 31)];
      const int nib = (local < 32) ? (qbyte & 0xF) : (qbyte >> 4);
      unsigned char sc, m;
      getScaleMinK4(s, scales, sc, m);
      sum += (d * sc * nib - dmin * m) * xb[i];
    }
  }
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) out[row] = sum;
}

__global__ void qmatmulQ6_KKernel(float* __restrict__ out, const std::uint8_t* __restrict__ W,
                                  const float* __restrict__ x, int rows, int cols) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) return;
  const int nSB = cols / kQKK;
  const std::size_t rowBytes = static_cast<std::size_t>(nSB) * kQ6KBytes;
  const std::uint8_t* rowPtr = W + static_cast<std::size_t>(row) * rowBytes;

  float sum = 0.0f;
  for (int sb = 0; sb < nSB; ++sb) {
    const std::uint8_t* blk = rowPtr + static_cast<std::size_t>(sb) * kQ6KBytes;
    const std::uint8_t* ql = blk;
    const std::uint8_t* qh = blk + 128;
    const signed char* sc = reinterpret_cast<const signed char*>(blk + 192);
    const float d = __half2float(__ushort_as_half(blk[208] | (static_cast<unsigned short>(blk[209]) << 8)));
    const float* xb = x + static_cast<std::size_t>(sb) * kQKK;
    for (int i = lane; i < kQKK; i += 32) {
      const int half = i >> 7, p = i & 127, l = p & 31, quarter = p >> 5, is = l >> 4;
      const std::uint8_t* qlh = ql + half * 64;
      const std::uint8_t* qhh = qh + half * 32;
      const signed char* sch = sc + half * 8;
      int lo, hi;
      if (quarter == 0) { lo = qlh[l] & 0xF;           hi = (qhh[l] >> 0) & 3; }
      else if (quarter == 1) { lo = qlh[l + 32] & 0xF; hi = (qhh[l] >> 2) & 3; }
      else if (quarter == 2) { lo = qlh[l] >> 4;       hi = (qhh[l] >> 4) & 3; }
      else { lo = qlh[l + 32] >> 4;                    hi = (qhh[l] >> 6) & 3; }
      const int q = (lo | (hi << 4)) - 32;
      sum += d * static_cast<float>(sch[is + quarter * 2]) * q * xb[i];
    }
  }
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
  if (lane == 0) out[row] = sum;
}

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

// F32 GEMV: out[r] = dot(W[r,:], x). One block per row, block-reduce. Used by the forward driver
// for its (F32) weights; quantized weights use qmatmulQ8_0Kernel.
__global__ void matmulF32Kernel(float* __restrict__ out, const float* __restrict__ W,
                                const float* __restrict__ x, int rows, int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const float* r = W + static_cast<std::size_t>(row) * cols;
  float partial = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) partial += r[c] * x[c];
  __shared__ float sh[256];
  sh[threadIdx.x] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[row] = sh[0];
}

// Copy one embedding row: dst[i] = table[token*d + i]. (A kernel keeps everything device-side.)
__global__ void embedRowKernel(float* __restrict__ dst, const float* __restrict__ table, int token,
                               int d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) dst[i] = table[static_cast<std::size_t>(token) * d + i];
}

// Copy a kvDim vector into the KV cache slot for (layer, pos): cache[(layer*maxSeq+pos)*kvDim+i].
__global__ void kvStoreKernel(float* __restrict__ cache, const float* __restrict__ src, int layer,
                              int pos, int maxSeq, int kvDim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kvDim) {
    cache[(static_cast<std::size_t>(layer) * maxSeq + pos) * kvDim + i] = src[i];
  }
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

SelfTestResult qmatmulQ4_KSelfTest() {
  if (deviceCount() <= 0) return {false, false, "no CUDA device present"};

  auto encodeHalf = [](float f) {
    __half h = __float2half(f);
    unsigned short b;
    std::memcpy(&b, &h, 2);
    return b;
  };
  auto decodeHalf = [](unsigned short b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
  };
  auto hostScaleMin = [](int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
      d = q[j] & 63;
      m = q[j + 4] & 63;
    } else {
      d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
      m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
  };

  const int rows = 32, cols = kQKK;  // one super-block per row
  std::vector<std::uint8_t> W(static_cast<std::size_t>(rows) * kQ4KBytes);
  std::vector<float> x(cols), ref(rows, 0.0f), gpu(rows, 0.0f);
  for (int i = 0; i < cols; ++i) x[i] = 0.02f * ((i % 9) - 4);

  for (int r = 0; r < rows; ++r) {
    std::uint8_t* blk = W.data() + static_cast<std::size_t>(r) * kQ4KBytes;
    const unsigned short dh = encodeHalf(0.02f * (r % 5 + 1));
    const unsigned short mh = encodeHalf(0.01f);
    blk[0] = dh & 0xFF; blk[1] = dh >> 8;
    blk[2] = mh & 0xFF; blk[3] = mh >> 8;
    for (int i = 0; i < 12; ++i) blk[4 + i] = static_cast<std::uint8_t>((r * 3 + i * 5) & 0x3F);
    for (int i = 0; i < 128; ++i) blk[16 + i] = static_cast<std::uint8_t>((r + i) & 0xFF);

    const float d = decodeHalf(dh), dmin = decodeHalf(mh);
    const std::uint8_t* scales = blk + 4;
    const std::uint8_t* qs = blk + 16;
    double acc = 0.0;
    for (int i = 0; i < kQKK; ++i) {
      const int s = i / 32, chunk = i / 64, local = i % 64;
      const std::uint8_t qbyte = qs[chunk * 32 + (local & 31)];
      const int nib = (local < 32) ? (qbyte & 0xF) : (qbyte >> 4);
      std::uint8_t sc, m;
      hostScaleMin(s, scales, sc, m);
      acc += (static_cast<double>(d) * sc * nib - static_cast<double>(dmin) * m) * x[i];
    }
    ref[r] = static_cast<float>(acc);
  }

  std::uint8_t* dW = nullptr;
  float *dX = nullptr, *dOut = nullptr;
  cudaMalloc(&dW, W.size());
  cudaMalloc(&dX, cols * sizeof(float));
  cudaMalloc(&dOut, rows * sizeof(float));
  cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dX, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
  qmatmulQ4_KKernel<<<qmatmulGridBlocks(rows), kWarpsPerBlock * 32>>>(dOut, dW, dX, rows, cols);
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  cudaMemcpy(gpu.data(), dOut, rows * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dW);
  cudaFree(dX);
  cudaFree(dOut);
  if (e != cudaSuccess) return {true, false, "Q4_K kernel launch: " + cudaErr(e)};

  float maxErr = 0.0f, maxRef = 1e-6f;
  for (int r = 0; r < rows; ++r) {
    maxErr = std::max(maxErr, std::fabs(gpu[r] - ref[r]));
    maxRef = std::max(maxRef, std::fabs(ref[r]));
  }
  const bool ok = maxErr / maxRef < 1e-3f;
  return {true, ok,
          (ok ? "GPU Q4_K matmul matches host reference (rel err " : "Q4_K disagrees with CPU (rel err ") +
              std::to_string(maxErr / maxRef) + ")"};
}

SelfTestResult qmatmulQ6_KSelfTest() {
  if (deviceCount() <= 0) return {false, false, "no CUDA device present"};

  auto encodeHalf = [](float f) {
    __half h = __float2half(f);
    unsigned short b;
    std::memcpy(&b, &h, 2);
    return b;
  };
  auto decodeHalf = [](unsigned short b) {
    __half h;
    std::memcpy(&h, &b, 2);
    return __half2float(h);
  };

  const int rows = 32, cols = kQKK;
  std::vector<std::uint8_t> W(static_cast<std::size_t>(rows) * kQ6KBytes);
  std::vector<float> x(cols), ref(rows, 0.0f), gpu(rows, 0.0f);
  for (int i = 0; i < cols; ++i) x[i] = 0.02f * ((i % 9) - 4);

  for (int r = 0; r < rows; ++r) {
    std::uint8_t* blk = W.data() + static_cast<std::size_t>(r) * kQ6KBytes;
    for (int i = 0; i < 128; ++i) blk[i] = static_cast<std::uint8_t>((r + i) & 0xFF);        // ql
    for (int i = 0; i < 64; ++i) blk[128 + i] = static_cast<std::uint8_t>((r * 2 + i) & 0xFF); // qh
    for (int i = 0; i < 16; ++i) blk[192 + i] = static_cast<std::uint8_t>((r + i) % 9 - 4);    // int8 scales
    const unsigned short dh = encodeHalf(0.01f * (r % 4 + 1));
    blk[208] = dh & 0xFF; blk[209] = dh >> 8;

    const std::uint8_t* ql = blk;
    const std::uint8_t* qh = blk + 128;
    const signed char* sc = reinterpret_cast<const signed char*>(blk + 192);
    const float d = decodeHalf(dh);
    double acc = 0.0;
    for (int i = 0; i < kQKK; ++i) {
      const int half = i >> 7, p = i & 127, l = p & 31, quarter = p >> 5, is = l >> 4;
      const std::uint8_t* qlh = ql + half * 64;
      const std::uint8_t* qhh = qh + half * 32;
      const signed char* sch = sc + half * 8;
      int lo, hi;
      if (quarter == 0) { lo = qlh[l] & 0xF; hi = (qhh[l] >> 0) & 3; }
      else if (quarter == 1) { lo = qlh[l + 32] & 0xF; hi = (qhh[l] >> 2) & 3; }
      else if (quarter == 2) { lo = qlh[l] >> 4; hi = (qhh[l] >> 4) & 3; }
      else { lo = qlh[l + 32] >> 4; hi = (qhh[l] >> 6) & 3; }
      const int q = (lo | (hi << 4)) - 32;
      acc += static_cast<double>(d) * sch[is + quarter * 2] * q * x[i];
    }
    ref[r] = static_cast<float>(acc);
  }

  std::uint8_t* dW = nullptr;
  float *dX = nullptr, *dOut = nullptr;
  cudaMalloc(&dW, W.size());
  cudaMalloc(&dX, cols * sizeof(float));
  cudaMalloc(&dOut, rows * sizeof(float));
  cudaMemcpy(dW, W.data(), W.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dX, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice);
  qmatmulQ6_KKernel<<<qmatmulGridBlocks(rows), kWarpsPerBlock * 32>>>(dOut, dW, dX, rows, cols);
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  cudaMemcpy(gpu.data(), dOut, rows * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dW);
  cudaFree(dX);
  cudaFree(dOut);
  if (e != cudaSuccess) return {true, false, "Q6_K kernel launch: " + cudaErr(e)};

  float maxErr = 0.0f, maxRef = 1e-6f;
  for (int r = 0; r < rows; ++r) {
    maxErr = std::max(maxErr, std::fabs(gpu[r] - ref[r]));
    maxRef = std::max(maxRef, std::fabs(ref[r]));
  }
  const bool ok = maxErr / maxRef < 1e-3f;
  return {true, ok,
          (ok ? "GPU Q6_K matmul matches host reference (rel err " : "Q6_K disagrees with CPU (rel err ") +
              std::to_string(maxErr / maxRef) + ")"};
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

SelfTestResult gpuForwardSelfTest() {
  if (deviceCount() <= 0) return {false, false, "no CUDA device present"};

  // Tiny synthetic Llama-style model.
  const int L = 2, d = 32, nHeads = 4, headDim = 8, nKv = 2, ffn = 64, vocab = 16, maxSeq = 8;
  const int kvDim = nKv * headDim;           // 16
  const int qDim = nHeads * headDim;         // 32 == d
  const int group = nHeads / nKv;
  const float eps = 1e-5f, freqBase = 10000.0f;

  auto gen = [](int seed, int i) { return 0.01f * (((i * 7 + seed * 13) % 17) - 8); };

  // Host weights (shared by the CPU reference and uploaded to the GPU).
  std::vector<float> tokEmbd(vocab * d), outNorm(d, 1.0f), output(vocab * d);
  for (int i = 0; i < vocab * d; ++i) tokEmbd[i] = gen(1, i);
  for (int i = 0; i < vocab * d; ++i) output[i] = gen(2, i);
  for (int i = 0; i < d; ++i) outNorm[i] = 1.0f + 0.01f * i;

  struct Layer {
    std::vector<float> attnNorm, wq, wk, wv, wo, ffnNorm, ffnGate, ffnUp, ffnDown;
  };
  std::vector<Layer> layers(L);
  for (int l = 0; l < L; ++l) {
    Layer& w = layers[l];
    w.attnNorm.assign(d, 0.0f);
    w.ffnNorm.assign(d, 0.0f);
    for (int i = 0; i < d; ++i) {
      w.attnNorm[i] = 1.0f + 0.01f * ((i + l) % 5);
      w.ffnNorm[i] = 1.0f + 0.01f * ((i + l) % 3);
    }
    w.wq.resize(qDim * d);
    w.wo.resize(d * qDim);
    w.wk.resize(kvDim * d);
    w.wv.resize(kvDim * d);
    w.ffnGate.resize(ffn * d);
    w.ffnUp.resize(ffn * d);
    w.ffnDown.resize(d * ffn);
    for (int i = 0; i < qDim * d; ++i) w.wq[i] = gen(10 + l, i);
    for (int i = 0; i < d * qDim; ++i) w.wo[i] = gen(20 + l, i);
    for (int i = 0; i < kvDim * d; ++i) w.wk[i] = gen(30 + l, i);
    for (int i = 0; i < kvDim * d; ++i) w.wv[i] = gen(40 + l, i);
    for (int i = 0; i < ffn * d; ++i) w.ffnGate[i] = gen(50 + l, i);
    for (int i = 0; i < ffn * d; ++i) w.ffnUp[i] = gen(60 + l, i);
    for (int i = 0; i < d * ffn; ++i) w.ffnDown[i] = gen(70 + l, i);
  }

  // ---- CPU reference forward (mirrors runtime/TextModel) ----
  auto rms = [&](std::vector<float>& o, const std::vector<float>& x, const std::vector<float>& w, int n) {
    double sq = 0.0;
    for (int i = 0; i < n; ++i) sq += static_cast<double>(x[i]) * x[i];
    const float sc = 1.0f / std::sqrt(static_cast<float>(sq / n) + eps);
    for (int i = 0; i < n; ++i) o[i] = x[i] * sc * w[i];
  };
  auto mm = [&](std::vector<float>& o, const std::vector<float>& W, const std::vector<float>& x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
      double a = 0.0;
      for (int c = 0; c < cols; ++c) a += static_cast<double>(W[r * cols + c]) * x[c];
      o[r] = static_cast<float>(a);
    }
  };
  auto rope = [&](std::vector<float>& v, int nh, int pos) {
    const int half = headDim / 2;
    for (int h = 0; h < nh; ++h)
      for (int i = 0; i < half; ++i) {
        const float th = pos * std::pow(freqBase, -2.0f * i / headDim);
        const float c = std::cos(th), s = std::sin(th);
        const float a = v[h * headDim + i], b = v[h * headDim + i + half];
        v[h * headDim + i] = a * c - b * s;
        v[h * headDim + i + half] = a * s + b * c;
      }
  };
  std::vector<float> hKc(static_cast<std::size_t>(L) * maxSeq * kvDim, 0.0f), hVc = hKc;
  auto cpuForward = [&](int token, int pos, std::vector<float>& logits) {
    std::vector<float> x(tokEmbd.begin() + token * d, tokEmbd.begin() + token * d + d);
    std::vector<float> xn(d), q(qDim), k(kvDim), v(kvDim), attn(qDim), tmp(d), g(ffn), u(ffn), act(ffn);
    for (int l = 0; l < L; ++l) {
      Layer& w = layers[l];
      rms(xn, x, w.attnNorm, d);
      mm(q, w.wq, xn, qDim, d);
      mm(k, w.wk, xn, kvDim, d);
      mm(v, w.wv, xn, kvDim, d);
      rope(q, nHeads, pos);
      rope(k, nKv, pos);
      for (int i = 0; i < kvDim; ++i) {
        hKc[(static_cast<std::size_t>(l) * maxSeq + pos) * kvDim + i] = k[i];
        hVc[(static_cast<std::size_t>(l) * maxSeq + pos) * kvDim + i] = v[i];
      }
      const float invSqrt = 1.0f / std::sqrt(static_cast<float>(headDim));
      for (int h = 0; h < nHeads; ++h) {
        const int kvHead = h / group;
        std::vector<float> sc(pos + 1);
        for (int t = 0; t <= pos; ++t) {
          float dot = 0.0f;
          for (int e = 0; e < headDim; ++e)
            dot += q[h * headDim + e] * hKc[(static_cast<std::size_t>(l) * maxSeq + t) * kvDim + kvHead * headDim + e];
          sc[t] = dot * invSqrt;
        }
        float mx = sc[0];
        for (int t = 1; t <= pos; ++t) mx = std::max(mx, sc[t]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
        for (int e = 0; e < headDim; ++e) {
          float a = 0.0f;
          for (int t = 0; t <= pos; ++t)
            a += (sc[t] / sum) * hVc[(static_cast<std::size_t>(l) * maxSeq + t) * kvDim + kvHead * headDim + e];
          attn[h * headDim + e] = a;
        }
      }
      mm(tmp, w.wo, attn, d, qDim);
      for (int i = 0; i < d; ++i) x[i] += tmp[i];
      rms(xn, x, w.ffnNorm, d);
      mm(g, w.ffnGate, xn, ffn, d);
      mm(u, w.ffnUp, xn, ffn, d);
      for (int i = 0; i < ffn; ++i) act[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
      mm(tmp, w.ffnDown, act, d, ffn);
      for (int i = 0; i < d; ++i) x[i] += tmp[i];
    }
    rms(xn, x, outNorm, d);
    logits.resize(vocab);
    mm(logits, output, xn, vocab, d);
  };

  // ---- GPU forward driver (weights + KV cache resident in VRAM) ----
  auto up = [](const std::vector<float>& v) {
    float* p = nullptr;
    cudaMalloc(&p, v.size() * sizeof(float));
    cudaMemcpy(p, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    return p;
  };
  float* dTokEmbd = up(tokEmbd);
  float* dOutNorm = up(outNorm);
  float* dOutput = up(output);
  std::vector<float*> dAttnNorm(L), dWq(L), dWk(L), dWv(L), dWo(L), dFfnNorm(L), dGate(L), dUp(L), dDown(L);
  for (int l = 0; l < L; ++l) {
    dAttnNorm[l] = up(layers[l].attnNorm);
    dWq[l] = up(layers[l].wq);
    dWk[l] = up(layers[l].wk);
    dWv[l] = up(layers[l].wv);
    dWo[l] = up(layers[l].wo);
    dFfnNorm[l] = up(layers[l].ffnNorm);
    dGate[l] = up(layers[l].ffnGate);
    dUp[l] = up(layers[l].ffnUp);
    dDown[l] = up(layers[l].ffnDown);
  }
  float *dx, *dxn, *dq, *dk, *dv, *dattn, *dtmp, *dg, *du, *dact, *dlogits, *dKc, *dVc;
  cudaMalloc(&dx, d * sizeof(float));
  cudaMalloc(&dxn, d * sizeof(float));
  cudaMalloc(&dq, qDim * sizeof(float));
  cudaMalloc(&dk, kvDim * sizeof(float));
  cudaMalloc(&dv, kvDim * sizeof(float));
  cudaMalloc(&dattn, qDim * sizeof(float));
  cudaMalloc(&dtmp, d * sizeof(float));
  cudaMalloc(&dg, ffn * sizeof(float));
  cudaMalloc(&du, ffn * sizeof(float));
  cudaMalloc(&dact, ffn * sizeof(float));
  cudaMalloc(&dlogits, vocab * sizeof(float));
  cudaMalloc(&dKc, static_cast<std::size_t>(L) * maxSeq * kvDim * sizeof(float));
  cudaMalloc(&dVc, static_cast<std::size_t>(L) * maxSeq * kvDim * sizeof(float));

  auto grid = [](int n) { return (n + 255) / 256; };
  auto gpuForward = [&](int token, int pos, std::vector<float>& logits) {
    embedRowKernel<<<grid(d), 256>>>(dx, dTokEmbd, token, d);
    for (int l = 0; l < L; ++l) {
      rmsnormKernel<<<1, 256>>>(dxn, dx, dAttnNorm[l], d, eps);
      matmulF32Kernel<<<qDim, 256>>>(dq, dWq[l], dxn, qDim, d);
      matmulF32Kernel<<<kvDim, 256>>>(dk, dWk[l], dxn, kvDim, d);
      matmulF32Kernel<<<kvDim, 256>>>(dv, dWv[l], dxn, kvDim, d);
      ropeNeoxKernel<<<grid(nHeads * headDim / 2), 256>>>(dq, nHeads, headDim, pos, freqBase);
      ropeNeoxKernel<<<grid(nKv * headDim / 2), 256>>>(dk, nKv, headDim, pos, freqBase);
      kvStoreKernel<<<grid(kvDim), 256>>>(dKc, dk, l, pos, maxSeq, kvDim);
      kvStoreKernel<<<grid(kvDim), 256>>>(dVc, dv, l, pos, maxSeq, kvDim);
      float* Kl = dKc + static_cast<std::size_t>(l) * maxSeq * kvDim;
      float* Vl = dVc + static_cast<std::size_t>(l) * maxSeq * kvDim;
      const std::size_t shBytes = (static_cast<std::size_t>(pos + 1) + headDim) * sizeof(float);
      attentionDecodeKernel<<<nHeads, headDim, shBytes>>>(dattn, dq, Kl, Vl, nHeads, nKv, headDim, pos + 1, kvDim);
      matmulF32Kernel<<<d, 256>>>(dtmp, dWo[l], dattn, d, qDim);
      addKernel<<<grid(d), 256>>>(dx, dtmp, d);
      rmsnormKernel<<<1, 256>>>(dxn, dx, dFfnNorm[l], d, eps);
      matmulF32Kernel<<<ffn, 256>>>(dg, dGate[l], dxn, ffn, d);
      matmulF32Kernel<<<ffn, 256>>>(du, dUp[l], dxn, ffn, d);
      swigluKernel<<<grid(ffn), 256>>>(dact, dg, du, ffn);
      matmulF32Kernel<<<d, 256>>>(dtmp, dDown[l], dact, d, ffn);
      addKernel<<<grid(d), 256>>>(dx, dtmp, d);
    }
    rmsnormKernel<<<1, 256>>>(dxn, dx, dOutNorm, d, eps);
    matmulF32Kernel<<<vocab, 256>>>(dlogits, dOutput, dxn, vocab, d);
    logits.resize(vocab);
    cudaMemcpy(logits.data(), dlogits, vocab * sizeof(float), cudaMemcpyDeviceToHost);
  };

  const int tokens[3] = {1, 5, 9};
  float maxErr = 0.0f;
  for (int pos = 0; pos < 3; ++pos) {
    std::vector<float> cpuLogits, gpuLogits;
    cpuForward(tokens[pos], pos, cpuLogits);
    gpuForward(tokens[pos], pos, gpuLogits);
    for (int i = 0; i < vocab; ++i) maxErr = std::max(maxErr, std::fabs(cpuLogits[i] - gpuLogits[i]));
  }
  const cudaError_t e = cudaDeviceSynchronize();

  // Cleanup.
  for (float* p : {dTokEmbd, dOutNorm, dOutput, dx, dxn, dq, dk, dv, dattn, dtmp, dg, du, dact, dlogits, dKc, dVc})
    cudaFree(p);
  for (int l = 0; l < L; ++l)
    for (float* p : {dAttnNorm[l], dWq[l], dWk[l], dWv[l], dWo[l], dFfnNorm[l], dGate[l], dUp[l], dDown[l]})
      cudaFree(p);

  if (e != cudaSuccess) return {true, false, "GPU forward kernels: " + cudaErr(e)};
  const bool ok = maxErr < 2e-2f;
  return {true, ok,
          (ok ? "GPU forward pass (2-layer, 3 positions) matches CPU reference (max err "
              : "GPU forward disagrees with CPU (max err ") +
              std::to_string(maxErr) + ")"};
}

// ---- GPU-resident model runner -------------------------------------------------------------

namespace {

std::size_t weightBytes(std::uint32_t type, int rows, int cols) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  switch (type) {
    case 0: return n * sizeof(float);   // F32
    case 8: return n / 32 * kQ8Bytes;   // Q8_0
    case 12: return n / kQKK * kQ4KBytes;  // Q4_K
    case 14: return n / kQKK * kQ6KBytes;  // Q6_K
    default: return 0;
  }
}

struct DevWeight {
  void* d = nullptr;
  std::uint32_t type = 0;
  int rows = 0;
  int cols = 0;
};

class GpuModelImpl final : public GpuModel {
 public:
  explicit GpuModelImpl(const GpuModelConfig& cfg) : cfg_(cfg) {}

  ~GpuModelImpl() override {
    auto f = [](void* p) { if (p) cudaFree(p); };
    f(dTokEmbd_);
    f(dOutNorm_);
    f(output_.d);
    for (auto& L : layers_) {
      f(L.attnNorm);
      f(L.ffnNorm);
      for (DevWeight* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.ffnGate, &L.ffnUp, &L.ffnDown}) f(w->d);
    }
    f(dKc_);
    f(dVc_);
    for (void* p : {dx_, dxn_, dq_, dk_, dv_, dattn_, dtmp_, dg_, du_, dact_, dlogits_}) f(p);
  }

  bool init(const float* tokenEmbdF32, const float* outputNorm, const GpuWeight& output,
            const std::vector<GpuLayer>& layers, std::string& err) {
    const int d = cfg_.dModel, kvDim = cfg_.nKv * cfg_.headDim, ffn = cfg_.ffn, vocab = cfg_.vocab;
    auto upF32 = [](const float* h, std::size_t n) {
      float* p = nullptr;
      if (cudaMalloc(&p, n * sizeof(float)) != cudaSuccess) return static_cast<float*>(nullptr);
      cudaMemcpy(p, h, n * sizeof(float), cudaMemcpyHostToDevice);
      return p;
    };
    auto upW = [&](const GpuWeight& w, DevWeight& out) -> bool {
      const std::size_t bytes = weightBytes(w.ggmlType, w.rows, w.cols);
      if (bytes == 0) { err = "unsupported weight type " + std::to_string(w.ggmlType); return false; }
      if (cudaMalloc(&out.d, bytes) != cudaSuccess) { err = "cudaMalloc weight failed"; return false; }
      cudaMemcpy(out.d, w.host, bytes, cudaMemcpyHostToDevice);
      out.type = w.ggmlType;
      out.rows = w.rows;
      out.cols = w.cols;
      return true;
    };

    dTokEmbd_ = upF32(tokenEmbdF32, static_cast<std::size_t>(vocab) * d);
    dOutNorm_ = upF32(outputNorm, d);
    if (!dTokEmbd_ || !dOutNorm_) { err = "cudaMalloc embedding/norm failed"; return false; }
    if (!upW(output, output_)) return false;

    layers_.resize(cfg_.nLayers);
    for (int l = 0; l < cfg_.nLayers; ++l) {
      const GpuLayer& s = layers[l];
      DevLayer& t = layers_[l];
      t.attnNorm = upF32(s.attnNorm, d);
      t.ffnNorm = upF32(s.ffnNorm, d);
      if (!t.attnNorm || !t.ffnNorm) { err = "cudaMalloc layer norm failed"; return false; }
      if (!upW(s.wq, t.wq) || !upW(s.wk, t.wk) || !upW(s.wv, t.wv) || !upW(s.wo, t.wo) ||
          !upW(s.ffnGate, t.ffnGate) || !upW(s.ffnUp, t.ffnUp) || !upW(s.ffnDown, t.ffnDown))
        return false;
    }

    auto alloc = [](float*& p, std::size_t n) { return cudaMalloc(&p, n * sizeof(float)) == cudaSuccess; };
    const std::size_t kvAll = static_cast<std::size_t>(cfg_.nLayers) * cfg_.maxSeq * kvDim;
    if (!alloc(dx_, d) || !alloc(dxn_, d) || !alloc(dq_, cfg_.nHeads * cfg_.headDim) ||
        !alloc(dk_, kvDim) || !alloc(dv_, kvDim) || !alloc(dattn_, cfg_.nHeads * cfg_.headDim) ||
        !alloc(dtmp_, d) || !alloc(dg_, ffn) || !alloc(du_, ffn) || !alloc(dact_, ffn) ||
        !alloc(dlogits_, vocab) || !alloc(dKc_, kvAll) || !alloc(dVc_, kvAll)) {
      err = "cudaMalloc scratch/KV failed";
      return false;
    }
    hostLogits_.resize(vocab);
    return true;
  }

  void reset() override { /* KV slots are overwritten per position; nothing to clear */ }

  const std::vector<float>& forward(int token, int pos) override {
    const int d = cfg_.dModel, kvDim = cfg_.nKv * cfg_.headDim, ffn = cfg_.ffn;
    const int qDim = cfg_.nHeads * cfg_.headDim, hd = cfg_.headDim, mx = cfg_.maxSeq;
    auto g = [](int n) { return (n + 255) / 256; };

    embedRowKernel<<<g(d), 256>>>(dx_, dTokEmbd_, token, d);
    for (int l = 0; l < cfg_.nLayers; ++l) {
      DevLayer& w = layers_[l];
      rmsnormKernel<<<1, 256>>>(dxn_, dx_, w.attnNorm, d, cfg_.normEps);
      matmul(dq_, w.wq, dxn_);
      matmul(dk_, w.wk, dxn_);
      matmul(dv_, w.wv, dxn_);
      ropeNeoxKernel<<<g(cfg_.nHeads * hd / 2), 256>>>(dq_, cfg_.nHeads, hd, pos, cfg_.ropeFreqBase);
      ropeNeoxKernel<<<g(cfg_.nKv * hd / 2), 256>>>(dk_, cfg_.nKv, hd, pos, cfg_.ropeFreqBase);
      kvStoreKernel<<<g(kvDim), 256>>>(dKc_, dk_, l, pos, mx, kvDim);
      kvStoreKernel<<<g(kvDim), 256>>>(dVc_, dv_, l, pos, mx, kvDim);
      float* Kl = dKc_ + static_cast<std::size_t>(l) * mx * kvDim;
      float* Vl = dVc_ + static_cast<std::size_t>(l) * mx * kvDim;
      const std::size_t sh = (static_cast<std::size_t>(pos + 1) + hd) * sizeof(float);
      attentionDecodeKernel<<<cfg_.nHeads, hd, sh>>>(dattn_, dq_, Kl, Vl, cfg_.nHeads, cfg_.nKv, hd, pos + 1, kvDim);
      matmul(dtmp_, w.wo, dattn_);
      addKernel<<<g(d), 256>>>(dx_, dtmp_, d);
      rmsnormKernel<<<1, 256>>>(dxn_, dx_, w.ffnNorm, d, cfg_.normEps);
      matmul(dg_, w.ffnGate, dxn_);
      matmul(du_, w.ffnUp, dxn_);
      swigluKernel<<<g(ffn), 256>>>(dact_, dg_, du_, ffn);
      matmul(dtmp_, w.ffnDown, dact_);
      addKernel<<<g(d), 256>>>(dx_, dtmp_, d);
    }
    rmsnormKernel<<<1, 256>>>(dxn_, dx_, dOutNorm_, d, cfg_.normEps);
    matmul(dlogits_, output_, dxn_);
    cudaMemcpy(hostLogits_.data(), dlogits_, cfg_.vocab * sizeof(float), cudaMemcpyDeviceToHost);
    return hostLogits_;
  }

 private:
  struct DevLayer {
    float* attnNorm = nullptr;
    float* ffnNorm = nullptr;
    DevWeight wq, wk, wv, wo, ffnGate, ffnUp, ffnDown;
  };

  // Dispatches out = W * x to the kernel for W's quant type.
  void matmul(float* out, const DevWeight& w, const float* x) {
    const int grid = (w.rows + kWarpsPerBlock - 1) / kWarpsPerBlock, threads = kWarpsPerBlock * 32;
    switch (w.type) {
      case 12: qmatmulQ4_KKernel<<<grid, threads>>>(out, static_cast<const std::uint8_t*>(w.d), x, w.rows, w.cols); break;
      case 14: qmatmulQ6_KKernel<<<grid, threads>>>(out, static_cast<const std::uint8_t*>(w.d), x, w.rows, w.cols); break;
      case 8: qmatmulQ8_0Kernel<<<grid, threads>>>(out, static_cast<const std::uint8_t*>(w.d), x, w.rows, w.cols); break;
      default: matmulF32Kernel<<<w.rows, 256>>>(out, static_cast<const float*>(w.d), x, w.rows, w.cols); break;
    }
  }

  GpuModelConfig cfg_;
  float* dTokEmbd_ = nullptr;
  float* dOutNorm_ = nullptr;
  DevWeight output_;
  std::vector<DevLayer> layers_;
  float *dKc_ = nullptr, *dVc_ = nullptr;
  float *dx_ = nullptr, *dxn_ = nullptr, *dq_ = nullptr, *dk_ = nullptr, *dv_ = nullptr;
  float *dattn_ = nullptr, *dtmp_ = nullptr, *dg_ = nullptr, *du_ = nullptr, *dact_ = nullptr, *dlogits_ = nullptr;
  std::vector<float> hostLogits_;
};

}  // namespace

std::unique_ptr<GpuModel> createGpuModel(const GpuModelConfig& cfg, const float* tokenEmbdF32,
                                         const float* outputNorm, const GpuWeight& output,
                                         const std::vector<GpuLayer>& layers, std::string& error) {
  if (deviceCount() <= 0) {
    error = "no CUDA device present";
    return nullptr;
  }
  auto m = std::make_unique<GpuModelImpl>(cfg);
  if (!m->init(tokenEmbdF32, outputNorm, output, layers, error)) return nullptr;
  return m;
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
