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
