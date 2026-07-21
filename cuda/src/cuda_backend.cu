// Real CUDA backend — compiled by nvcc only when QORVIX_ENABLE_CUDA is on and a toolkit is
// found (see cuda/CMakeLists.txt). The CPU stub (cuda_backend_stub.cpp) provides the same
// symbols otherwise; the two are never compiled together.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
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
