// CPU stub for the CUDA backend — compiled when QORVIX_ENABLE_CUDA is off or no toolkit is
// found. Every entry point reports "no CUDA", so the rest of the codebase compiles and runs
// unchanged on machines without a GPU toolchain. The real implementation lives in
// cuda_backend.cu; the two are never compiled together.
#include "qorvix/cuda/backend.hpp"
#include "qorvix/cuda/gpu_memory.hpp"
#include "qorvix/cuda/gpu_model.hpp"

namespace qorvix::cuda {

std::unique_ptr<GpuModel> createGpuModel(const GpuModelConfig&, const float*, const float*,
                                         const GpuWeight&, const std::vector<GpuLayer>&,
                                         std::string& error) {
  error = "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)";
  return nullptr;
}

bool builtWithCuda() noexcept { return false; }
int deviceCount() { return 0; }
std::vector<DeviceInfo> enumerateDevices() { return {}; }
std::optional<DeviceInfo> deviceInfo(int) { return std::nullopt; }
bool setDevice(int) { return false; }

SelfTestResult selfTest() {
  return {false, false, "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)"};
}
SelfTestResult gemmSelfTest() {
  return {false, false, "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)"};
}
SelfTestResult qmatmulSelfTest() {
  return {false, false, "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)"};
}
SelfTestResult qmatmulQ4_KSelfTest() {
  return {false, false, "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)"};
}
SelfTestResult qmatmulQ6_KSelfTest() {
  return {false, false, "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)"};
}
SelfTestResult opsSelfTest() {
  return {false, false, "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)"};
}
SelfTestResult attentionSelfTest() {
  return {false, false, "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)"};
}
SelfTestResult gpuForwardSelfTest() {
  return {false, false, "CUDA support not built in (rebuild with -DQORVIX_ENABLE_CUDA=ON)"};
}

std::unique_ptr<memory::ISlabAllocator> makeGpuSlabAllocator() { return nullptr; }
std::unique_ptr<memory::ITransferEngine> makeCudaTransferEngine() { return nullptr; }

std::unique_ptr<memory::MemoryManager> makeGpuMemoryManager(std::size_t, std::size_t, std::size_t,
                                                            const std::filesystem::path&) {
  return nullptr;
}

}  // namespace qorvix::cuda
