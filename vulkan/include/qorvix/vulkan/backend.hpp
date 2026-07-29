#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Vulkan compute backend facade. Mirrors the CUDA facade (qorvix/cuda/backend.hpp) so the CLI,
// scheduler and tests can drive whichever GPU backend is present without their own #ifdefs.
//
// WHY Vulkan (in addition to CUDA): one compute backend runs on every vendor — NVIDIA, AMD,
// Apple (via MoltenVK), and Intel — instead of NVIDIA-only. The kernels are GLSL compute shaders
// compiled to SPIR-V at build time and embedded in the binary.
//
// The whole API is callable whether or not Vulkan was compiled in: a build with
// QORVIX_ENABLE_VULKAN=OFF links a stub where builtWithVulkan() is false and deviceCount() is 0.
// The real implementation (vulkan_backend.cpp) is compiled only when the Vulkan headers/loader are
// found. Correctness is checked headlessly on Mesa's software device (lavapipe / llvmpipe), so the
// argmax-parity gate runs with no physical GPU.
namespace qorvix::vulkan {

struct DeviceInfo {
  int index = 0;
  std::string name;
  std::uint32_t apiVersion = 0;
  std::uint32_t vendorID = 0;
  std::uint32_t deviceID = 0;
  // VkPhysicalDeviceType as an int (0=other,1=integrated,2=discrete,3=virtual,4=cpu). Kept as a
  // plain int so this header never needs to include <vulkan/vulkan.h>.
  int deviceType = 0;
  std::size_t deviceLocalMem = 0;
};

// Result of a runtime self-test. `ran` is false when there is no device (or Vulkan isn't built in),
// which callers treat as "skipped", not "failed" — same contract as the CUDA facade.
struct SelfTestResult {
  bool ran = false;
  bool passed = false;
  std::string message;
};

// True iff this binary was compiled with the Vulkan backend (a compile-time constant).
bool builtWithVulkan() noexcept;

// Usable Vulkan physical-device count (0 if none, or if Vulkan isn't built in). Never throws.
int deviceCount();

std::vector<DeviceInfo> enumerateDevices();
std::optional<DeviceInfo> deviceInfo(int index);

// Host->device->host round-trip through a trivial compute dispatch, verified on the host: the
// "hello compute" bring-up test. ran=false when no device is present.
SelfTestResult selfTest();

// Native Q8_0 quantized matmul as a compute shader: a GEMV over Q8_0 weights dequantized in-shader
// (never expanded to F32 in memory), checked against a host reference on a small matrix and then
// timed on a large one. The Vulkan analogue of qorvix::cuda::qmatmulSelfTest().
SelfTestResult qmatmulQ8_0SelfTest();

// Native Q4_K / Q6_K matmul compute shaders (the weight types real GGUF models use), each checked
// against a host reference. The Vulkan analogue of qorvix::cuda::qmatmulQ4_K/Q6_KSelfTest().
SelfTestResult qmatmulQ4_KSelfTest();
SelfTestResult qmatmulQ6_KSelfTest();

// Forward-pass building-block compute shaders (RMSNorm, RoPE, SwiGLU, residual add) checked against
// a CPU reference. The Vulkan analogue of qorvix::cuda::opsSelfTest().
SelfTestResult opsSelfTest();

// Single-query GQA decode attention over a cached K/V, checked against a CPU reference. The Vulkan
// analogue of qorvix::cuda::attentionSelfTest().
SelfTestResult attentionSelfTest();

}  // namespace qorvix::vulkan
