// CPU stub for the Vulkan backend: linked when QORVIX_ENABLE_VULKAN is OFF or the Vulkan loader/
// headers were not found. Every facade entry point stays callable and simply reports "no Vulkan",
// so callers (CLI, tests) need no #ifdefs. Mirrors cuda/src/cuda_backend_stub.cpp.
#include "qorvix/vulkan/backend.hpp"

namespace qorvix::vulkan {

bool builtWithVulkan() noexcept { return false; }

int deviceCount() { return 0; }

std::vector<DeviceInfo> enumerateDevices() { return {}; }

std::optional<DeviceInfo> deviceInfo(int) { return std::nullopt; }

SelfTestResult selfTest() { return {false, false, "Vulkan support not compiled in"}; }

SelfTestResult qmatmulQ8_0SelfTest() { return {false, false, "Vulkan support not compiled in"}; }

SelfTestResult qmatmulQ4_KSelfTest() { return {false, false, "Vulkan support not compiled in"}; }

SelfTestResult qmatmulQ6_KSelfTest() { return {false, false, "Vulkan support not compiled in"}; }

SelfTestResult opsSelfTest() { return {false, false, "Vulkan support not compiled in"}; }

SelfTestResult attentionSelfTest() { return {false, false, "Vulkan support not compiled in"}; }

}  // namespace qorvix::vulkan
