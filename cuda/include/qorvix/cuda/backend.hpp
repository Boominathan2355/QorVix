#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

// CUDA backend facade. The whole API is callable whether or not CUDA support was compiled in —
// in a CPU build it links a stub where builtWithCuda() is false and deviceCount() is 0, so
// callers (CLI, scheduler, tests) never need their own #ifdefs. The real implementation
// (cuda_backend.cu) is compiled only when QORVIX_ENABLE_CUDA is on and a CUDA toolkit is found.
namespace qorvix::cuda {

struct DeviceInfo {
  int index = 0;
  std::string name;
  std::size_t totalGlobalMem = 0;
  std::size_t freeMem = 0;
  int computeMajor = 0;
  int computeMinor = 0;
  int multiProcessorCount = 0;
};

// Result of a runtime self-test. `ran` is false when there is no device (or CUDA isn't built in),
// which callers treat as "skipped", not "failed".
struct SelfTestResult {
  bool ran = false;
  bool passed = false;
  std::string message;
};

// True iff this binary was compiled with CUDA support (a compile-time constant).
bool builtWithCuda() noexcept;

// Usable CUDA device count (0 if none, or if CUDA isn't built in). Never throws.
int deviceCount();

std::vector<DeviceInfo> enumerateDevices();
std::optional<DeviceInfo> deviceInfo(int index);
bool setDevice(int index);

// Host->device->host round-trip through a scale kernel; verifies the result on the host.
// The "hello tensor" bring-up test (SPEC Phase 4). ran=false when no device is present.
SelfTestResult selfTest();

// One GEMM via cuBLAS (C = A*B with A = identity, so C must equal B), verified on the host.
SelfTestResult gemmSelfTest();

// Native quantized matmul on the GPU: a block-per-row GEMV over Q8_0 weights (dequantized in
// registers, never copied to F32 — the GPU form of the CPU qmatmul). Checks correctness against a
// host reference on a small matrix, then times a large one; message reports GFLOP/s and GB/s.
SelfTestResult qmatmulSelfTest();

}  // namespace qorvix::cuda
