#pragma once

#include <string>

// Generalized CPU backend support: detect the running CPU's ISA at RUNTIME and dispatch the hot
// kernels to the best available SIMD implementation — one portable binary that runs optimally on
// any CPU, exactly as the single Vulkan backend serves every GPU. This is the CPU analogue of
// backend auto-selection: scalar everywhere, AVX2/AVX-512 on x86, NEON/SVE on ARM/Apple Silicon,
// vector on RISC-V — chosen once at startup, no rebuild and no -march=native needed.
//
// Verification note: the x86 (AVX2) path is built and benchmarked in the reference environment;
// NEON/SVE (ARM/Apple), RISC-V vector, NUMA-aware placement, and thread affinity are architected
// here but need their own hardware to verify (like the HIP/Metal GPU backends).
namespace qorvix::runtime::cpu {

struct Features {
  std::string arch = "unknown";  // "x86_64" | "aarch64" | "riscv64" | ...
  // x86
  bool sse2 = false;
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool avx512f = false;
  // ARM / Apple Silicon
  bool neon = false;
  bool sve = false;
  unsigned hardwareThreads = 1;
};

// Detected once, on first call.
const Features& features();

// Runtime-dispatched dot product: out = sum_i a[i]*b[i], using the fastest kernel this CPU supports.
// This is the runtime's hot path (the quantized GEMV folds each dequantized block through it).
float dotProductF32(const float* a, const float* b, int n);

// Name of the kernel dotProductF32 actually dispatches to on this CPU ("avx2" | "neon" | "scalar").
const char* activeDotKernel();

// One-line human summary for `qorvix cpuinfo`.
std::string summary();

}  // namespace qorvix::runtime::cpu
