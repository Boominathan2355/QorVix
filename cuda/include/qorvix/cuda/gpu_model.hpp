#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// GPU-resident model runner: uploads a model's weights to VRAM and runs the transformer forward
// pass on the device (kernels from cuda_backend.cu), dispatching each matmul to the kernel for
// that weight's quant type. Built from plain descriptors so the cuda module stays independent of
// gguf/runtime types — the caller (core) bridges from a loaded GGUF. Returns null in CPU-only
// builds (no CUDA / no device).
namespace qorvix::cuda {

struct GpuModelConfig {
  int nLayers = 0;
  int dModel = 0;
  int nHeads = 0;
  int headDim = 0;
  int nKv = 0;
  int ffn = 0;
  int vocab = 0;
  int maxSeq = 0;
  float normEps = 1e-5f;
  float ropeFreqBase = 10000.0f;
};

// A matmul weight [rows, cols] as raw host bytes in its GGUF quant type (ggmlType: F32=0,
// Q8_0=8, Q4_K=12, Q6_K=14). Uploaded verbatim to VRAM; the byte size is derived from the type.
struct GpuWeight {
  const void* host = nullptr;
  std::uint32_t ggmlType = 0;
  int rows = 0;
  int cols = 0;
};

struct GpuLayer {
  const float* attnNorm = nullptr;  // [dModel], F32
  const float* ffnNorm = nullptr;   // [dModel], F32
  GpuWeight wq, wk, wv, wo, ffnGate, ffnUp, ffnDown;
};

class GpuModel {
 public:
  virtual ~GpuModel() = default;
  // Runs the transformer for `token` at position `pos` (updating the on-device KV cache) and
  // returns logits ([vocab]) copied to host.
  virtual const std::vector<float>& forward(int token, int pos) = 0;
  virtual void reset() = 0;  // clear the KV cache
};

// Builds a GpuModel. `tokenEmbdF32` is the embedding table already dequantized to F32
// ([vocab*dModel], host) so the on-device embedding lookup is a copy. `output` is the (quantized)
// LM-head weight. Returns nullptr with `error` set on failure (no device, alloc failure).
std::unique_ptr<GpuModel> createGpuModel(const GpuModelConfig& cfg, const float* tokenEmbdF32,
                                         const float* outputNorm, const GpuWeight& output,
                                         const std::vector<GpuLayer>& layers, std::string& error);

}  // namespace qorvix::cuda
