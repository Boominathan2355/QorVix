#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// GPU-resident model runner on Vulkan: uploads a model's weights to device buffers and runs the
// transformer forward pass with the compute shaders (vulkan_backend.cpp), dispatching each matmul
// to the shader for that weight's quant type. Mirrors qorvix::cuda::GpuModel / gpu_model.hpp so the
// core bridge (buildGpuModel) has a near-identical Vulkan twin. Built from plain descriptors so the
// vulkan module stays independent of gguf/runtime types. Returns null when Vulkan isn't built in or
// no device is present.
namespace qorvix::vulkan {

struct VulkanModelConfig {
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

// A matmul weight [rows, cols] as raw host bytes in its GGUF quant type (ggmlType: F32=0, Q8_0=8,
// Q4_K=12, Q6_K=14). Uploaded verbatim to a device buffer; byte size is derived from the type.
struct VulkanWeight {
  const void* host = nullptr;
  std::uint32_t ggmlType = 0;
  int rows = 0;
  int cols = 0;
};

struct VulkanLayer {
  const float* attnNorm = nullptr;  // [dModel], F32
  const float* ffnNorm = nullptr;   // [dModel], F32
  VulkanWeight wq, wk, wv, wo, ffnGate, ffnUp, ffnDown;
};

class VulkanModel {
 public:
  virtual ~VulkanModel() = default;

  // Runs the transformer for `token` at `pos`, updating the KV cache. The returned reference is
  // valid only until the next call (a single reused host buffer) — same contract as GpuModel.
  virtual const std::vector<float>& forward(int token, int pos) = 0;
  virtual void reset() = 0;  // clear the KV cache
};

// Builds a VulkanModel. `tokenEmbdF32` is the embedding table already dequantized to F32
// ([vocab*dModel], host) so the on-device lookup is a copy. `output` is the (quantized) LM-head
// weight. Returns nullptr with `error` set on failure (no device, alloc/pipeline failure).
std::unique_ptr<VulkanModel> createVulkanModel(const VulkanModelConfig& cfg, const float* tokenEmbdF32,
                                               const float* outputNorm, const VulkanWeight& output,
                                               const std::vector<VulkanLayer>& layers,
                                               std::string& error);

}  // namespace qorvix::vulkan
