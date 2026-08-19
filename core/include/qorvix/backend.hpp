#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/cuda/backend.hpp"
#include "qorvix/cuda/gpu_model.hpp"
#include "qorvix/cuda/multi_gpu.hpp"
#include "qorvix/embeddings/bert_model.hpp"
#include "qorvix/embeddings/embedding_engine.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/gpu_engine.hpp"
#include "qorvix/runtime/dequant.hpp"
#include "qorvix/runtime/inference_engine.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/runtime/weights.hpp"
#include "qorvix/vulkan/backend.hpp"
#include "qorvix/vulkan/vulkan_model.hpp"
#include "qorvix/vulkan_engine.hpp"

// Unified backend layer. Every compute backend (CPU / CUDA / Vulkan) is one implementation of the
// runtime::IInferenceEngine seam, and createEngine() is the single place that builds one from a
// GGUF file. `generate`, `serve`, and the *-check commands all go through this — there is no
// per-backend branching or parallel generation loop anywhere above this header.
//
// Phase 11a adds a SECOND seam alongside it: embeddings::IEmbeddingEngine, built by
// createEmbeddingEngine(). One seam per task, one factory per seam — the invariant being kept is
// "no parallel path", not "one interface for everything" (see embedding_engine.hpp for why an
// encoder cannot implement the generation seam).
namespace qorvix {

enum class Backend { Cpu, Cuda, Vulkan };

inline const char* backendName(Backend b) {
  switch (b) {
    case Backend::Cuda: return "cuda";
    case Backend::Vulkan: return "vulkan";
    default: return "cpu";
  }
}

// A backend is usable iff it is compiled in AND has a device (CPU is always usable).
inline bool backendAvailable(Backend b) {
  switch (b) {
    case Backend::Cuda: return cuda::builtWithCuda() && cuda::deviceCount() > 0;
    case Backend::Vulkan: return vulkan::builtWithVulkan() && vulkan::deviceCount() > 0;
    default: return true;
  }
}

// Best available backend, fastest first: CUDA > Vulkan > CPU.
inline Backend selectBestBackend() {
  if (backendAvailable(Backend::Cuda)) return Backend::Cuda;
  if (backendAvailable(Backend::Vulkan)) return Backend::Vulkan;
  return Backend::Cpu;
}

// Parse a backend name; "auto" resolves to the best available. nullopt on an unknown name.
inline std::optional<Backend> parseBackend(std::string_view s) {
  if (s == "cpu") return Backend::Cpu;
  if (s == "cuda" || s == "gpu") return Backend::Cuda;
  if (s == "vulkan") return Backend::Vulkan;
  if (s == "auto") return selectBestBackend();
  return std::nullopt;
}

namespace detail {

// GGUF weight -> backend weight descriptor (same raw bytes, borrowed from the mmap).
inline cuda::GpuWeight toGpuWeight(const runtime::WeightMat& m) {
  return cuda::GpuWeight{m.quant, m.type, m.rows, m.cols};
}
inline vulkan::VulkanWeight toVkWeight(const runtime::WeightMat& m) {
  return vulkan::VulkanWeight{m.quant, m.type, m.rows, m.cols};
}

// The one bridge from a loaded GGUF (runtime Weights) to a backend model config + layer list.
// Templated over the backend's descriptor types so CUDA and Vulkan share the exact same wiring —
// the only per-backend difference is the concrete Config/Weight/Layer structs.
template <typename Config, typename Weight, typename Layer, typename ToWeight>
Config makeConfig(const runtime::ModelConfig& cfg, int maxSeq) {
  Config gc;
  gc.nLayers = static_cast<int>(cfg.blockCount);
  gc.dModel = static_cast<int>(cfg.embeddingLength);
  gc.nHeads = static_cast<int>(cfg.headCount);
  gc.headDim = static_cast<int>(cfg.headDim());
  gc.nKv = static_cast<int>(cfg.headCountKv);
  gc.ffn = static_cast<int>(cfg.feedForwardLength);
  gc.vocab = static_cast<int>(cfg.vocabSize);
  gc.maxSeq = maxSeq;
  gc.normEps = cfg.normEpsilon;
  gc.ropeFreqBase = cfg.ropeFreqBase;
  return gc;
}

}  // namespace detail

// Uploads a loaded GGUF model's weights to VRAM and returns a CUDA model runner (null on failure).
// Dequantizes the embedding table to F32 for the on-device lookup; layer weights stay quantized.
inline std::unique_ptr<cuda::GpuModel> buildGpuModel(const runtime::ModelConfig& cfg,
                                                     const runtime::Weights& w, int maxSeq,
                                                     std::string& err, int maxSessions = 1) {
  namespace rt = qorvix::runtime;
  const int d = static_cast<int>(cfg.embeddingLength), vocab = static_cast<int>(cfg.vocabSize);
  std::vector<float> embF32(static_cast<std::size_t>(vocab) * d);
  if (!rt::dequantize(w.tokenEmbd.type, w.tokenEmbd.quant, embF32.data(),
                      static_cast<std::size_t>(vocab) * d)) {
    err = "failed to dequantize token_embd";
    return nullptr;
  }
  auto gc = detail::makeConfig<cuda::GpuModelConfig, cuda::GpuWeight, cuda::GpuLayer, int>(cfg, maxSeq);
  cuda::GpuWeight output =
      w.output.valid() ? detail::toGpuWeight(w.output) : detail::toGpuWeight(w.tokenEmbd);
  std::vector<cuda::GpuLayer> gl(cfg.blockCount);
  for (std::uint32_t l = 0; l < cfg.blockCount; ++l) {
    const auto& L = w.layers[l];
    gl[l] = {L.attnNorm.data(), L.ffnNorm.data(), detail::toGpuWeight(L.wq),
             detail::toGpuWeight(L.wk), detail::toGpuWeight(L.wv), detail::toGpuWeight(L.wo),
             detail::toGpuWeight(L.ffnGate), detail::toGpuWeight(L.ffnUp),
             detail::toGpuWeight(L.ffnDown)};
  }
  return cuda::createGpuModel(gc, embF32.data(), w.outputNorm.data(), output, gl, err, maxSessions);
}

// The tensor-parallel twin of buildGpuModel. Same GGUF bridge, same descriptors, same maxSeq — the
// only difference is that the cuda module slices the weights across `devices` and returns a model
// whose VRAM footprint per device is roughly 1/world of the whole. Deliberately NOT a separate
// engine type: the result is a cuda::GpuModel, so GpuEngine and everything above it are untouched.
inline std::unique_ptr<cuda::GpuModel> buildShardedGpuModel(const runtime::ModelConfig& cfg,
                                                            const runtime::Weights& w, int maxSeq,
                                                            const std::vector<int>& devices,
                                                            std::string& err,
                                                            int maxSessions = 1) {
  namespace rt = qorvix::runtime;
  const int d = static_cast<int>(cfg.embeddingLength), vocab = static_cast<int>(cfg.vocabSize);
  std::vector<float> embF32(static_cast<std::size_t>(vocab) * d);
  if (!rt::dequantize(w.tokenEmbd.type, w.tokenEmbd.quant, embF32.data(),
                      static_cast<std::size_t>(vocab) * d)) {
    err = "failed to dequantize token_embd";
    return nullptr;
  }
  auto gc = detail::makeConfig<cuda::GpuModelConfig, cuda::GpuWeight, cuda::GpuLayer, int>(cfg, maxSeq);
  cuda::GpuWeight output =
      w.output.valid() ? detail::toGpuWeight(w.output) : detail::toGpuWeight(w.tokenEmbd);
  std::vector<cuda::GpuLayer> gl(cfg.blockCount);
  for (std::uint32_t l = 0; l < cfg.blockCount; ++l) {
    const auto& L = w.layers[l];
    gl[l] = {L.attnNorm.data(), L.ffnNorm.data(), detail::toGpuWeight(L.wq),
             detail::toGpuWeight(L.wk), detail::toGpuWeight(L.wv), detail::toGpuWeight(L.wo),
             detail::toGpuWeight(L.ffnGate), detail::toGpuWeight(L.ffnUp),
             detail::toGpuWeight(L.ffnDown)};
  }
  return cuda::createShardedGpuModel(gc, embF32.data(), w.outputNorm.data(), output, gl, devices,
                                     err, maxSessions);
}

// The Vulkan twin of buildGpuModel — same bridge, cross-vendor backend.
inline std::unique_ptr<vulkan::VulkanModel> buildVulkanModel(const runtime::ModelConfig& cfg,
                                                             const runtime::Weights& w, int maxSeq,
                                                             std::string& err, int maxSessions = 1) {
  namespace rt = qorvix::runtime;
  const int d = static_cast<int>(cfg.embeddingLength), vocab = static_cast<int>(cfg.vocabSize);
  std::vector<float> embF32(static_cast<std::size_t>(vocab) * d);
  if (!rt::dequantize(w.tokenEmbd.type, w.tokenEmbd.quant, embF32.data(),
                      static_cast<std::size_t>(vocab) * d)) {
    err = "failed to dequantize token_embd";
    return nullptr;
  }
  auto gc =
      detail::makeConfig<vulkan::VulkanModelConfig, vulkan::VulkanWeight, vulkan::VulkanLayer, int>(
          cfg, maxSeq);
  vulkan::VulkanWeight output =
      w.output.valid() ? detail::toVkWeight(w.output) : detail::toVkWeight(w.tokenEmbd);
  std::vector<vulkan::VulkanLayer> gl(cfg.blockCount);
  for (std::uint32_t l = 0; l < cfg.blockCount; ++l) {
    const auto& L = w.layers[l];
    gl[l] = {L.attnNorm.data(), L.ffnNorm.data(), detail::toVkWeight(L.wq),
             detail::toVkWeight(L.wk), detail::toVkWeight(L.wv), detail::toVkWeight(L.wo),
             detail::toVkWeight(L.ffnGate), detail::toVkWeight(L.ffnUp),
             detail::toVkWeight(L.ffnDown)};
  }
  return vulkan::createVulkanModel(gc, embF32.data(), w.outputNorm.data(), output, gl, err,
                                   maxSessions);
}

// THE unified construction point: builds an IInferenceEngine for `backend` from an opened GGUF file
// (moved in). For CPU the returned engine owns the file so its borrowed quantized weights stay
// mapped; for CUDA/Vulkan the weights are uploaded to the device and the file is released here.
// Returns nullptr with `err` set on failure (unavailable backend, bad model, alloc failure).
inline std::unique_ptr<runtime::IInferenceEngine> createEngine(Backend backend, gguf::GgufFile file,
                                                               std::uint32_t maxSeq,
                                                               std::uint32_t maxSessions,
                                                               std::string& err,
                                                               const std::vector<int>& tpDevices = {}) {
  namespace rt = qorvix::runtime;
  if (!backendAvailable(backend)) {
    err = std::string(backendName(backend)) + " backend unavailable (not built in, or no device)";
    return nullptr;
  }

  if (backend == Backend::Cpu) {
    auto m = rt::TextModel::fromGguf(std::move(file), err, maxSeq, maxSessions);
    if (!m) return nullptr;
    return std::make_unique<rt::TextModel>(std::move(*m));
  }

  // Device backends share the weight-loading front half; the file's mmap must outlive the upload.
  const auto cfg = rt::configFromGguf(file, err);
  if (!cfg.valid()) {
    if (err.empty()) err = "invalid model config";
    return nullptr;
  }
  // The CPU path is guarded inside TextModel::fromGguf; guard the device paths here, before
  // loadWeights goes looking for tensors an encoder does not have.
  if (cfg.isEncoder()) {
    err = "'" + cfg.architecture +
          "' is an encoder model with no LM head — use `qorvix embed`, not the generation path";
    return nullptr;
  }
  auto weights = rt::loadWeights(file, cfg, err);
  if (!weights) return nullptr;

  if (backend == Backend::Cuda) {
    // Tensor parallelism is a property of the MODEL, not of a different engine: when a device list
    // is given the weights are sharded across it and the same GpuEngine drives the result, so no
    // caller above this line learns that the world size changed.
    if (tpDevices.size() > 1) {
      auto gm = buildShardedGpuModel(cfg, *weights, static_cast<int>(maxSeq), tpDevices, err,
                                     static_cast<int>(maxSessions));
      if (!gm) return nullptr;
      return std::make_unique<GpuEngine>(std::move(gm), cfg, maxSeq,
                                         "cuda-tp" + std::to_string(tpDevices.size()));
    }
    auto gm = buildGpuModel(cfg, *weights, static_cast<int>(maxSeq), err,
                            static_cast<int>(maxSessions));
    if (!gm) return nullptr;
    return std::make_unique<GpuEngine>(std::move(gm), cfg, maxSeq);
  }
  auto vm = buildVulkanModel(cfg, *weights, static_cast<int>(maxSeq), err,
                             static_cast<int>(maxSessions));
  if (!vm) return nullptr;
  return std::make_unique<VulkanEngine>(std::move(vm), cfg, maxSeq);
}

// THE construction point for encoders — the embedding twin of createEngine(). Two seams, two
// factories: an encoder's unit of work is `tokens[N] -> vector[d]`, which IInferenceEngine cannot
// express (see embedding_engine.hpp). There is still no parallel path — `embed`, `embed-check`,
// `serve --embed-model` and the RAG commands all come through here.
//
// Phase 11a implements Backend::Cpu only. The device backends report honestly rather than
// silently falling back to CPU, because a silent fallback would make `--gpu` a lie in the logs.
inline std::unique_ptr<embeddings::IEmbeddingEngine> createEmbeddingEngine(
    Backend backend, gguf::GgufFile file, std::uint32_t maxSeq, std::string& err) {
  if (!backendAvailable(backend)) {
    err = std::string(backendName(backend)) + " backend unavailable (not built in, or no device)";
    return nullptr;
  }
  if (backend != Backend::Cpu) {
    err = std::string(backendName(backend)) +
          " embedding backend is not implemented yet — use the CPU path";
    return nullptr;
  }
  auto m = embeddings::BertModel::fromGguf(std::move(file), err, maxSeq);
  if (!m) return nullptr;
  return std::make_unique<embeddings::BertModel>(std::move(*m));
}

}  // namespace qorvix
