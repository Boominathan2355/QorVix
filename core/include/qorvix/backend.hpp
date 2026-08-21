#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/cuda/backend.hpp"
#include "qorvix/cuda/clip_model.hpp"
#include "qorvix/cuda/embedding_model.hpp"
#include "qorvix/cuda/gpu_model.hpp"
#include "qorvix/cuda/multi_gpu.hpp"
#include "qorvix/embeddings/bert_model.hpp"
#include "qorvix/embeddings/embedding_engine.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/gpu_engine.hpp"
#include "qorvix/runtime/dequant.hpp"
#include "qorvix/runtime/encoder_weights.hpp"
#include "qorvix/runtime/inference_engine.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/runtime/weights.hpp"
#include "qorvix/vulkan/backend.hpp"
#include "qorvix/vulkan/clip_model.hpp"
#include "qorvix/vulkan/embedding_model.hpp"
#include "qorvix/vulkan/vulkan_model.hpp"
#include "qorvix/vulkan_engine.hpp"
#include "qorvix/vision/clip_model.hpp"
#include "qorvix/vision/clip_weights.hpp"

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
inline cuda::EmbedWeight toEmbedWeight(const runtime::WeightMat& m) {
  return cuda::EmbedWeight{m.quant, m.type, m.rows, m.cols};
}
inline cuda::GpuClipWeight toGpuClipWeight(const runtime::WeightMat& m) {
  return cuda::GpuClipWeight{m.quant, m.type, m.rows, m.cols};
}
inline vulkan::VulkanWeight toVkWeight(const runtime::WeightMat& m) {
  return vulkan::VulkanWeight{m.quant, m.type, m.rows, m.cols};
}
inline vulkan::VulkanClipWeight toVkClipWeight(const runtime::WeightMat& m) {
  return vulkan::VulkanClipWeight{m.quant, m.type, m.rows, m.cols};
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
// Phase 11c adds CUDA and Vulkan GPU backends alongside CPU. GPU backends report honestly rather
// than silently falling back to CPU, because a silent fallback would make `--gpu` a lie in the logs.
inline std::unique_ptr<embeddings::IEmbeddingEngine> createEmbeddingEngine(
    Backend backend, gguf::GgufFile file, std::uint32_t maxSeq, std::string& err) {
  if (!backendAvailable(backend)) {
    err = std::string(backendName(backend)) + " backend unavailable (not built in, or no device)";
    return nullptr;
  }

  if (backend == Backend::Cpu) {
    auto m = embeddings::BertModel::fromGguf(std::move(file), err, maxSeq);
    if (!m) return nullptr;
    return std::make_unique<embeddings::BertModel>(std::move(*m));
  }

  // GPU backends: load the GGUF, extract weights, build the GPU embedding model.
  auto cfg = runtime::configFromGguf(file, err);
  if (!cfg.valid()) { if (err.empty()) err = "invalid model config"; return nullptr; }
  if (!cfg.isEncoder()) {
    err = "'" + cfg.architecture + "' is a decoder model — use the generation path";
    return nullptr;
  }
  auto weights = runtime::loadEncoderWeights(file, cfg, err);
  if (!weights) return nullptr;

  const std::uint32_t cap = cfg.contextLength ? cfg.contextLength : 512;
  const std::uint32_t seq = (maxSeq == 0 || maxSeq > cap) ? cap : maxSeq;

  if (backend == Backend::Cuda) {
    cuda::EmbeddingConfig ec;
    ec.nLayers = static_cast<int>(cfg.blockCount);
    ec.dModel = static_cast<int>(cfg.embeddingLength);
    ec.nHeads = static_cast<int>(cfg.headCount);
    ec.headDim = static_cast<int>(cfg.headDim());
    ec.ffn = static_cast<int>(cfg.feedForwardLength);
    ec.vocab = static_cast<int>(cfg.vocabSize);
    ec.maxSeq = static_cast<int>(seq);
    ec.normEps = cfg.normEpsilon;
    ec.ffnGated = cfg.ffnGated;
    ec.hasTokenTypes = cfg.tokenTypeCount > 0;
    ec.tokenTypeCount = static_cast<int>(cfg.tokenTypeCount);
    ec.hasPositionEmbd = cfg.hasPositionEmbd;
    ec.hasEmbdNorm = !weights->embdNorm.empty();
    ec.defaultPooling = static_cast<int>(cfg.pooling);
    ec.defaultNormalize = true;

    // Dequantize embedding tables to F32 (same as buildGpuModel does for the decoder path).
    const std::size_t vocabD = static_cast<std::size_t>(cfg.vocabSize) * cfg.embeddingLength;
    std::vector<float> embF32(vocabD);
    if (!runtime::dequantize(weights->tokenEmbd.type, weights->tokenEmbd.quant,
                             embF32.data(), vocabD)) {
      err = "failed to dequantize token_embd for embedding engine";
      return nullptr;
    }

    cuda::GpuEmbeddingTables tables{};
    // Token embedding: dequantized F32 is in embF32; upload happens in createEmbeddingModel
    // via the tables struct. The F32 data pointer is borrowed from embF32 which stays alive
    // through the createEmbeddingModel call (it uploads to VRAM synchronously).
    tables.tokenEmbd = embF32.data();

    std::vector<float> posF32;
    if (cfg.hasPositionEmbd) {
      const std::size_t posD = static_cast<std::size_t>(cfg.contextLength) * cfg.embeddingLength;
      posF32.resize(posD);
      if (!runtime::dequantize(weights->positionEmbd.type, weights->positionEmbd.quant,
                               posF32.data(), posD)) {
        err = "failed to dequantize position_embd";
        return nullptr;
      }
      tables.positionEmbd = posF32.data();
    }

    std::vector<float> typesF32;
    if (cfg.tokenTypeCount > 0 && weights->tokenTypes.valid()) {
      const std::size_t typesD = static_cast<std::size_t>(cfg.tokenTypeCount) * cfg.embeddingLength;
      typesF32.resize(typesD);
      if (!runtime::dequantize(weights->tokenTypes.type, weights->tokenTypes.quant,
                               typesF32.data(), typesD)) {
        err = "failed to dequantize token_types";
        return nullptr;
      }
      tables.tokenTypes = typesF32.data();
    }

    if (!weights->embdNorm.empty()) tables.embdNorm = weights->embdNorm.data();
    if (!weights->embdNormB.empty()) tables.embdNormB = weights->embdNormB.data();

    std::vector<cuda::GpuEmbeddingLayer> gl(cfg.blockCount);
    for (std::uint32_t l = 0; l < cfg.blockCount; ++l) {
      const auto& L = weights->layers[l];
      auto& g = gl[l];
      g.attnNorm = L.attnNorm.data();
      g.attnNormB = L.attnNormB.empty() ? nullptr : L.attnNormB.data();
      g.ffnNorm = L.ffnNorm.data();
      g.ffnNormB = L.ffnNormB.empty() ? nullptr : L.ffnNormB.data();
      g.wq = detail::toEmbedWeight(L.wq);
      g.wk = detail::toEmbedWeight(L.wk);
      g.wv = detail::toEmbedWeight(L.wv);
      g.wo = detail::toEmbedWeight(L.wo);
      g.bq = L.bq.empty() ? nullptr : L.bq.data();
      g.bk = L.bk.empty() ? nullptr : L.bk.data();
      g.bv = L.bv.empty() ? nullptr : L.bv.data();
      g.bo = L.bo.empty() ? nullptr : L.bo.data();
      g.ffnUp = detail::toEmbedWeight(L.ffnUp);
      g.ffnDown = detail::toEmbedWeight(L.ffnDown);
      g.ffnUpB = L.ffnUpB.empty() ? nullptr : L.ffnUpB.data();
      g.ffnDownB = L.ffnDownB.empty() ? nullptr : L.ffnDownB.data();
      if (L.ffnGate.valid()) g.ffnGate = detail::toEmbedWeight(L.ffnGate);
    }

    auto m = cuda::createEmbeddingModel(ec, tables, gl, err);
    if (!m) return nullptr;
    // Wrap the cuda::EmbeddingModel in an IEmbeddingEngine adapter
    struct CudaEmbedAdapter : public embeddings::IEmbeddingEngine {
      std::unique_ptr<cuda::EmbeddingModel> impl;
      runtime::ModelConfig cfg;
      explicit CudaEmbedAdapter(std::unique_ptr<cuda::EmbeddingModel> m, runtime::ModelConfig c)
          : impl(std::move(m)), cfg(std::move(c)) {}
      bool embed(const std::vector<int>& t, std::vector<float>& o, std::string& e) override { return impl->embed(t, o, e); }
      bool embedWith(const std::vector<int>& t, embeddings::PoolingType p, bool n, std::vector<float>& o, std::string& e) override {
        return impl->embedWith(t, static_cast<int>(p), n, o, e);
      }
      bool embedTokens(const std::vector<int>& t, std::vector<float>& o, std::string& e) override { return impl->embedTokens(t, o, e); }
      bool embedBatch(const std::vector<std::vector<int>>& b, std::vector<std::vector<float>>& o, std::string& e) override {
        return impl->embedBatch(b, o, e);
      }
      std::uint32_t dim() const override { return static_cast<std::uint32_t>(impl->dim()); }
      std::uint32_t maxSeqLen() const override { return static_cast<std::uint32_t>(impl->maxSeqLen()); }
      embeddings::PoolingType defaultPooling() const override { return cfg.pooling; }
      bool defaultNormalize() const override { return impl->defaultNormalize(); }
      const runtime::ModelConfig& config() const override { return cfg; }
      std::string backendName() const override { return impl->backendName(); }
    };
    return std::make_unique<CudaEmbedAdapter>(std::move(m), std::move(cfg));
  }

  // Vulkan: stub for now (returns nullptr with error)
  err = "vulkan embedding backend not yet implemented";
  return nullptr;
}

// THE construction point for CLIP vision towers — the vision twin of createEngine() and
// createEmbeddingEngine(). Supports CPU and CUDA (and reports honestly for Vulkan).
inline std::unique_ptr<vision::ClipVisionModel> createClipVisionModel(
    Backend backend, gguf::GgufFile file, std::string& err) {
  if (!backendAvailable(backend)) {
    err = std::string(backendName(backend)) + " backend unavailable (not built in, or no device)";
    return nullptr;
  }

  if (backend == Backend::Cpu) {
    auto m = vision::ClipVisionModel::fromGguf(std::move(file), err);
    if (!m) return nullptr;
    return std::make_unique<vision::ClipVisionModel>(std::move(*m));
  }

  auto cfg = vision::clipConfigFromGguf(file, err);
  if (!cfg.valid()) {
    if (err.empty()) err = "invalid clip config";
    return nullptr;
  }
  auto weights = vision::loadClipWeights(file, cfg, err);
  if (!weights) return nullptr;

  if (backend == Backend::Cuda) {
    cuda::ClipConfig cc;
    cc.imageSize = static_cast<int>(cfg.imageSize);
    cc.patchSize = static_cast<int>(cfg.patchSize);
    cc.embeddingLength = static_cast<int>(cfg.embeddingLength);
    cc.feedForwardLength = static_cast<int>(cfg.feedForwardLength);
    cc.headCount = static_cast<int>(cfg.headCount);
    cc.blockCount = static_cast<int>(cfg.blockCount);
    cc.projectionDim = static_cast<int>(cfg.projectionDim);
    cc.normEpsilon = cfg.normEpsilon;
    cc.useGelu = cfg.useGelu;
    cc.hasProjector = cfg.hasLlavaProjector;
    cc.projectedDim = weights->projectedDim();
    cc.imageMean[0] = cfg.imageMean[0];
    cc.imageMean[1] = cfg.imageMean[1];
    cc.imageMean[2] = cfg.imageMean[2];
    cc.imageStd[0] = cfg.imageStd[0];
    cc.imageStd[1] = cfg.imageStd[1];
    cc.imageStd[2] = cfg.imageStd[2];

    cuda::GpuClipWeights gw{};
    gw.patchEmbd = detail::toGpuClipWeight(weights->patchEmbd);
    gw.classEmbd = weights->classEmbd.data();

    // Dequantize positionEmbd to F32
    const std::size_t posElems = static_cast<std::size_t>(cfg.tokenCount()) * cfg.embeddingLength;
    std::vector<float> posF32(posElems);
    if (!runtime::dequantize(weights->positionEmbd.type, weights->positionEmbd.quant,
                             posF32.data(), posElems)) {
      err = "failed to dequantize clip position_embd";
      return nullptr;
    }
    gw.positionEmbd = posF32.data();

    gw.preLnW = weights->preLnW.empty() ? nullptr : weights->preLnW.data();
    gw.preLnB = weights->preLnB.empty() ? nullptr : weights->preLnB.data();
    gw.postLnW = weights->postLnW.empty() ? nullptr : weights->postLnW.data();
    gw.postLnB = weights->postLnB.empty() ? nullptr : weights->postLnB.data();

    if (weights->hasProjector()) {
      gw.mm0 = detail::toGpuClipWeight(weights->mm0);
      gw.mm0B = weights->mm0B.empty() ? nullptr : weights->mm0B.data();
      gw.mm2 = detail::toGpuClipWeight(weights->mm2);
      gw.mm2B = weights->mm2B.empty() ? nullptr : weights->mm2B.data();
    }

    std::vector<cuda::GpuClipLayer> gl(cfg.blockCount);
    for (std::uint32_t i = 0; i < cfg.blockCount; ++i) {
      const auto& L = weights->layers[i];
      auto& g = gl[i];
      g.wq = detail::toGpuClipWeight(L.wq);
      g.wk = detail::toGpuClipWeight(L.wk);
      g.wv = detail::toGpuClipWeight(L.wv);
      g.wo = detail::toGpuClipWeight(L.wo);
      g.bq = L.bq.empty() ? nullptr : L.bq.data();
      g.bk = L.bk.empty() ? nullptr : L.bk.data();
      g.bv = L.bv.empty() ? nullptr : L.bv.data();
      g.bo = L.bo.empty() ? nullptr : L.bo.data();
      g.ln1W = L.ln1W.data();
      g.ln1B = L.ln1B.empty() ? nullptr : L.ln1B.data();
      g.ln2W = L.ln2W.data();
      g.ln2B = L.ln2B.empty() ? nullptr : L.ln2B.data();
      g.ffnExpand = detail::toGpuClipWeight(L.ffnExpand);
      g.ffnExpandB = L.ffnExpandB.empty() ? nullptr : L.ffnExpandB.data();
      g.ffnContract = detail::toGpuClipWeight(L.ffnContract);
      g.ffnContractB = L.ffnContractB.empty() ? nullptr : L.ffnContractB.data();
    }

    auto cm = cuda::createClipVisionModel(cc, gw, gl, err);
    if (!cm) return nullptr;

    struct CudaVisionAdapter : public vision::ClipVisionModel {
      std::unique_ptr<cuda::ClipVisionModel> impl;
      vision::ClipConfig cfg;
      int projDim = 0;
      bool hasProj = false;
      explicit CudaVisionAdapter(std::unique_ptr<cuda::ClipVisionModel> m, vision::ClipConfig c,
                                 int pDim, bool hProj)
          : impl(std::move(m)), cfg(std::move(c)), projDim(pDim), hasProj(hProj) {}
      bool encodePixels(const std::vector<float>& chw, std::vector<float>& out,
                        std::string& e) override {
        return impl->encodePixels(chw, out, e);
      }
      bool project(const std::vector<float>& hidden, std::vector<float>& out,
                   std::string& e) override {
        return impl->project(hidden, out, e);
      }
      const vision::ClipConfig& config() const override { return cfg; }
      vision::PreprocessConfig preprocessConfig() const override {
        vision::PreprocessConfig p;
        p.size = static_cast<int>(cfg.imageSize);
        p.mean = cfg.imageMean;
        p.std = cfg.imageStd;
        return p;
      }
      std::uint32_t embeddingLength() const override {
        return static_cast<std::uint32_t>(impl->embeddingLength());
      }
      std::uint32_t patchTokens() const override {
        return static_cast<std::uint32_t>(impl->patchTokens());
      }
      int projectedDim() const override { return projDim; }
      bool hasProjector() const override { return hasProj; }
      std::string backendName() const override { return impl->backendName(); }
    };
    return std::make_unique<CudaVisionAdapter>(std::move(cm), std::move(cfg),
                                              weights->projectedDim(), weights->hasProjector());
  }

  err = "vulkan clip backend not yet implemented";
  return nullptr;
}

}  // namespace qorvix
