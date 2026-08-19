#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "qorvix/cuda/gpu_model.hpp"
#include "qorvix/memory/kv_cache.hpp"
#include "qorvix/runtime/inference_engine.hpp"
#include "qorvix/runtime/model_config.hpp"

// Adapts a cuda::GpuModel to runtime::IInferenceEngine so the scheduler — and therefore `serve`
// and the whole HTTP layer — can drive the GPU path. Before this, the GPU was reachable only from
// `generate --gpu`'s private loop, so the fast path could not be served over HTTP.
//
// Header-only and living in `core` on purpose: it is the one place that already sees both the cuda
// and runtime modules, and making either one depend on the other would invert the dependency (the
// cuda module is deliberately independent of runtime/gguf types — see gpu_model.hpp).
namespace qorvix {

class GpuEngine final : public runtime::IInferenceEngine {
 public:
  // `name` is what backendName() reports. It exists so a tensor-parallel model can say so in the
  // logs ("cuda-tp4") without the engine growing a second implementation — the parallelism lives
  // entirely inside the GpuModel, and the name is the only part of it the seam can observe.
  GpuEngine(std::unique_ptr<cuda::GpuModel> model, runtime::ModelConfig cfg, std::uint32_t maxSeq,
            std::string name = "cuda")
      : model_(std::move(model)), cfg_(std::move(cfg)), maxSeq_(maxSeq), name_(std::move(name)) {}

  // SessionId 0 is memory::kInvalidSession (the "no session" sentinel), but GPU session 0 is a
  // perfectly valid slot — so the two spaces are offset by one rather than passed through.
  memory::SessionId openSession() override {
    const int s = model_->openSession();
    if (s == cuda::kNoGpuSession) return memory::kInvalidSession;
    return static_cast<memory::SessionId>(s + 1);
  }
  void closeSession(memory::SessionId s) override {
    if (s != memory::kInvalidSession) model_->closeSession(toGpu(s));
  }
  void resetSession(memory::SessionId s) override {
    if (s != memory::kInvalidSession) model_->resetSession(toGpu(s));
  }

  const std::vector<float>& forward(memory::SessionId session, int token, int pos) override {
    return model_->forward(toGpu(session), token, pos);
  }

  // Phase 11b-2 opened this seam and both device backends refused it until Phase 11c: they keep the
  // embedding table in VRAM and look rows up on-device, so accepting a host vector needed an upload
  // path and a different kernel head, not a wrapper. It exists now, so the multimodal surfaces stop
  // refusing --gpu / --vulkan.
  bool acceptsInputEmbeddings() const override { return true; }

  const std::vector<float>& forwardEmbedding(memory::SessionId session, const float* embedding,
                                             int pos) override {
    return model_->forwardEmbedding(toGpu(session), embedding, pos);
  }

  std::uint32_t maxSeqLen() const override { return maxSeq_; }
  const runtime::ModelConfig& config() const override { return cfg_; }
  std::string backendName() const override { return name_; }

 private:
  static int toGpu(memory::SessionId s) { return static_cast<int>(s) - 1; }

  std::unique_ptr<cuda::GpuModel> model_;
  runtime::ModelConfig cfg_;
  std::uint32_t maxSeq_ = 0;
  std::string name_ = "cuda";
};

}  // namespace qorvix
