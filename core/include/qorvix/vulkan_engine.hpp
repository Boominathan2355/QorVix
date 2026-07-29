#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "qorvix/memory/kv_cache.hpp"
#include "qorvix/runtime/inference_engine.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/vulkan/vulkan_model.hpp"

// Adapts a vulkan::VulkanModel to runtime::IInferenceEngine so the scheduler — and therefore the
// unified createEngine() factory, `generate`, and `serve` — drive the cross-vendor Vulkan path
// through the exact same seam as CPU and CUDA. The Vulkan twin of GpuEngine.
//
// VulkanModel is currently single-sequence (one KV cache), so this engine exposes a single session
// slot: the first openSession() succeeds, further ones report "at capacity" (kInvalidSession) until
// it is closed. That is enough for `generate` and single-stream `serve`; a multi-session VulkanModel
// (per-session KV slices, like GpuModel) is the next step to raise --max-concurrent above 1.
namespace qorvix {

class VulkanEngine final : public runtime::IInferenceEngine {
 public:
  VulkanEngine(std::unique_ptr<vulkan::VulkanModel> model, runtime::ModelConfig cfg,
               std::uint32_t maxSeq)
      : model_(std::move(model)), cfg_(std::move(cfg)), maxSeq_(maxSeq) {}

  memory::SessionId openSession() override {
    if (inUse_) return memory::kInvalidSession;  // single slot
    inUse_ = true;
    model_->reset();
    return kSlot;
  }
  void closeSession(memory::SessionId s) override {
    if (s == kSlot) inUse_ = false;
  }
  void resetSession(memory::SessionId s) override {
    if (s == kSlot) model_->reset();
  }

  const std::vector<float>& forward(memory::SessionId, int token, int pos) override {
    return model_->forward(token, pos);
  }

  std::uint32_t maxSeqLen() const override { return maxSeq_; }
  const runtime::ModelConfig& config() const override { return cfg_; }
  std::string backendName() const override { return "vulkan"; }

 private:
  static constexpr memory::SessionId kSlot = 1;  // any non-kInvalidSession id

  std::unique_ptr<vulkan::VulkanModel> model_;
  runtime::ModelConfig cfg_;
  std::uint32_t maxSeq_ = 0;
  bool inUse_ = false;
};

}  // namespace qorvix
