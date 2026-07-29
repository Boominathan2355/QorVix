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
// through the exact same seam as CPU and CUDA. The Vulkan twin of GpuEngine, multi-session: each
// session owns an independent KV slice, so `serve --vulkan --max-concurrent N` works like CUDA.
namespace qorvix {

class VulkanEngine final : public runtime::IInferenceEngine {
 public:
  VulkanEngine(std::unique_ptr<vulkan::VulkanModel> model, runtime::ModelConfig cfg,
               std::uint32_t maxSeq)
      : model_(std::move(model)), cfg_(std::move(cfg)), maxSeq_(maxSeq) {}

  // SessionId 0 is memory::kInvalidSession (the "no session" sentinel), but Vulkan session 0 is a
  // valid slot — so the two spaces are offset by one (identical to GpuEngine).
  memory::SessionId openSession() override {
    const int s = model_->openSession();
    if (s == vulkan::kNoVkSession) return memory::kInvalidSession;
    return static_cast<memory::SessionId>(s + 1);
  }
  void closeSession(memory::SessionId s) override {
    if (s != memory::kInvalidSession) model_->closeSession(toVk(s));
  }
  void resetSession(memory::SessionId s) override {
    if (s != memory::kInvalidSession) model_->resetSession(toVk(s));
  }

  const std::vector<float>& forward(memory::SessionId session, int token, int pos) override {
    return model_->forward(toVk(session), token, pos);
  }

  std::uint32_t maxSeqLen() const override { return maxSeq_; }
  const runtime::ModelConfig& config() const override { return cfg_; }
  std::string backendName() const override { return "vulkan"; }

 private:
  static int toVk(memory::SessionId s) { return static_cast<int>(s) - 1; }

  std::unique_ptr<vulkan::VulkanModel> model_;
  runtime::ModelConfig cfg_;
  std::uint32_t maxSeq_ = 0;
};

}  // namespace qorvix
