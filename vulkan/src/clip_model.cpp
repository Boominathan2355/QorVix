// Vulkan CLIP vision tower (Phase 11c) - stub implementation.
// The full Vulkan implementation needs compute shaders for patch embedding, pre-norm attention,
// quick-GELU, and the MLP projector. Factory returns nullptr with an error until ready.
#include "qorvix/vulkan/clip_model.hpp"

namespace qorvix::vulkan {

std::unique_ptr<VulkanClipVisionModel> createVulkanClipVisionModel(
    const ClipConfig& cfg, const VulkanClipWeights& weights,
    const std::vector<VulkanClipLayer>& layers, std::string& error) {
  (void)cfg; (void)weights; (void)layers;
  error = "vulkan clip backend not yet implemented";
  return nullptr;
}

}  // namespace qorvix::vulkan
