// Vulkan BERT-family embedding engine (Phase 11c) - stub implementation.
// The full Vulkan implementation requires compute shaders for LayerNorm, GELU, bidirectional
// attention, etc. This stub returns nullptr so the factory can wire in CPU as fallback.
// The real implementation will follow the same pattern as vulkan_backend.cpp.

#include "qorvix/vulkan/embedding_model.hpp"

namespace qorvix::vulkan {

std::unique_ptr<VulkanEmbeddingModel> createVulkanEmbeddingModel(
    const EmbeddingConfig& cfg, const VulkanEmbeddingTables& tables,
    const std::vector<VulkanEmbedLayer>& layers, std::string& error) {
  (void)cfg; (void)tables; (void)layers;
  error = "vulkan embedding backend not yet implemented";
  return nullptr;
}

}  // namespace qorvix::vulkan
