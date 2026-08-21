#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Vulkan CLIP vision tower - stub for Phase 11c.
// Full implementation requires compute shaders for patch embedding, pre-norm attention,
// quick-GELU, and the MLP projector. Factory returns nullptr until the shaders are ready.
namespace qorvix::vulkan {

struct ClipConfig {
    int imageSize = 336;
    int patchSize = 14;
    int embeddingLength = 1024;
    int feedForwardLength = 4096;
    int headCount = 16;
    int blockCount = 23;
    int projectionDim = 768;
    float normEpsilon = 1e-5f;
    bool useGelu = false;
    bool hasProjector = false;
    int projectedDim = 0;

    int patchesPerSide() const { return patchSize ? imageSize / patchSize : 0; }
    int tokenCount() const { return patchesPerSide() * patchesPerSide() + 1; }
    int headDim() const { return headCount ? embeddingLength / headCount : 0; }
};

struct VulkanClipWeight {
    const void* host = nullptr;
    std::uint32_t ggmlType = 0;
    int rows = 0;
    int cols = 0;
};

struct VulkanClipLayer {
    VulkanClipWeight wq, wk, wv, wo;
    const float* bq = nullptr, *bk = nullptr, *bv = nullptr, *bo = nullptr;
    const float* ln1W = nullptr, *ln1B = nullptr;
    const float* ln2W = nullptr, *ln2B = nullptr;
    VulkanClipWeight ffnExpand, ffnContract;
    const float* ffnExpandB = nullptr, *ffnContractB = nullptr;
};

struct VulkanClipWeights {
    VulkanClipWeight patchEmbd;
    const float* classEmbd = nullptr;
    VulkanClipWeight positionEmbd;
    const float* preLnW = nullptr, *preLnB = nullptr;
    const float* postLnW = nullptr, *postLnB = nullptr;
    VulkanClipWeight mm0, mm2;
    const float* mm0B = nullptr, *mm2B = nullptr;
};

class VulkanClipVisionModel {
public:
    virtual ~VulkanClipVisionModel() = default;
    virtual bool encodePixels(const std::vector<float>& chw, std::vector<float>& out,
                              std::string& error) = 0;
    virtual bool project(const std::vector<float>& hidden, std::vector<float>& out,
                         std::string& error) = 0;
    virtual int embeddingLength() const = 0;
    virtual int patchTokens() const = 0;
    virtual int projectedDim() const = 0;
    virtual bool hasProjector() const = 0;
    virtual std::string backendName() const = 0;
};

std::unique_ptr<VulkanClipVisionModel> createVulkanClipVisionModel(
    const ClipConfig& cfg, const VulkanClipWeights& weights,
    const std::vector<VulkanClipLayer>& layers, std::string& error);

}  // namespace qorvix::vulkan
