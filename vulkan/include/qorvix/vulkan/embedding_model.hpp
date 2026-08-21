#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Vulkan BERT-family embedding engine - GPU implementation of IEmbeddingEngine.
// Same architecture as the CUDA embedding engine: uploads quantized weights to device buffers,
// runs the encoder with compute shaders, returns pooled/normalized vectors.
namespace qorvix::vulkan {

struct EmbeddingConfig {
    int nLayers = 0;
    int dModel = 0;
    int nHeads = 0;
    int headDim = 0;
    int ffn = 0;
    int vocab = 0;
    int maxSeq = 0;
    float normEps = 1e-5f;
    bool ffnGated = false;
    bool hasTokenTypes = false;
    int tokenTypeCount = 0;
    bool hasPositionEmbd = true;
    bool hasEmbdNorm = false;
    int defaultPooling = 0;  // 0=CLS, 1=Last, 2=Mean
    bool defaultNormalize = true;
};

struct VulkanEmbedWeight {
    const void* host = nullptr;
    std::uint32_t ggmlType = 0;
    int rows = 0;
    int cols = 0;
    bool valid() const { return host != nullptr && rows > 0 && cols > 0; }
};

struct VulkanEmbedLayer {
    const float* attnNorm = nullptr;
    const float* attnNormB = nullptr;
    VulkanEmbedWeight wq, wk, wv, wo;
    const float* bq = nullptr;
    const float* bk = nullptr;
    const float* bv = nullptr;
    const float* bo = nullptr;
    const float* ffnNorm = nullptr;
    const float* ffnNormB = nullptr;
    VulkanEmbedWeight ffnUp, ffnDown;
    const float* ffnUpB = nullptr;
    const float* ffnDownB = nullptr;
    VulkanEmbedWeight ffnGate;
};

struct VulkanEmbeddingTables {
    float* tokenEmbd = nullptr;
    float* positionEmbd = nullptr;
    float* tokenTypes = nullptr;
    float* embdNorm = nullptr;
    float* embdNormB = nullptr;
};

class VulkanEmbeddingModel {
public:
    virtual ~VulkanEmbeddingModel() = default;
    virtual bool embed(const std::vector<int>& tokens, std::vector<float>& out,
                       std::string& error) = 0;
    virtual bool embedWith(const std::vector<int>& tokens, int pooling, bool normalize,
                           std::vector<float>& out, std::string& error) = 0;
    virtual bool embedTokens(const std::vector<int>& tokens, std::vector<float>& out,
                             std::string& error) = 0;
    virtual bool embedBatch(const std::vector<std::vector<int>>& batch,
                            std::vector<std::vector<float>>& out, std::string& error) {
      out.resize(batch.size());
      for (std::size_t i = 0; i < batch.size(); ++i) {
        if (!embed(batch[i], out[i], error)) return false;
      }
      return true;
    }
    virtual int dim() const = 0;
    virtual int maxSeqLen() const = 0;
    virtual int defaultPooling() const = 0;
    virtual bool defaultNormalize() const = 0;
    virtual std::string backendName() const = 0;
};

std::unique_ptr<VulkanEmbeddingModel> createVulkanEmbeddingModel(
    const EmbeddingConfig& cfg, const VulkanEmbeddingTables& tables,
    const std::vector<VulkanEmbedLayer>& layers, std::string& error);

}  // namespace qorvix::vulkan
