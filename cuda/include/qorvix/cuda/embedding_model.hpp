#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// CUDA BERT-family embedding engine — GPU implementation of qorvix::embeddings::IEmbeddingEngine.
// Uploads all weights to VRAM (quantized, never dequantized) and runs the encoder forward pass
// on-device. Mirrors the CPU BertModel: post-norm LayerNorm with bias, bidirectional attention,
// GELU FFN, learned position embeddings, and CLS/mean/last pooling. Returns nullptr in CPU-only
// builds or when no device is available.
namespace qorvix::cuda {

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
    bool hasPositionEmbd = true;   // false => RoPE (not supported yet)
    bool hasEmbdNorm = false;
    int defaultPooling = 2;        // 0=CLS, 1=Last, 2=Mean, 3=None
    bool defaultNormalize = true;
};

// Quantized weight [rows, cols] as raw host bytes in its GGUF quant type (ggmlType: F32=0,
// F16=1, BF16=2, Q8_0=8, Q4_K=12, Q6_K=14). Uploaded verbatim to VRAM.
// Named EmbedWeight to avoid collision with cuda::GpuWeight in gpu_model.hpp.
struct EmbedWeight {
    const void* host = nullptr;
    std::uint32_t ggmlType = 0;
    int rows = 0;
    int cols = 0;
    bool valid() const { return host != nullptr && rows > 0 && cols > 0; }
};

// Per-layer weights for the encoder.
struct GpuEmbeddingLayer {
    const float* attnNorm = nullptr;       // [dModel], F32
    const float* attnNormB = nullptr;      // [dModel], F32 (bias)
    EmbedWeight wq, wk, wv, wo;            // attention projections
    const float* bq = nullptr;             // [dModel] or null
    const float* bk = nullptr;
    const float* bv = nullptr;
    const float* bo = nullptr;             // [dModel] or null
    const float* ffnNorm = nullptr;        // [dModel], F32
    const float* ffnNormB = nullptr;       // [dModel], F32 (bias)
    EmbedWeight ffnGate, ffnUp, ffnDown;    // FFN projections
    const float* ffnUpB = nullptr;         // [ffn] or null
    const float* ffnGateB = nullptr;       // [ffn] or null
    const float* ffnDownB = nullptr;       // [dModel] or null
};

// Embedding + position + token-type tables (all F32 on device, dequantized at load time).
struct GpuEmbeddingTables {
    float* tokenEmbd = nullptr;        // [vocab * dModel], F32
    float* positionEmbd = nullptr;     // [maxSeq * dModel], F32 (or null if RoPE)
    float* tokenTypes = nullptr;       // [tokenTypeCount * dModel], F32 (or null)
    float* embdNorm = nullptr;         // [dModel], F32
    float* embdNormB = nullptr;        // [dModel], F32 (bias)
};

class EmbeddingModel {
  public:
    virtual ~EmbeddingModel() = default;

    // Encodes one token sequence using the model's own pooling and normalization.
    // Writes dim() floats into `out`. Returns false with `error` set on failure.
    virtual bool embed(const std::vector<int>& tokens, std::vector<float>& out,
                       std::string& error) = 0;

    // Same, with pooling and normalization overridden.
    virtual bool embedWith(const std::vector<int>& tokens, int pooling, bool normalize,
                           std::vector<float>& out, std::string& error) = 0;

    // Per-token hidden states [tokens.size() * dim()] row-major, unpooled/unnormalized.
    virtual bool embedTokens(const std::vector<int>& tokens, std::vector<float>& out,
                             std::string& error) = 0;

    // Batch encode in request order.
    virtual bool embedBatch(const std::vector<std::vector<int>>& batch,
                            std::vector<std::vector<float>>& out, std::string& error) = 0;

    virtual int dim() const = 0;              // embedding width (== d_model)
    virtual int maxSeqLen() const = 0;        // hard truncation limit
    virtual int defaultPooling() const = 0;   // 0=CLS, 1=Last, 2=Mean, 3=None
    virtual bool defaultNormalize() const = 0;
    virtual std::string backendName() const = 0;  // "cuda"
};

// Builds an EmbeddingModel from descriptors. Returns nullptr with `error` set on failure (no
// device, alloc failure, unsupported model).
std::unique_ptr<EmbeddingModel> createEmbeddingModel(
    const EmbeddingConfig& cfg, const GpuEmbeddingTables& tables,
    const std::vector<GpuEmbeddingLayer>& layers, std::string& error);

}  // namespace qorvix::cuda