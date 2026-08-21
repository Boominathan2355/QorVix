#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// CUDA CLIP vision tower - GPU implementation of the vision encoder.
// Reuses the same encoder kernels as the embedding engine (LayerNorm with bias, bidirectional
// attention, quantized GEMV) with CLIP-specific differences: pre-norm, quick-GELU, patch
// embedding via matmul, learned position embeddings, and no pooling (returns per-patch states).
namespace qorvix::cuda {

struct ClipConfig {
    int imageSize = 336;
    int patchSize = 14;
    int embeddingLength = 1024;
    int feedForwardLength = 4096;
    int headCount = 16;
    int blockCount = 23;
    int projectionDim = 768;
    float normEpsilon = 1e-5f;
    bool useGelu = false;          // true=GELU, false=quick-GELU
    bool hasProjector = false;
    int projectedDim = 0;
    float imageMean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    float imageStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};

    int patchesPerSide() const { return patchSize ? imageSize / patchSize : 0; }
    int tokenCount() const { return patchesPerSide() * patchesPerSide() + 1; } // +1 for class token
    int headDim() const { return headCount ? embeddingLength / headCount : 0; }
};

struct GpuClipWeight {
    const void* host = nullptr;
    std::uint32_t ggmlType = 0;
    int rows = 0;
    int cols = 0;
};

struct GpuClipLayer {
    GpuClipWeight wq, wk, wv, wo;
    const float* bq = nullptr;
    const float* bk = nullptr;
    const float* bv = nullptr;
    const float* bo = nullptr;
    const float* ln1W = nullptr;
    const float* ln1B = nullptr;
    const float* ln2W = nullptr;
    const float* ln2B = nullptr;
    GpuClipWeight ffnExpand;
    const float* ffnExpandB = nullptr;
    GpuClipWeight ffnContract;
    const float* ffnContractB = nullptr;
};

struct GpuClipWeights {
    GpuClipWeight patchEmbd;
    const float* classEmbd = nullptr;
    GpuClipWeight positionEmbd;
    const float* preLnW = nullptr;
    const float* preLnB = nullptr;
    const float* postLnW = nullptr;
    const float* postLnB = nullptr;
    GpuClipWeight mm0;
    const float* mm0B = nullptr;
    GpuClipWeight mm2;
    const float* mm2B = nullptr;
};

class ClipVisionModel {
public:
    virtual ~ClipVisionModel() = default;
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

std::unique_ptr<ClipVisionModel> createClipVisionModel(
    const ClipConfig& cfg, const GpuClipWeights& weights,
    const std::vector<GpuClipLayer>& layers, std::string& error);

}  // namespace qorvix::cuda
