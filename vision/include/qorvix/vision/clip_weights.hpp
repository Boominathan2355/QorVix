#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/runtime/weights.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::vision {

// Hyperparameters of a CLIP vision tower, read from a `clip` GGUF (llama.cpp's mmproj format).
struct ClipConfig {
  std::string name;
  std::uint32_t imageSize = 336;
  std::uint32_t patchSize = 14;
  std::uint32_t embeddingLength = 1024;    // d_model of the vision tower
  std::uint32_t feedForwardLength = 4096;
  std::uint32_t headCount = 16;
  std::uint32_t blockCount = 23;
  std::uint32_t projectionDim = 768;
  float normEpsilon = 1e-5f;

  // clip.use_gelu selects the FFN activation: true -> exact GELU, false -> "quick" GELU
  // (x * sigmoid(1.702x)), which is what OpenAI's CLIP was trained with and what the LLaVA
  // towers ship. Defaulting the wrong way is a silent ~1e-2 drift, not an error.
  bool useGelu = false;

  bool hasVisionEncoder = true;
  bool hasLlavaProjector = false;
  std::string projectorType = "mlp";

  // Preprocessing constants live in the file rather than in our source, so a tower trained with
  // different statistics cannot be silently normalized with OpenAI's.
  std::array<float, 3> imageMean{0.48145466f, 0.4578275f, 0.40821073f};
  std::array<float, 3> imageStd{0.26862954f, 0.26130258f, 0.27577711f};

  std::uint32_t patchesPerSide() const { return patchSize ? imageSize / patchSize : 0; }
  // +1 for the class token that CLIP prepends.
  std::uint32_t tokenCount() const { return patchesPerSide() * patchesPerSide() + 1; }
  std::uint32_t headDim() const { return headCount ? embeddingLength / headCount : 0; }

  bool valid() const {
    return embeddingLength && headCount && blockCount && feedForwardLength && patchSize &&
           imageSize && imageSize % patchSize == 0 && embeddingLength % headCount == 0;
  }
};

ClipConfig clipConfigFromGguf(const gguf::GgufFile& file, std::string& error);

// One CLIP vision block. Pre-norm, unlike the BERT encoder's post-norm: ln1 runs BEFORE attention
// and ln2 BEFORE the MLP, with the residual added after each sublayer.
struct ClipLayerWeights {
  runtime::WeightMat wq, wk, wv, wo;         // v.blk.N.attn_{q,k,v,out}.weight  [d, d]
  std::vector<float> bq, bk, bv, bo;
  std::vector<float> ln1W, ln1B;             // v.blk.N.ln1.*
  std::vector<float> ln2W, ln2B;             // v.blk.N.ln2.*

  // NOTE THE INVERTED NAMES. In a clip mmproj GGUF, `ffn_down` is the EXPANSION (d -> ffn) and
  // `ffn_up` is the CONTRACTION (ffn -> d) — the opposite of every other architecture in this
  // repo. Verified against the file: v.blk.0.ffn_down.weight is [1024,4096] with a [4096] bias,
  // and v.blk.0.ffn_up.weight is [4096,1024] with a [1024] bias. Loading them by name into the
  // usual slots produces a shape error at best and silent garbage at worst, so the fields here
  // are named for what they DO, not for what the tensor is called.
  runtime::WeightMat ffnExpand;              // <- v.blk.N.ffn_down.weight  [ffn, d]
  std::vector<float> ffnExpandB;             // <- v.blk.N.ffn_down.bias    [ffn]
  runtime::WeightMat ffnContract;            // <- v.blk.N.ffn_up.weight    [d, ffn]
  std::vector<float> ffnContractB;           // <- v.blk.N.ffn_up.bias      [d]
};

struct ClipWeights {
  runtime::WeightMat patchEmbd;              // v.patch_embd.weight, as [d, patch*patch*3]
  std::vector<float> classEmbd;              // v.class_embd  [d]
  runtime::WeightMat positionEmbd;           // v.position_embd.weight  [tokens, d]
  std::vector<float> preLnW, preLnB;         // v.pre_ln.*
  std::vector<float> postLnW, postLnB;       // v.post_ln.* — absent in LLaVA towers
  std::vector<ClipLayerWeights> layers;

  // LLaVA "mlp" projector: mm.0 (d -> llm), GELU, mm.2 (llm -> llm).
  runtime::WeightMat mm0, mm2;
  std::vector<float> mm0B, mm2B;
  bool hasProjector() const { return mm0.valid() && mm2.valid(); }
  int projectedDim() const { return mm2.valid() ? mm2.rows : 0; }
};

std::optional<ClipWeights> loadClipWeights(const gguf::GgufFile& file, const ClipConfig& cfg,
                                           std::string& error);

}  // namespace qorvix::vision
