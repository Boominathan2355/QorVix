#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/vision/clip_weights.hpp"
#include "qorvix/vision/image.hpp"
#include "qorvix/vision/preprocess.hpp"

namespace qorvix::vision {

// CPU CLIP vision tower — SPEC's "Image → Vision Encoder → Projected Embeddings" stage.
//
// Architecturally a ViT is a BERT with patch embeddings instead of token embeddings, so this
// reuses the Phase 11a machinery wholesale: bidirectional attention, LayerNorm with bias, the
// batched quantized GEMV. The differences that matter are all called out where they occur —
// PRE-norm rather than post-norm, quick-GELU rather than GELU, a prepended class token, and a
// learned position table over patches.
//
// Not thread-safe: per-call scratch lives in members so it is allocated once. Callers sharing an
// instance must serialize it, as `serve` does for the other engines.
class ClipVisionModel {
 public:
  static std::optional<ClipVisionModel> fromGguf(gguf::GgufFile file, std::string& error);

  ClipVisionModel(ClipConfig cfg, ClipWeights weights);
  ClipVisionModel(ClipVisionModel&&) noexcept = default;
  ClipVisionModel& operator=(ClipVisionModel&&) noexcept = default;

  // Encodes preprocessed CHW pixel data ([3, imageSize, imageSize]) into per-patch hidden states,
  // [patchTokens(), embeddingLength()] row-major, with the class token dropped.
  //
  // LLaVA consumes the SECOND-TO-LAST hidden state of the original 24-layer tower. The mmproj
  // conversion already dropped the final block — hence block_count 23 — so running every block
  // present here yields exactly that state, with no layer-skipping logic needed.
  bool encodePixels(const std::vector<float>& chw, std::vector<float>& out, std::string& error);

  // Full path: decode-independent image → preprocess → encode. `out` is as above.
  bool encodeImage(const Image& img, std::vector<float>& out, std::string& error);

  // Applies the LLaVA mlp projector to per-patch hidden states, producing the vectors the
  // language model consumes: [patchTokens(), projectedDim()].
  bool project(const std::vector<float>& hidden, std::vector<float>& out, std::string& error);

  const ClipConfig& config() const { return cfg_; }
  PreprocessConfig preprocessConfig() const;
  std::uint32_t embeddingLength() const { return cfg_.embeddingLength; }
  // Patch tokens only — the class token is dropped, which is what LLaVA feeds the decoder.
  std::uint32_t patchTokens() const { return cfg_.tokenCount() - 1; }
  int projectedDim() const { return w_.projectedDim(); }
  bool hasProjector() const { return w_.hasProjector(); }

 private:
  void attention(const ClipLayerWeights& L, int n);

  ClipConfig cfg_;
  ClipWeights w_;
  std::unique_ptr<gguf::GgufFile> file_;  // keeps the mmap alive behind the borrowed weights

  std::vector<float> states_, norm_, q_, k_, v_, attn_, tmp_, ffn_, scores_;
};

}  // namespace qorvix::vision
