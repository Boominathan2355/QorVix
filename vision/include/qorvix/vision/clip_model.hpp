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

  ClipVisionModel() = default;
  ClipVisionModel(ClipConfig cfg, ClipWeights weights);
  virtual ~ClipVisionModel() = default;
  ClipVisionModel(ClipVisionModel&&) noexcept = default;
  ClipVisionModel& operator=(ClipVisionModel&&) noexcept = default;

  // Encodes preprocessed CHW pixel data ([3, imageSize, imageSize]) into per-patch hidden states,
  // [patchTokens(), embeddingLength()] row-major, with the class token dropped.
  //
  // LLaVA consumes the SECOND-TO-LAST hidden state of the original 24-layer tower. The mmproj
  // conversion already dropped the final block — hence block_count 23 — so running every block
  // present here yields exactly that state, with no layer-skipping logic needed.
  virtual bool encodePixels(const std::vector<float>& chw, std::vector<float>& out, std::string& error);

  // Full path: decode-independent image → preprocess → encode. `out` is as above.
  virtual bool encodeImage(const Image& img, std::vector<float>& out, std::string& error);

  // Applies the LLaVA mlp projector to per-patch hidden states, producing the vectors the
  // language model consumes: [patchTokens(), projectedDim()].
  virtual bool project(const std::vector<float>& hidden, std::vector<float>& out, std::string& error);

  // The whole vision half of a VLM turn in one call: image -> preprocess -> tower -> projector,
  // producing [patchTokens(), projectedDim()] ready to splice into the decoder's input embeddings
  // (Phase 11b-2). Fails with a clear error when the file carries no projector, since a tower-only
  // mmproj yields 1024-d vectors that would silently mismatch a 4096-d decoder.
  virtual bool encodeProjected(const Image& img, std::vector<float>& out, std::string& error);

  virtual const ClipConfig& config() const { return cfg_; }
  virtual PreprocessConfig preprocessConfig() const;
  virtual std::uint32_t embeddingLength() const { return cfg_.embeddingLength; }
  // Patch tokens only — the class token is dropped, which is what LLaVA feeds the decoder.
  virtual std::uint32_t patchTokens() const { return cfg_.tokenCount() > 0 ? cfg_.tokenCount() - 1 : 0; }
  virtual int projectedDim() const { return w_.projectedDim(); }
  virtual bool hasProjector() const { return w_.hasProjector(); }
  virtual std::string backendName() const { return "cpu"; }

 private:
  void attention(const ClipLayerWeights& L, int n);

  ClipConfig cfg_;
  ClipWeights w_;
  std::unique_ptr<gguf::GgufFile> file_;  // keeps the mmap alive behind the borrowed weights

  std::vector<float> states_, norm_, q_, k_, v_, attn_, tmp_, ffn_, scores_;
};

}  // namespace qorvix::vision
