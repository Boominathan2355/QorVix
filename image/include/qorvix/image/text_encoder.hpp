#pragma once

#include <string>
#include <vector>

#include "qorvix/image/sd_weights.hpp"

namespace qorvix::image {

// The CLIP text encoder that produces a diffusion model's conditioning.
//
// Architecturally this is the CLIP VISION tower of Phase 11b-1 with two substitutions: token +
// position embeddings instead of patch embeddings, and a CAUSAL mask instead of a bidirectional
// one. Everything else — pre-norm blocks, quick-GELU, LayerNorm with bias — is the same, and the
// reuse is literal: the same batched quantized GEMV, the same ops.
//
// The causal mask is not decoration. CLIP was trained with it and every position's vector depends
// on it, so a bidirectional run here yields conditioning that is smooth, confident and not what
// the UNet was trained against.
//
// Not thread-safe: scratch lives in members so it is allocated once per instance.
class TextEncoder {
 public:
  TextEncoder(const SdTextConfig& cfg, const TextEncoderWeights& weights);

  // Encodes exactly `contextLength` token ids into [contextLength, dModel].
  //
  // `clipSkip` follows the convention the tooling around these models uses: 1 means the final
  // hidden layer (the default, and what every SD 1.x/2.x pipeline runs), 2 means the penultimate
  // one. diffusers spells the same thing `clip_skip=None` and `clip_skip=1`. The final LayerNorm
  // is applied either way — skipping a layer skips the BLOCK, not the norm.
  bool encode(const std::vector<int>& ids, int clipSkip, std::vector<float>& out,
              std::string& error);

  int dim() const { return cfg_.dModel; }

 private:
  SdTextConfig cfg_;
  const TextEncoderWeights* w_;
  std::vector<float> hidden_, norm_, q_, k_, v_, attn_, proj_, ffn_;
};

}  // namespace qorvix::image
