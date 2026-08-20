#include "qorvix/image/text_encoder.hpp"

#include "qorvix/image/nn.hpp"
#include "qorvix/runtime/ops.hpp"

namespace qorvix::image {

namespace ops = qorvix::runtime::ops;

TextEncoder::TextEncoder(const SdTextConfig& cfg, const TextEncoderWeights& weights)
    : cfg_(cfg), w_(&weights) {}

bool TextEncoder::encode(const std::vector<int>& ids, int clipSkip, std::vector<float>& out,
                         std::string& error) {
  const int n = cfg_.contextLength;
  const int d = cfg_.dModel;
  if (static_cast<int>(ids.size()) != n) {
    error = "text encoder expects exactly " + std::to_string(n) + " token ids, got " +
            std::to_string(ids.size());
    return false;
  }
  const int layers = static_cast<int>(w_->layers.size());
  if (clipSkip < 1 || clipSkip > layers) {
    error = "clip-skip must be between 1 and " + std::to_string(layers);
    return false;
  }
  const int runLayers = layers - (clipSkip - 1);

  hidden_.assign(static_cast<std::size_t>(n) * d, 0.0f);
  for (int i = 0; i < n; ++i) {
    const int id = ids[static_cast<std::size_t>(i)];
    if (id < 0 || id >= cfg_.vocab) {
      error = "token id " + std::to_string(id) + " is outside the vocabulary";
      return false;
    }
    float* row = hidden_.data() + static_cast<std::size_t>(i) * d;
    runtime::embeddingRow(w_->tokenEmbd, id, row);
    const float* pos = w_->positionEmbd.data() + static_cast<std::size_t>(i) * d;
    for (int j = 0; j < d; ++j) row[j] += pos[j];
  }

  norm_.resize(hidden_.size());
  q_.resize(hidden_.size());
  k_.resize(hidden_.size());
  v_.resize(hidden_.size());
  attn_.resize(hidden_.size());
  proj_.resize(hidden_.size());
  ffn_.resize(static_cast<std::size_t>(n) * cfg_.ffn);

  for (int l = 0; l < runLayers; ++l) {
    const TextLayerWeights& L = w_->layers[static_cast<std::size_t>(l)];

    for (int i = 0; i < n; ++i) {
      ops::layernorm(norm_.data() + static_cast<std::size_t>(i) * d,
                     hidden_.data() + static_cast<std::size_t>(i) * d, L.ln1.w.data(),
                     L.ln1.b.data(), d, cfg_.normEpsilon);
    }
    runtime::wmatmulNBias(q_.data(), L.q.w, norm_.data(), n, L.q.b);
    runtime::wmatmulNBias(k_.data(), L.k.w, norm_.data(), n, L.k.b);
    runtime::wmatmulNBias(v_.data(), L.v.w, norm_.data(), n, L.v.b);
    attention(attn_.data(), q_.data(), k_.data(), v_.data(), n, n, cfg_.heads, cfg_.headDim(),
              /*causal=*/true);
    runtime::wmatmulNBias(proj_.data(), L.out.w, attn_.data(), n, L.out.b);
    addInPlace(hidden_.data(), proj_.data(), hidden_.size());

    for (int i = 0; i < n; ++i) {
      ops::layernorm(norm_.data() + static_cast<std::size_t>(i) * d,
                     hidden_.data() + static_cast<std::size_t>(i) * d, L.ln2.w.data(),
                     L.ln2.b.data(), d, cfg_.normEpsilon);
    }
    runtime::wmatmulNBias(ffn_.data(), L.ffnUp.w, norm_.data(), n, L.ffnUp.b);
    // quick-GELU on the SD 1.x CLIP-L encoder, exact GELU on the SD 2.x OpenCLIP one. Which of
    // the two is a per-file fact (sd.text.activation), not a per-architecture constant.
    if (cfg_.quickGelu) {
      ops::geluQuickInPlace(ffn_.data(), static_cast<int>(ffn_.size()));
    } else {
      ops::geluInPlace(ffn_.data(), static_cast<int>(ffn_.size()));
    }
    runtime::wmatmulNBias(proj_.data(), L.ffnDown.w, ffn_.data(), n, L.ffnDown.b);
    addInPlace(hidden_.data(), proj_.data(), hidden_.size());
  }

  out.resize(hidden_.size());
  for (int i = 0; i < n; ++i) {
    ops::layernorm(out.data() + static_cast<std::size_t>(i) * d,
                   hidden_.data() + static_cast<std::size_t>(i) * d, w_->lnFinal.w.data(),
                   w_->lnFinal.b.data(), d, cfg_.normEpsilon);
  }
  return true;
}

}  // namespace qorvix::image
