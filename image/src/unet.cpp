#include "qorvix/image/unet.hpp"

#include <cmath>

#include "qorvix/image/blocks.hpp"
#include "qorvix/runtime/ops.hpp"

namespace qorvix::image {

namespace ops = qorvix::runtime::ops;

Unet::Unet(const SdUnetConfig& cfg, const UnetWeights& weights) : cfg_(cfg), w_(&weights) {}

void Unet::timestepEmbedding(int timestep, int dim, float freqShift, bool flipSinToCos,
                             std::vector<float>& out) {
  const int half = dim / 2;
  out.assign(static_cast<std::size_t>(dim), 0.0f);
  for (int i = 0; i < half; ++i) {
    // exp(-ln(10000) * i / (half - shift)) — the same geometric frequency ladder as a transformer's
    // sinusoidal positions, with `freqShift` the one knob diffusers exposes over it.
    const double exponent = -std::log(10000.0) * static_cast<double>(i) /
                            (static_cast<double>(half) - static_cast<double>(freqShift));
    const double arg = static_cast<double>(timestep) * std::exp(exponent);
    // flip_sin_to_cos puts the cosines FIRST. It is true on every Stable Diffusion release, and
    // the two halves are otherwise identical in magnitude — so flipping it produces a perfectly
    // reasonable-looking embedding that the network has never seen.
    if (flipSinToCos) {
      out[static_cast<std::size_t>(i)] = static_cast<float>(std::cos(arg));
      out[static_cast<std::size_t>(half + i)] = static_cast<float>(std::sin(arg));
    } else {
      out[static_cast<std::size_t>(i)] = static_cast<float>(std::sin(arg));
      out[static_cast<std::size_t>(half + i)] = static_cast<float>(std::cos(arg));
    }
  }
}

bool Unet::transformer(const SpatialTransformerWeights& t, FeatureMap& x, const float* context,
                       int contextTokens, int heads, std::string& error) {
  const int c = x.c;
  const int tokens = static_cast<int>(x.positions());
  if (heads <= 0 || c % heads != 0) {
    error = "spatial transformer: " + std::to_string(c) + " channels do not divide into " +
            std::to_string(heads) + " heads";
    return false;
  }
  const int headDim = c / heads;
  const int inner = c * 4;

  tfResidual_ = x;
  // 1e-6 here, not the resnets' 1e-5. See the header.
  groupNorm(x, t.norm.w, t.norm.b, cfg_.normGroups, cfg_.transformerNormEpsilon);
  if (!pointwise(tfIn_, x, t.projIn.w, t.projIn.b, error)) return false;

  // The feature map already IS [tokens, channels] — the "reshape to a sequence" every reference
  // implementation performs here is, in this layout, nothing at all.
  float* h = tfIn_.data.data();
  const std::size_t hn = tfIn_.size();
  tfNorm_.resize(hn);
  tfQ_.resize(hn);
  tfK_.resize(static_cast<std::size_t>(contextTokens > tokens ? contextTokens : tokens) * c);
  tfV_.resize(tfK_.size());
  tfAttn_.resize(hn);
  tfProj_.resize(hn);
  tfFfn_.resize(static_cast<std::size_t>(tokens) * 2 * inner);

  for (const auto& b : t.blocks) {
    for (int i = 0; i < tokens; ++i) {
      ops::layernorm(tfNorm_.data() + static_cast<std::size_t>(i) * c,
                     h + static_cast<std::size_t>(i) * c, b.ln1.w.data(), b.ln1.b.data(), c);
    }
    // Self-attention. No biases on q/k/v — the converter refuses a checkpoint that has them.
    runtime::wmatmulN(tfQ_.data(), b.q1, tfNorm_.data(), tokens);
    runtime::wmatmulN(tfK_.data(), b.k1, tfNorm_.data(), tokens);
    runtime::wmatmulN(tfV_.data(), b.v1, tfNorm_.data(), tokens);
    attention(tfAttn_.data(), tfQ_.data(), tfK_.data(), tfV_.data(), tokens, tokens, heads, headDim);
    runtime::wmatmulNBias(tfProj_.data(), b.out1.w, tfAttn_.data(), tokens, b.out1.b);
    addInPlace(h, tfProj_.data(), hn);

    for (int i = 0; i < tokens; ++i) {
      ops::layernorm(tfNorm_.data() + static_cast<std::size_t>(i) * c,
                     h + static_cast<std::size_t>(i) * c, b.ln2.w.data(), b.ln2.b.data(), c);
    }
    // Cross-attention: queries from the image, keys and values from the prompt. This is the only
    // place the text reaches the picture.
    runtime::wmatmulN(tfQ_.data(), b.q2, tfNorm_.data(), tokens);
    runtime::wmatmulN(tfK_.data(), b.k2, context, contextTokens);
    runtime::wmatmulN(tfV_.data(), b.v2, context, contextTokens);
    attention(tfAttn_.data(), tfQ_.data(), tfK_.data(), tfV_.data(), tokens, contextTokens, heads,
              headDim);
    runtime::wmatmulNBias(tfProj_.data(), b.out2.w, tfAttn_.data(), tokens, b.out2.b);
    addInPlace(h, tfProj_.data(), hn);

    for (int i = 0; i < tokens; ++i) {
      ops::layernorm(tfNorm_.data() + static_cast<std::size_t>(i) * c,
                     h + static_cast<std::size_t>(i) * c, b.ln3.w.data(), b.ln3.b.data(), c);
    }
    // GEGLU: one projection to 2*inner, split into a value half and a gate half — value FIRST.
    // Swapping them is not a shape error and produces a plausible network.
    runtime::wmatmulNBias(tfFfn_.data(), b.geglu.w, tfNorm_.data(), tokens, b.geglu.b);
    for (int i = 0; i < tokens; ++i) {
      float* row = tfFfn_.data() + static_cast<std::size_t>(i) * 2 * inner;
      for (int j = 0; j < inner; ++j) row[j] *= ops::gelu(row[inner + j]);
    }
    // The value half is strided by 2*inner, so it is compacted into the front of the buffer
    // before the contraction rather than teaching the GEMV about a stride.
    for (int i = 1; i < tokens; ++i) {
      float* dst = tfFfn_.data() + static_cast<std::size_t>(i) * inner;
      const float* src = tfFfn_.data() + static_cast<std::size_t>(i) * 2 * inner;
      for (int j = 0; j < inner; ++j) dst[j] = src[j];
    }
    runtime::wmatmulNBias(tfProj_.data(), b.ffnOut.w, tfFfn_.data(), tokens, b.ffnOut.b);
    addInPlace(h, tfProj_.data(), hn);
  }

  if (!pointwise(tfOut_, tfIn_, t.projOut.w, t.projOut.b, error)) return false;
  x = std::move(tfOut_);
  addInPlace(x.data.data(), tfResidual_.data.data(), x.size());
  return true;
}

bool Unet::forward(const FeatureMap& latent, int timestep, const std::vector<float>& context,
                   int contextTokens, FeatureMap& out, std::string& error) {
  if (latent.c != cfg_.inChannels) {
    error = "unet takes " + std::to_string(cfg_.inChannels) + " latent channels, got " +
            std::to_string(latent.c);
    return false;
  }
  if (static_cast<int>(context.size()) != contextTokens * cfg_.crossDim) {
    error = "conditioning is " + std::to_string(context.size()) + " floats, expected " +
            std::to_string(contextTokens) + " x " + std::to_string(cfg_.crossDim);
    return false;
  }

  timestepEmbedding(timestep, cfg_.channels.front(), cfg_.freqShift, cfg_.flipSinToCos, temb_);
  emb_.resize(static_cast<std::size_t>(cfg_.timeEmbedDim));
  runtime::wmatmulBias(emb_.data(), w_->timeMlp0.w, temb_.data(), w_->timeMlp0.b);
  siluInPlace(emb_.data(), emb_.size());
  emb2_.resize(static_cast<std::size_t>(cfg_.timeEmbedDim));
  runtime::wmatmulBias(emb2_.data(), w_->timeMlp1.w, emb_.data(), w_->timeMlp1.b);

  FeatureMap x;
  if (!convForward(x, latent, w_->convIn, 3, 1, 1, block_.im2col, error)) return false;

  // The skip stack. conv_in's output is pushed BEFORE any block runs — that is the entry an
  // off-by-one loses.
  std::vector<FeatureMap> skips;
  skips.reserve(static_cast<std::size_t>(cfg_.blocks()) * (cfg_.layersPerBlock + 1) + 1);
  skips.push_back(x);

  const int n = cfg_.blocks();
  for (int i = 0; i < n; ++i) {
    const auto& blk = w_->down[static_cast<std::size_t>(i)];
    for (int j = 0; j < cfg_.layersPerBlock; ++j) {
      FeatureMap y;
      if (!resnetForward(blk.resnets[static_cast<std::size_t>(j)], x, &emb2_,
                         cfg_.normGroups, cfg_.normEpsilon, y, block_, error)) {
        return false;
      }
      x = std::move(y);
      if (!blk.attns.empty() &&
          !transformer(blk.attns[static_cast<std::size_t>(j)], x, context.data(), contextTokens,
                       cfg_.headCounts[static_cast<std::size_t>(i)], error)) {
        return false;
      }
      skips.push_back(x);
    }
    if (blk.downsample.valid()) {
      FeatureMap y;
      // Stride 2 with padding 1 — a learned downsample, not a pooling.
      if (!convForward(y, x, blk.downsample, 3, 2, 1, block_.im2col, error)) return false;
      x = std::move(y);
      skips.push_back(x);
    }
  }

  {
    FeatureMap y;
    if (!resnetForward(w_->midResnet0, x, &emb2_, cfg_.normGroups, cfg_.normEpsilon, y,
                       block_, error)) {
      return false;
    }
    x = std::move(y);
    if (!transformer(w_->midAttn, x, context.data(), contextTokens, cfg_.headCounts.back(), error)) {
      return false;
    }
    if (!resnetForward(w_->midResnet1, x, &emb2_, cfg_.normGroups, cfg_.normEpsilon, y,
                       block_, error)) {
      return false;
    }
    x = std::move(y);
  }

  for (int i = 0; i < n; ++i) {
    const auto& blk = w_->up[static_cast<std::size_t>(i)];
    const int layers = cfg_.layersPerBlock + 1;
    for (int j = 0; j < layers; ++j) {
      if (skips.empty()) {
        error = "unet skip stack ran dry at up block " + std::to_string(i);
        return false;
      }
      FeatureMap joined;
      if (!concatChannels(joined, x, skips.back(), error)) return false;
      skips.pop_back();
      FeatureMap y;
      if (!resnetForward(blk.resnets[static_cast<std::size_t>(j)], joined, &emb2_,
                         cfg_.normGroups, cfg_.normEpsilon, y, block_, error)) {
        return false;
      }
      x = std::move(y);
      if (!blk.attns.empty() &&
          !transformer(blk.attns[static_cast<std::size_t>(j)], x, context.data(), contextTokens,
                       cfg_.headCounts[static_cast<std::size_t>(n - 1 - i)], error)) {
        return false;
      }
    }
    if (blk.upsample.valid()) {
      FeatureMap up;
      upsampleNearest2x(up, x);
      if (!convForward(x, up, blk.upsample, 3, 1, 1, block_.im2col, error)) return false;
    }
  }

  if (!skips.empty()) {
    error = "unet finished with " + std::to_string(skips.size()) + " unused skip connections";
    return false;
  }

  groupNorm(x, w_->normOut.w, w_->normOut.b, cfg_.normGroups, cfg_.normEpsilon);
  siluInPlace(x.data.data(), x.size());
  return convForward(out, x, w_->convOut, 3, 1, 1, block_.im2col, error);
}

}  // namespace qorvix::image
