#include "qorvix/image/vae.hpp"

#include <algorithm>
#include <cmath>

namespace qorvix::image {

VaeDecoder::VaeDecoder(const SdConfig& cfg, const VaeDecoderWeights& weights)
    : cfg_(cfg), w_(&weights) {}

bool VaeDecoder::decode(const FeatureMap& latent, FeatureMap& out, std::string& error) {
  if (latent.c != cfg_.latentChannels) {
    error = "vae takes " + std::to_string(cfg_.latentChannels) + " latent channels, got " +
            std::to_string(latent.c);
    return false;
  }
  const int groups = cfg_.vae.normGroups;
  const float eps = cfg_.vae.normEpsilon;

  // Undo the scaling the encoder applied so its latents had roughly unit variance. It is a
  // property of the VAE, which is why it travels in the file next to these weights.
  scaled_ = latent;
  const float inv = cfg_.scaleFactor != 0.0f ? 1.0f / cfg_.scaleFactor : 1.0f;
  for (float& v : scaled_.data) v *= inv;

  FeatureMap x;
  if (w_->postQuantConv.valid()) {
    if (!pointwise(x, scaled_, w_->postQuantConv.w, w_->postQuantConv.b, error)) return false;
  } else {
    x = scaled_;
  }

  FeatureMap y;
  if (!convForward(y, x, w_->convIn, 3, 1, 1, block_.im2col, error)) return false;
  x = std::move(y);
  if (!resnetForward(w_->midResnet0, x, nullptr, groups, eps, y, block_, error)) return false;
  x = std::move(y);

  // The one attention block, over every spatial position at the coarsest resolution. ONE head of
  // full width — not the multi-head arrangement the UNet's transformers use — and unlike those,
  // its q/k/v projections carry biases.
  {
    const int c = x.c;
    const int tokens = static_cast<int>(x.positions());
    attnResidual_ = x;
    groupNorm(x, w_->midAttn.norm.w, w_->midAttn.norm.b, groups, eps);
    attnQ_.resize(x.size());
    attnK_.resize(x.size());
    attnV_.resize(x.size());
    attnOut_.resize(x.size());
    runtime::wmatmulNBias(attnQ_.data(), w_->midAttn.q.w, x.data.data(), tokens, w_->midAttn.q.b);
    runtime::wmatmulNBias(attnK_.data(), w_->midAttn.k.w, x.data.data(), tokens, w_->midAttn.k.b);
    runtime::wmatmulNBias(attnV_.data(), w_->midAttn.v.w, x.data.data(), tokens, w_->midAttn.v.b);
    attention(attnOut_.data(), attnQ_.data(), attnK_.data(), attnV_.data(), tokens, tokens,
              /*heads=*/1, c);
    runtime::wmatmulNBias(x.data.data(), w_->midAttn.out.w, attnOut_.data(), tokens,
                          w_->midAttn.out.b);
    // The residual is the map BEFORE the GroupNorm, which is what makes this block a refinement
    // rather than a replacement.
    addInPlace(x.data.data(), attnResidual_.data.data(), x.size());
  }

  if (!resnetForward(w_->midResnet1, x, nullptr, groups, eps, y, block_, error)) return false;
  x = std::move(y);

  for (const auto& blk : w_->up) {
    for (const auto& r : blk.resnets) {
      if (!resnetForward(r, x, nullptr, groups, eps, y, block_, error)) return false;
      x = std::move(y);
    }
    if (blk.upsample.valid()) {
      FeatureMap up;
      upsampleNearest2x(up, x);
      if (!convForward(x, up, blk.upsample, 3, 1, 1, block_.im2col, error)) return false;
    }
  }

  groupNorm(x, w_->normOut.w, w_->normOut.b, groups, eps);
  siluInPlace(x.data.data(), x.size());
  return convForward(out, x, w_->convOut, 3, 1, 1, block_.im2col, error);
}

bool VaeDecoder::decodeToImage(const FeatureMap& latent, vision::Image& out, std::string& error) {
  FeatureMap rgb;
  if (!decode(latent, rgb, error)) return false;
  if (rgb.c != 3) {
    error = "vae produced " + std::to_string(rgb.c) + " channels, expected 3";
    return false;
  }
  out.width = rgb.w;
  out.height = rgb.h;
  out.rgb.resize(static_cast<std::size_t>(rgb.w) * rgb.h * 3);
  // (x / 2 + 0.5), clamped, then 8 bits. The clamp is load-bearing: the decoder routinely
  // overshoots slightly at high-contrast edges, and wrapping instead of clamping turns a bright
  // highlight into a black speck.
  for (std::size_t i = 0; i < out.rgb.size(); ++i) {
    const float v = std::clamp(rgb.data[i] * 0.5f + 0.5f, 0.0f, 1.0f);
    out.rgb[i] = static_cast<std::uint8_t>(std::lround(v * 255.0f));
  }
  return true;
}

}  // namespace qorvix::image
