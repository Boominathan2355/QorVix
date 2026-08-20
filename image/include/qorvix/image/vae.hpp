#pragma once

#include <string>
#include <vector>

#include "qorvix/image/blocks.hpp"
#include "qorvix/image/nn.hpp"
#include "qorvix/image/sd_weights.hpp"
#include "qorvix/vision/image.hpp"

namespace qorvix::image {

// The VAE decoder: the last stage, which turns a 64x64x4 latent into a 512x512x3 picture.
//
// It is cheap in parameters and expensive in time, and those two facts pull in opposite
// directions when reading a profile. The UNet holds 860 M weights and runs at 64x64; the decoder
// holds 50 M and runs at up to 512x512, so its convolutions move far more data per weight. On this
// CPU it is a meaningful fraction of a whole sample, and all of it lands in one call at the end.
//
// Architecturally it is the UNet's right half with the skip connections deleted: a bottleneck with
// one self-attention block, then blocks that upsample and shed channels. Its residual blocks are
// the same `resnetForward` the UNet runs, with no timestep and a 1e-6 epsilon.
//
// Not thread-safe; scratch is in members.
class VaeDecoder {
 public:
  VaeDecoder(const SdConfig& cfg, const VaeDecoderWeights& weights);

  // `latent` is the sampler's output, still in latent units — this applies the model's own
  // `scale_factor` before decoding, because that constant belongs to the VAE and not to the
  // sampling loop that produced the latent.
  //
  // `out` is [3, h, w] in the decoder's native range, roughly [-1, 1].
  bool decode(const FeatureMap& latent, FeatureMap& out, std::string& error);

  // decode() followed by the standard `(x / 2 + 0.5)` remap, clamped and quantized to 8 bits.
  bool decodeToImage(const FeatureMap& latent, vision::Image& out, std::string& error);

 private:
  SdConfig cfg_;
  const VaeDecoderWeights* w_;
  BlockScratch block_;
  std::vector<float> attnQ_, attnK_, attnV_, attnOut_;
  FeatureMap scaled_, attnResidual_;
};

}  // namespace qorvix::image
