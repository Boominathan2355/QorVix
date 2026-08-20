#pragma once

#include <string>
#include <vector>

#include "qorvix/image/blocks.hpp"
#include "qorvix/image/nn.hpp"
#include "qorvix/image/sd_weights.hpp"

namespace qorvix::image {

// The denoising UNet: the model that is evaluated once per sampling step and accounts for
// essentially all of the time a picture takes.
//
// SHAPE OF THE THING. A stack of residual blocks that halve the resolution and double the
// channels, a bottleneck, then a mirror stack that upsamples — with every down-block output saved
// and concatenated into the matching up block. Cross-attention to the text conditioning is
// interleaved at the resolutions the architecture asks for (`transformer_depth`), and the
// timestep enters every residual block as a learned per-channel vector rather than as an input
// channel.
//
// TWO THINGS HERE ARE EASY TO GET WRONG AND SILENT WHEN WRONG:
//
//   * The skip stack. Down blocks push one activation per resnet PLUS one per downsampler, and
//     the first thing pushed is the output of conv_in — so the stack is one deeper than the
//     resnet count suggests, and an up block pops `layers_per_block + 1` of them. Off by one and
//     every skip connects to the wrong resolution's neighbour; the shapes still line up for most
//     of them, and the image is merely wrong.
//   * The two GroupNorm epsilons. Residual blocks use the config's `norm_eps` (1e-5); the spatial
//     transformer's own norm is fixed at 1e-6 in diffusers and is not in any config file.
//
// Not thread-safe: every scratch buffer is a member so a 50-step sample allocates once.
class Unet {
 public:
  Unet(const SdUnetConfig& cfg, const UnetWeights& weights);

  // One denoising step's forward pass. `latent` is [inChannels, h, w]; `context` is the text
  // encoder's [contextTokens, crossDim] output; `out` is the model's prediction, the same shape
  // as the latent — an epsilon or a v depending on the checkpoint, which is the scheduler's
  // business rather than this class's.
  bool forward(const FeatureMap& latent, int timestep, const std::vector<float>& context,
               int contextTokens, FeatureMap& out, std::string& error);

  // The sinusoidal timestep embedding, before the two-layer MLP. Exposed because it is a pure
  // function of the config and therefore the one part of this network a test can pin exactly
  // without any weights at all.
  static void timestepEmbedding(int timestep, int dim, float freqShift, bool flipSinToCos,
                                std::vector<float>& out);

 private:
  // In place: a spatial transformer's output has the same shape as its input, and its residual is
  // added at the end, so there is nothing to be gained by handing back a second buffer.
  bool transformer(const SpatialTransformerWeights& t, FeatureMap& x, const float* context,
                   int contextTokens, int heads, std::string& error);

  SdUnetConfig cfg_;
  const UnetWeights* w_;

  BlockScratch block_;
  std::vector<float> temb_, emb_, emb2_;
  // Transformer scratch. Sized by the largest resolution any attention block runs at, which is
  // reached on the first step and never grows again.
  std::vector<float> tfNorm_, tfQ_, tfK_, tfV_, tfAttn_, tfProj_, tfFfn_;
  FeatureMap tfResidual_, tfIn_, tfOut_;
};

}  // namespace qorvix::image
