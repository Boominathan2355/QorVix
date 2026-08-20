#pragma once

#include <string>
#include <vector>

#include "qorvix/image/nn.hpp"
#include "qorvix/image/sd_weights.hpp"

namespace qorvix::image {

// The residual block, shared by the UNet and the VAE decoder.
//
// It lives here rather than as a private method on either because they run the SAME block with
// two differences, both of which are parameters: the VAE has no timestep to condition on, and it
// normalizes at 1e-6 where the UNet uses 1e-5. Writing it twice would have meant two places to
// keep the ordering right — norm, activate, convolve, ADD THE TIMESTEP, norm, activate, convolve —
// and the second copy is exactly where a reordering survives review.
struct BlockScratch {
  std::vector<float> im2col;
  std::vector<float> tembSilu, tproj;
  FeatureMap in, tmp, skip;
};

// A convolution or a 1x1 projection, dispatching on the kernel so a 1x1 skips im2col entirely.
bool convForward(FeatureMap& out, const FeatureMap& in, const MatWeights& w, int kernel, int stride,
                 int pad, std::vector<float>& im2col, std::string& error);

// `timeEmb` is null for the VAE. When present it is the UNet's [timeEmbedDim] vector BEFORE the
// block's own SiLU and projection — the activation comes first, which is easy to invert and
// impossible to see afterwards.
bool resnetForward(const ResnetWeights& r, const FeatureMap& in, const std::vector<float>* timeEmb,
                   int groups, float eps, FeatureMap& out, BlockScratch& s, std::string& error);

}  // namespace qorvix::image
