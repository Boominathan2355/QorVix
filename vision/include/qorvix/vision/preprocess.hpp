#pragma once

#include <array>
#include <string>
#include <vector>

#include "qorvix/vision/image.hpp"

namespace qorvix::vision {

// CLIP's image preprocessing, matching HuggingFace's CLIPImageProcessor.
//
// This is the highest-risk part of the vision path and the least obvious: the transformer weights
// are exact, but if the resize filter, the crop origin or the normalization constants differ by a
// little, every output vector differs by a lot — and nothing errors. The pipeline is:
//
//   1. resize the SHORTEST side to `size` with bicubic interpolation, preserving aspect ratio
//   2. centre-crop to size x size
//   3. rescale to [0,1] (divide by 255)
//   4. normalize per channel with the model's own mean/std (read from the GGUF, not hardcoded)
//
// PIL's bicubic is reproduced rather than approximated: a=-0.5, a support of 2 scaled by the
// downsampling factor, and per-output-pixel filter normalization. A "simple" bicubic that samples
// 4x4 neighbours without scaling the support aliases badly when shrinking a large photo to 336px,
// which is exactly the common case.
struct PreprocessConfig {
  int size = 336;                                       // target square edge
  std::array<float, 3> mean{0.48145466f, 0.4578275f, 0.40821073f};
  std::array<float, 3> std{0.26862954f, 0.26130258f, 0.27577711f};
};

// Returns CHW float data: [3, size, size], channel-planar, which is the layout the patch
// embedding consumes.
bool preprocessClip(const Image& img, const PreprocessConfig& cfg, std::vector<float>& out,
                    std::string& error);

// Bicubic resize to an explicit target, exposed for tests. `out` is HWC RGB u8.
bool resizeBicubic(const Image& src, int dstW, int dstH, Image& out, std::string& error);

}  // namespace qorvix::vision
