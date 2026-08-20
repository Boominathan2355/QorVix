#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "qorvix/runtime/weights.hpp"

// The primitives a convolutional diffusion network is built from, which the transformer runtime
// had no reason to own: 2-D convolution, GroupNorm, SiLU, nearest-neighbour upsampling, and
// attention over a variable-length pair of sequences.
//
// ACTIVATIONS ARE POSITION-MAJOR. A feature map is `[h*w, c]` — channel index fastest — not the
// `[c, h, w]` that PyTorch and every conv library use. That is the single decision this file is
// organised around, and it pays four times over:
//
//   * an im2col patch becomes k*k contiguous runs of `c` floats instead of c*k*k strided loads;
//   * the GEMM's natural output, `[positions, out_channels]`, IS the next layer's input, so no
//     transpose sits between convolutions;
//   * the spatial transformer's "flatten to a sequence of tokens" step is a no-op — a feature map
//     already IS `[tokens, channels]`, which is what attention wants;
//   * a per-channel bias, a GroupNorm scale and a resnet's timestep vector all broadcast along the
//     contiguous axis.
//
// The one operation it costs is GroupNorm's statistics, which need per-channel sums across
// positions. Those are accumulated into a `c`-wide array while reading the map front to back, so
// even that reads linearly.
namespace qorvix::image {

// A feature map, `[h*w, c]` row-major (see the header comment). `data[(y*w + x)*c + ch]`.
struct FeatureMap {
  int c = 0, h = 0, w = 0;
  std::vector<float> data;

  void resize(int channels, int height, int width) {
    c = channels;
    h = height;
    w = width;
    data.assign(size(), 0.0f);
  }
  std::size_t positions() const { return static_cast<std::size_t>(h) * w; }
  std::size_t size() const { return positions() * static_cast<std::size_t>(c); }
  float* at(int y, int x) { return data.data() + (static_cast<std::size_t>(y) * w + x) * c; }
  const float* at(int y, int x) const {
    return data.data() + (static_cast<std::size_t>(y) * w + x) * c;
  }
  bool matches(const FeatureMap& o) const { return c == o.c && h == o.h && w == o.w; }
};

// 2-D convolution as im2col + the runtime's batched GEMV.
//
// `weight` is [out_channels, k*k*in_channels] with the axis order the converter writes
// (kh, kw, in) — see scripts/convert_sd_to_gguf.py. Padding is zero padding, which is what every
// conv in this architecture uses.
//
// `scratch` is the caller's im2col buffer, kept across calls so a 50-step sample does not
// allocate 50 times. The patch matrix is built in row STRIPS rather than all at once: a 512x512
// VAE convolution over 128 channels would need 1.2 GiB of patches in one go, which is not a
// tuning knob but the difference between running and not.
bool conv2d(FeatureMap& out, const FeatureMap& in, const runtime::WeightMat& weight,
            const std::vector<float>& bias, int kernel, int stride, int pad,
            std::vector<float>& scratch, std::string& error);

// A 1x1 convolution, i.e. an independent [out, in] linear map at every position. Same thing as
// conv2d with kernel 1, without the im2col copy — which is the whole cost at kernel 1.
bool pointwise(FeatureMap& out, const FeatureMap& in, const runtime::WeightMat& weight,
               const std::vector<float>& bias, std::string& error);

// GroupNorm over a position-major map: channels are split into `groups` contiguous runs, and each
// run is normalized over (positions x channels-in-group) — NOT per channel and NOT per position.
void groupNorm(FeatureMap& x, const std::vector<float>& weight, const std::vector<float>& bias,
               int groups, float eps);

// SiLU / swish, in place. The activation of every resnet and of both output stems.
void siluInPlace(float* x, std::size_t n);

// out[i] += x[i].
void addInPlace(float* dst, const float* src, std::size_t n);

// Adds a per-channel vector to every position. Resnet timestep conditioning and conv biases.
void addPerChannel(FeatureMap& x, const float* perChannel);

// Nearest-neighbour 2x upsampling, matching torch's `interpolate(scale_factor=2, mode="nearest")`.
void upsampleNearest2x(FeatureMap& out, const FeatureMap& in);

// Concatenates `b`'s channels after `a`'s at every position — the UNet's skip connection. `a` and
// `b` must have the same spatial extent.
bool concatChannels(FeatureMap& out, const FeatureMap& a, const FeatureMap& b, std::string& error);

// Scaled dot-product attention with `heads` heads over a query sequence and a key/value sequence
// that may be a different length (self-attention passes the same length twice; cross-attention
// does not).
//
// q is [nq, heads*headDim], k and v are [nkv, heads*headDim], out is [nq, heads*headDim]. The
// scale is headDim^-0.5.
//
// `causal` masks every key after the query's own position, which is what the CLIP text encoder
// needs and what none of the image-side attention does. It is a flag rather than a second
// function because the two differ by one comparison in the innermost loop.
void attention(float* out, const float* q, const float* k, const float* v, int nq, int nkv,
               int heads, int headDim, bool causal = false);

}  // namespace qorvix::image
