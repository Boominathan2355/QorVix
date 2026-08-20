#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "qorvix/image/blocks.hpp"
#include "qorvix/image/clip_tokenizer.hpp"
#include "qorvix/image/nn.hpp"
#include "qorvix/image/rng.hpp"
#include "qorvix/image/scheduler.hpp"
#include "qorvix/image/sd_weights.hpp"
#include "qorvix/image/unet.hpp"
#include "qorvix/vision/image.hpp"

using namespace qorvix::image;
using Catch::Matchers::WithinAbs;

namespace {

// A direct, obvious convolution — quadruple loop, no im2col, no GEMV — so the fast path has an
// independent implementation to be wrong against rather than only itself.
std::vector<float> directConv(const FeatureMap& in, const std::vector<float>& w, int outC,
                              int kernel, int stride, int pad) {
  const int outH = (in.h + 2 * pad - kernel) / stride + 1;
  const int outW = (in.w + 2 * pad - kernel) / stride + 1;
  std::vector<float> out(static_cast<std::size_t>(outH) * outW * outC, 0.0f);
  for (int oy = 0; oy < outH; ++oy) {
    for (int ox = 0; ox < outW; ++ox) {
      for (int oc = 0; oc < outC; ++oc) {
        float acc = 0.0f;
        for (int ky = 0; ky < kernel; ++ky) {
          for (int kx = 0; kx < kernel; ++kx) {
            const int iy = oy * stride - pad + ky;
            const int ix = ox * stride - pad + kx;
            if (iy < 0 || iy >= in.h || ix < 0 || ix >= in.w) continue;
            for (int ic = 0; ic < in.c; ++ic) {
              // Weights are [outC, kh, kw, inC] — the axis order the converter writes.
              const std::size_t wi =
                  ((static_cast<std::size_t>(oc) * kernel + ky) * kernel + kx) * in.c + ic;
              acc += w[wi] * in.at(iy, ix)[ic];
            }
          }
        }
        out[(static_cast<std::size_t>(oy) * outW + ox) * outC + oc] = acc;
      }
    }
  }
  return out;
}

FeatureMap ramp(int c, int h, int w, float scale = 0.1f) {
  FeatureMap m;
  m.resize(c, h, w);
  for (std::size_t i = 0; i < m.size(); ++i) {
    m.data[i] = std::sin(static_cast<float>(i) * 0.37f) * scale;
  }
  return m;
}

std::vector<float> weightRamp(std::size_t n) {
  std::vector<float> w(n);
  for (std::size_t i = 0; i < n; ++i) w[i] = std::cos(static_cast<float>(i) * 0.21f) * 0.3f;
  return w;
}

MatWeights mat(int rows, int cols, float value) {
  MatWeights m;
  m.w = qorvix::runtime::WeightMat::f32(std::vector<float>(
                                            static_cast<std::size_t>(rows) * cols, value),
                                        rows, cols);
  m.b.assign(static_cast<std::size_t>(rows), 0.0f);
  return m;
}

NormWeights unitNorm(int n) {
  NormWeights nw;
  nw.w.assign(static_cast<std::size_t>(n), 1.0f);
  nw.b.assign(static_cast<std::size_t>(n), 0.0f);
  return nw;
}

ResnetWeights resnetOf(int inC, int outC, int tembDim) {
  ResnetWeights r;
  r.norm1 = unitNorm(inC);
  r.conv1 = mat(outC, 9 * inC, 0.01f);
  if (tembDim > 0) r.timeEmb = mat(outC, tembDim, 0.001f);
  r.norm2 = unitNorm(outC);
  r.conv2 = mat(outC, 9 * outC, 0.01f);
  if (inC != outC) r.skip = mat(outC, inC, 0.5f);
  return r;
}

// A UNet with no attention anywhere, so what it exercises is the part that has no numerical
// reference to check it: the skip stack's depth, and the channel arithmetic the up blocks derive
// from the down blocks.
SdUnetConfig tinyUnetConfig(std::vector<int> channels, int layersPerBlock) {
  SdUnetConfig cfg;
  cfg.inChannels = 4;
  cfg.outChannels = 4;
  cfg.channels = std::move(channels);
  cfg.headCounts.assign(cfg.channels.size(), 1);
  cfg.transformerDepth.assign(cfg.channels.size(), 0);
  cfg.layersPerBlock = layersPerBlock;
  cfg.crossDim = 8;
  cfg.normGroups = 1;
  cfg.timeEmbedDim = cfg.channels.front() * 4;
  return cfg;
}

UnetWeights tinyUnetWeights(const SdUnetConfig& cfg) {
  UnetWeights w;
  const int ch0 = cfg.channels.front();
  w.convIn = mat(ch0, 9 * cfg.inChannels, 0.02f);
  w.timeMlp0 = mat(cfg.timeEmbedDim, ch0, 0.01f);
  w.timeMlp1 = mat(cfg.timeEmbedDim, cfg.timeEmbedDim, 0.01f);

  const int n = cfg.blocks();
  w.down.resize(static_cast<std::size_t>(n));
  int prev = ch0;
  for (int i = 0; i < n; ++i) {
    const int outC = cfg.channels[static_cast<std::size_t>(i)];
    for (int j = 0; j < cfg.layersPerBlock; ++j) {
      w.down[static_cast<std::size_t>(i)].resnets.push_back(
          resnetOf(j == 0 ? prev : outC, outC, cfg.timeEmbedDim));
    }
    if (i < n - 1) w.down[static_cast<std::size_t>(i)].downsample = mat(outC, 9 * outC, 0.02f);
    prev = outC;
  }

  const int mid = cfg.channels.back();
  w.midResnet0 = resnetOf(mid, mid, cfg.timeEmbedDim);
  w.midResnet1 = resnetOf(mid, mid, cfg.timeEmbedDim);
  // The mid block's transformer runs unconditionally, so it needs its projections even though this
  // config has no transformer BLOCKS in it. `projOut` is zeroed, which makes the whole thing the
  // identity on its residual — deliberately, so this test measures the skip stack and nothing else.
  w.midAttn.norm = unitNorm(mid);
  w.midAttn.projIn = mat(mid, mid, 1.0f);
  w.midAttn.projOut = mat(mid, mid, 0.0f);

  std::vector<int> rev(cfg.channels.rbegin(), cfg.channels.rend());
  w.up.resize(static_cast<std::size_t>(n));
  int prevOut = rev.front();
  for (int i = 0; i < n; ++i) {
    const int outC = rev[static_cast<std::size_t>(i)];
    const int inC = rev[static_cast<std::size_t>(std::min(i + 1, n - 1))];
    const int layers = cfg.layersPerBlock + 1;
    for (int j = 0; j < layers; ++j) {
      const int skipC = (j == layers - 1) ? inC : outC;
      const int resIn = (j == 0 ? prevOut : outC) + skipC;
      w.up[static_cast<std::size_t>(i)].resnets.push_back(resnetOf(resIn, outC, cfg.timeEmbedDim));
    }
    if (i < n - 1) w.up[static_cast<std::size_t>(i)].upsample = mat(outC, 9 * outC, 0.02f);
    prevOut = outC;
  }

  w.normOut = unitNorm(ch0);
  w.convOut = mat(cfg.outChannels, 9 * ch0, 0.02f);
  return w;
}

}  // namespace

// ---- PNG writing ------------------------------------------------------------------------------

TEST_CASE("a written PNG round-trips through this repo's own decoder", "[image]") {
  // The encoder and the decoder are the two halves of one format, and this is the only test that
  // can catch a disagreement between them — a wrong CRC, a wrong zlib header or a wrong stored
  // block length all produce a file that only a real decoder rejects.
  qorvix::vision::Image src;
  src.width = 37;  // deliberately not a multiple of anything
  src.height = 11;
  src.rgb.resize(static_cast<std::size_t>(37) * 11 * 3);
  for (std::size_t i = 0; i < src.rgb.size(); ++i) {
    src.rgb[i] = static_cast<std::uint8_t>((i * 7 + i / 13) & 0xFF);
  }

  std::vector<std::uint8_t> png;
  std::string err;
  REQUIRE(qorvix::vision::encodePng(src, png, err));
  REQUIRE(png.size() > 8);
  REQUIRE(png[0] == 0x89);
  REQUIRE(png[1] == 'P');

  qorvix::vision::Image back;
  REQUIRE(qorvix::vision::decodePng(png.data(), png.size(), back, err));
  REQUIRE(back.width == src.width);
  REQUIRE(back.height == src.height);
  REQUIRE(back.rgb == src.rgb);
}

TEST_CASE("a PNG larger than one stored block still round-trips", "[image]") {
  // DEFLATE stored blocks cap at 65535 bytes, so anything past that exercises the multi-block
  // path and the BFINAL flag landing on the LAST block rather than the first.
  qorvix::vision::Image src;
  src.width = 200;
  src.height = 200;
  src.rgb.resize(static_cast<std::size_t>(200) * 200 * 3);
  for (std::size_t i = 0; i < src.rgb.size(); ++i) src.rgb[i] = static_cast<std::uint8_t>(i * 31);

  std::vector<std::uint8_t> png;
  std::string err;
  REQUIRE(qorvix::vision::encodePng(src, png, err));
  REQUIRE(png.size() > 65535);

  qorvix::vision::Image back;
  REQUIRE(qorvix::vision::decodePng(png.data(), png.size(), back, err));
  REQUIRE(back.rgb == src.rgb);
}

TEST_CASE("encoding refuses an image whose buffer disagrees with its dimensions", "[image]") {
  qorvix::vision::Image bad;
  bad.width = 4;
  bad.height = 4;
  bad.rgb.resize(10);  // not 4*4*3
  std::vector<std::uint8_t> png;
  std::string err;
  REQUIRE_FALSE(qorvix::vision::encodePng(bad, png, err));
  REQUIRE_FALSE(err.empty());
}

// ---- convolution ------------------------------------------------------------------------------

TEST_CASE("conv2d matches a direct convolution", "[image]") {
  const FeatureMap in = ramp(3, 5, 7);
  const int outC = 4, K = 3;
  const std::vector<float> w = weightRamp(static_cast<std::size_t>(outC) * K * K * in.c);
  const std::vector<float> bias(static_cast<std::size_t>(outC), 0.0f);

  FeatureMap out;
  std::vector<float> scratch;
  std::string err;
  REQUIRE(conv2d(out, in, qorvix::runtime::WeightMat::f32(w, outC, K * K * in.c), bias, K, 1, 1,
                 scratch, err));
  REQUIRE(out.c == outC);
  REQUIRE(out.h == 5);
  REQUIRE(out.w == 7);

  const std::vector<float> want = directConv(in, w, outC, K, 1, 1);
  REQUIRE(want.size() == out.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    REQUIRE_THAT(out.data[i], WithinAbs(want[i], 1e-5));
  }
}

TEST_CASE("conv2d halves the resolution at stride 2", "[image]") {
  const FeatureMap in = ramp(2, 8, 8);
  const int outC = 3, K = 3;
  const std::vector<float> w = weightRamp(static_cast<std::size_t>(outC) * K * K * in.c);
  const std::vector<float> bias(static_cast<std::size_t>(outC), 0.0f);
  FeatureMap out;
  std::vector<float> scratch;
  std::string err;
  REQUIRE(conv2d(out, in, qorvix::runtime::WeightMat::f32(w, outC, K * K * in.c), bias, K, 2, 1,
                 scratch, err));
  REQUIRE(out.h == 4);
  REQUIRE(out.w == 4);
  const std::vector<float> want = directConv(in, w, outC, K, 2, 1);
  for (std::size_t i = 0; i < want.size(); ++i) REQUIRE_THAT(out.data[i], WithinAbs(want[i], 1e-5));
}

TEST_CASE("a 1x1 convolution and pointwise agree", "[image]") {
  // pointwise() exists only to skip the im2col copy, so the two must be interchangeable — this is
  // what stops the shortcut from drifting away from the thing it is a shortcut for.
  const FeatureMap in = ramp(6, 3, 4);
  const int outC = 5;
  const std::vector<float> w = weightRamp(static_cast<std::size_t>(outC) * in.c);
  std::vector<float> bias(static_cast<std::size_t>(outC));
  for (int i = 0; i < outC; ++i) bias[static_cast<std::size_t>(i)] = 0.1f * i;
  const auto wm = qorvix::runtime::WeightMat::f32(w, outC, in.c);

  FeatureMap viaConv, viaPointwise;
  std::vector<float> scratch;
  std::string err;
  REQUIRE(conv2d(viaConv, in, wm, bias, 1, 1, 0, scratch, err));
  REQUIRE(pointwise(viaPointwise, in, wm, bias, err));
  REQUIRE(viaConv.size() == viaPointwise.size());
  for (std::size_t i = 0; i < viaConv.size(); ++i) {
    REQUIRE_THAT(viaConv.data[i], WithinAbs(viaPointwise.data[i], 1e-6));
  }
}

TEST_CASE("conv2d refuses a weight whose input width is not the patch size", "[image]") {
  const FeatureMap in = ramp(3, 4, 4);
  FeatureMap out;
  std::vector<float> scratch;
  std::string err;
  // 3x3 over 3 channels is 27 inputs per position, not 20.
  REQUIRE_FALSE(conv2d(out, in, qorvix::runtime::WeightMat::f32(std::vector<float>(40, 0.0f), 2, 20),
                       {}, 3, 1, 1, scratch, err));
  REQUIRE(err.find("27") != std::string::npos);
}

// ---- GroupNorm --------------------------------------------------------------------------------

TEST_CASE("groupNorm normalizes each group over positions AND its own channels", "[image]") {
  FeatureMap x = ramp(4, 3, 5, 2.0f);
  const int groups = 2;
  groupNorm(x, {}, {}, groups, 1e-5f);

  const int perGroup = x.c / groups;
  for (int g = 0; g < groups; ++g) {
    double sum = 0.0, sumSq = 0.0;
    std::size_t count = 0;
    for (std::size_t p = 0; p < x.positions(); ++p) {
      for (int i = g * perGroup; i < (g + 1) * perGroup; ++i) {
        const double v = x.data[p * x.c + i];
        sum += v;
        sumSq += v * v;
        ++count;
      }
    }
    const double mean = sum / count;
    REQUIRE_THAT(mean, WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(sumSq / count - mean * mean, WithinAbs(1.0, 1e-3));
  }
}

TEST_CASE("groupNorm applies its per-channel scale and shift", "[image]") {
  FeatureMap plain = ramp(2, 2, 2, 1.5f);
  FeatureMap affine = plain;
  groupNorm(plain, {}, {}, 1, 1e-5f);
  groupNorm(affine, {2.0f, 3.0f}, {-1.0f, 0.5f}, 1, 1e-5f);
  for (std::size_t p = 0; p < plain.positions(); ++p) {
    REQUIRE_THAT(affine.data[p * 2 + 0], WithinAbs(plain.data[p * 2 + 0] * 2.0f - 1.0f, 1e-4));
    REQUIRE_THAT(affine.data[p * 2 + 1], WithinAbs(plain.data[p * 2 + 1] * 3.0f + 0.5f, 1e-4));
  }
}

// ---- resampling and joining ---------------------------------------------------------------------

TEST_CASE("nearest upsampling replicates each cell into a 2x2 block", "[image]") {
  FeatureMap in;
  in.resize(2, 2, 3);
  for (std::size_t i = 0; i < in.size(); ++i) in.data[i] = static_cast<float>(i);
  FeatureMap out;
  upsampleNearest2x(out, in);
  REQUIRE(out.h == 4);
  REQUIRE(out.w == 6);
  for (int y = 0; y < in.h; ++y) {
    for (int x = 0; x < in.w; ++x) {
      for (int c = 0; c < in.c; ++c) {
        const float v = in.at(y, x)[c];
        REQUIRE(out.at(2 * y, 2 * x)[c] == v);
        REQUIRE(out.at(2 * y, 2 * x + 1)[c] == v);
        REQUIRE(out.at(2 * y + 1, 2 * x)[c] == v);
        REQUIRE(out.at(2 * y + 1, 2 * x + 1)[c] == v);
      }
    }
  }
}

TEST_CASE("skip concatenation puts the incoming map's channels first", "[image]") {
  // The order is not arbitrary: the up block's resnet weights were trained expecting
  // [hidden, skip], and swapping them is not a shape error.
  FeatureMap a, b, out;
  a.resize(2, 1, 2);
  b.resize(3, 1, 2);
  for (std::size_t i = 0; i < a.size(); ++i) a.data[i] = 10.0f + i;
  for (std::size_t i = 0; i < b.size(); ++i) b.data[i] = 100.0f + i;
  std::string err;
  REQUIRE(concatChannels(out, a, b, err));
  REQUIRE(out.c == 5);
  REQUIRE(out.at(0, 0)[0] == 10.0f);
  REQUIRE(out.at(0, 0)[1] == 11.0f);
  REQUIRE(out.at(0, 0)[2] == 100.0f);
  REQUIRE(out.at(0, 1)[0] == 12.0f);
  REQUIRE(out.at(0, 1)[2] == 103.0f);
}

TEST_CASE("skip concatenation refuses maps of different extents", "[image]") {
  FeatureMap a, b, out;
  a.resize(2, 4, 4);
  b.resize(2, 2, 2);
  std::string err;
  REQUIRE_FALSE(concatChannels(out, a, b, err));
  REQUIRE(err.find("4x4") != std::string::npos);
}

// ---- attention --------------------------------------------------------------------------------

TEST_CASE("causal attention leaves the first position looking only at itself", "[image]") {
  const int heads = 2, hd = 3, n = 4, dim = heads * hd;
  std::vector<float> q(static_cast<std::size_t>(n) * dim), k = q, v = q, out = q;
  for (std::size_t i = 0; i < q.size(); ++i) {
    q[i] = std::sin(static_cast<float>(i));
    k[i] = std::cos(static_cast<float>(i));
    v[i] = static_cast<float>(i);
  }
  attention(out.data(), q.data(), k.data(), v.data(), n, n, heads, hd, /*causal=*/true);
  // Softmax over one key is 1, so the output must be that key's value verbatim.
  for (int i = 0; i < dim; ++i) REQUIRE_THAT(out[static_cast<std::size_t>(i)], WithinAbs(v[i], 1e-5));
}

TEST_CASE("attention over identical keys averages the values", "[image]") {
  const int heads = 1, hd = 2, nq = 1, nkv = 3, dim = 2;
  const std::vector<float> q{1.0f, 0.0f};
  const std::vector<float> k{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // all keys equal -> uniform
  const std::vector<float> v{1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f};
  std::vector<float> out(dim);
  attention(out.data(), q.data(), k.data(), v.data(), nq, nkv, heads, hd);
  REQUIRE_THAT(out[0], WithinAbs(2.0, 1e-5));
  REQUIRE_THAT(out[1], WithinAbs(20.0, 1e-5));
}

// ---- timestep embedding -----------------------------------------------------------------------

TEST_CASE("the timestep embedding puts cosines first when flip_sin_to_cos is set", "[image]") {
  std::vector<float> flipped, plain;
  Unet::timestepEmbedding(0, 8, 0.0f, /*flipSinToCos=*/true, flipped);
  Unet::timestepEmbedding(0, 8, 0.0f, /*flipSinToCos=*/false, plain);
  REQUIRE(flipped.size() == 8);
  // At t = 0 every argument is 0, so the cosine half is all ones and the sine half all zeros.
  for (int i = 0; i < 4; ++i) {
    REQUIRE_THAT(flipped[static_cast<std::size_t>(i)], WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(flipped[static_cast<std::size_t>(4 + i)], WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(plain[static_cast<std::size_t>(i)], WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(plain[static_cast<std::size_t>(4 + i)], WithinAbs(1.0, 1e-6));
  }
}

TEST_CASE("the timestep embedding's frequencies fall geometrically", "[image]") {
  std::vector<float> e;
  Unet::timestepEmbedding(1, 8, 0.0f, /*flipSinToCos=*/false, e);
  // sin(t * exp(-ln(10000) * i / half)) with t = 1, half = 4.
  for (int i = 0; i < 4; ++i) {
    const double want = std::sin(std::exp(-std::log(10000.0) * i / 4.0));
    REQUIRE_THAT(e[static_cast<std::size_t>(i)], WithinAbs(want, 1e-6));
  }
}

// ---- the sampler ------------------------------------------------------------------------------

namespace {

SdSchedulerConfig sdSched() {
  SdSchedulerConfig c;
  c.trainTimesteps = 1000;
  c.betaStart = 0.00085f;
  c.betaEnd = 0.012f;
  c.betaSchedule = "scaled_linear";
  c.timestepSpacing = "leading";
  c.stepsOffset = 1;
  return c;
}

}  // namespace

TEST_CASE("leading spacing walks the training grid downward from steps_offset", "[image]") {
  std::string err;
  auto s = Scheduler::make(sdSched(), SamplerKind::Ddim, 20, err);
  REQUIRE(s);
  REQUIRE(s->steps() == 20);
  REQUIRE(s->timesteps().front() == 951);  // 19 * 50 + 1
  REQUIRE(s->timesteps().back() == 1);
  for (std::size_t i = 1; i < s->timesteps().size(); ++i) {
    REQUIRE(s->timesteps()[i] < s->timesteps()[i - 1]);
  }
}

TEST_CASE("trailing spacing starts at the last training step", "[image]") {
  // The distilled few-step checkpoints ship with this spacing, and starting at 951 instead of 999
  // under-denoises the first step in a way nothing reports.
  SdSchedulerConfig cfg = sdSched();
  cfg.timestepSpacing = "trailing";
  std::string err;
  auto s = Scheduler::make(cfg, SamplerKind::Ddim, 4, err);
  REQUIRE(s);
  REQUIRE(s->timesteps().front() == 999);
}

TEST_CASE("alpha_bar falls monotonically from nearly one to nearly zero", "[image]") {
  std::string err;
  auto s = Scheduler::make(sdSched(), SamplerKind::Ddim, 10, err);
  REQUIRE(s);
  const auto& a = s->alphasCumprod();
  REQUIRE(a.size() == 1000);
  REQUIRE(a.front() > 0.999f);
  REQUIRE(a.back() < 0.01f);
  for (std::size_t i = 1; i < a.size(); ++i) REQUIRE(a[i] < a[i - 1]);
}

TEST_CASE("scaled_linear and linear are different beta ladders", "[image]") {
  SdSchedulerConfig lin = sdSched();
  lin.betaSchedule = "linear";
  std::string err;
  auto a = Scheduler::make(sdSched(), SamplerKind::Ddim, 10, err);
  auto b = Scheduler::make(lin, SamplerKind::Ddim, 10, err);
  REQUIRE(a);
  REQUIRE(b);
  // Same endpoints, different curve — so the middle of the ladder must disagree.
  REQUIRE(std::abs(a->alphasCumprod()[500] - b->alphasCumprod()[500]) > 0.05f);
}

TEST_CASE("a DDIM step with a perfectly predicted noise recovers the clean latent", "[image]") {
  // If the model's epsilon IS the noise that was added, one step of DDIM to t=0 must land on the
  // original sample. This is the closed form the implementation has to satisfy, independent of
  // any checkpoint.
  SdSchedulerConfig cfg = sdSched();
  cfg.setAlphaToOne = true;
  std::string err;
  auto s = Scheduler::make(cfg, SamplerKind::Ddim, 1, err);
  REQUIRE(s);
  const int t = s->timesteps().front();
  const double alpha = s->alphasCumprod()[static_cast<std::size_t>(t)];

  FeatureMap clean, noise, sample, pred;
  clean.resize(1, 1, 2);
  clean.data = {0.3f, -0.7f};
  noise.resize(1, 1, 2);
  noise.data = {1.1f, -0.4f};
  sample.resize(1, 1, 2);
  for (int i = 0; i < 2; ++i) {
    sample.data[static_cast<std::size_t>(i)] =
        static_cast<float>(std::sqrt(alpha) * clean.data[static_cast<std::size_t>(i)] +
                           std::sqrt(1.0 - alpha) * noise.data[static_cast<std::size_t>(i)]);
  }
  pred = noise;

  GaussianRng rng(1);
  REQUIRE(s->step(sample, pred, 0, rng, err));
  // alpha_prev is 1 (set_alpha_to_one), so the direction term vanishes and only pred_original
  // survives.
  REQUIRE_THAT(sample.data[0], WithinAbs(0.3, 1e-4));
  REQUIRE_THAT(sample.data[1], WithinAbs(-0.7, 1e-4));
}

TEST_CASE("v-prediction and epsilon read the same model output differently", "[image]") {
  SdSchedulerConfig eps = sdSched();
  SdSchedulerConfig vp = sdSched();
  vp.predictionType = "v_prediction";
  std::string err;
  auto a = Scheduler::make(eps, SamplerKind::Ddim, 4, err);
  auto b = Scheduler::make(vp, SamplerKind::Ddim, 4, err);
  REQUIRE(a);
  REQUIRE(b);

  FeatureMap sampleA, sampleB, pred;
  sampleA.resize(1, 1, 3);
  sampleA.data = {0.5f, -1.2f, 0.9f};
  sampleB = sampleA;
  pred.resize(1, 1, 3);
  pred.data = {0.2f, 0.4f, -0.3f};
  GaussianRng rng(0);
  REQUIRE(a->step(sampleA, pred, 0, rng, err));
  REQUIRE(b->step(sampleB, pred, 0, rng, err));
  REQUIRE(std::abs(sampleA.data[0] - sampleB.data[0]) > 1e-3f);
}

TEST_CASE("the Euler samplers scale their input and DDIM does not", "[image]") {
  std::string err;
  auto ddim = Scheduler::make(sdSched(), SamplerKind::Ddim, 10, err);
  auto euler = Scheduler::make(sdSched(), SamplerKind::Euler, 10, err);
  auto euler50 = Scheduler::make(sdSched(), SamplerKind::Euler, 50, err);
  REQUIRE(ddim);
  REQUIRE(euler);
  REQUIRE(euler50);
  REQUIRE_THAT(ddim->initNoiseSigma(), WithinAbs(1.0, 1e-6));
  // Pinned against the numbers diffusers' EulerDiscreteScheduler reports for this exact config
  // (8.45007 and 13.15847), not against a made-up threshold. They depend on the beta ladder, the
  // spacing AND the step count, so a schedule that drifted in any of the three would move them.
  REQUIRE_THAT(euler->initNoiseSigma(), WithinAbs(8.45007, 1e-4));
  REQUIRE_THAT(euler50->initNoiseSigma(), WithinAbs(13.15847, 1e-4));

  FeatureMap x;
  x.resize(1, 1, 1);
  x.data = {4.0f};
  FeatureMap y = x;
  ddim->scaleModelInput(x, 0);
  euler->scaleModelInput(y, 0);
  REQUIRE_THAT(x.data[0], WithinAbs(4.0, 1e-6));
  REQUIRE(y.data[0] < 4.0f);
}

TEST_CASE("the sampler refuses schedules it cannot build", "[image]") {
  std::string err;
  REQUIRE_FALSE(Scheduler::make(sdSched(), SamplerKind::Ddim, 0, err));
  REQUIRE_FALSE(err.empty());
  REQUIRE_FALSE(Scheduler::make(sdSched(), SamplerKind::Ddim, 5000, err));

  SdSchedulerConfig odd = sdSched();
  odd.betaSchedule = "cosine";
  REQUIRE_FALSE(Scheduler::make(odd, SamplerKind::Ddim, 10, err));
  REQUIRE(err.find("cosine") != std::string::npos);

  odd = sdSched();
  odd.predictionType = "sample";
  REQUIRE_FALSE(Scheduler::make(odd, SamplerKind::Ddim, 10, err));
  REQUIRE(err.find("sample") != std::string::npos);

  odd = sdSched();
  odd.timestepSpacing = "sideways";
  REQUIRE_FALSE(Scheduler::make(odd, SamplerKind::Ddim, 10, err));
}

TEST_CASE("sampler names round-trip", "[image]") {
  for (auto kind : {SamplerKind::Ddim, SamplerKind::Euler, SamplerKind::EulerAncestral}) {
    const auto back = samplerFromName(samplerName(kind));
    REQUIRE(back);
    REQUIRE(*back == kind);
  }
  REQUIRE_FALSE(samplerFromName("dpm++"));
}

// ---- the noise source ---------------------------------------------------------------------------

TEST_CASE("the same seed gives the same noise and a different seed does not", "[image]") {
  GaussianRng a(1234), b(1234), c(1235);
  for (int i = 0; i < 8; ++i) REQUIRE(a.normal() == b.normal());
  GaussianRng d(1234);
  bool anyDifferent = false;
  for (int i = 0; i < 8; ++i) {
    if (d.normal() != c.normal()) anyDifferent = true;
  }
  REQUIRE(anyDifferent);
}

TEST_CASE("the noise is standard normal to a sane tolerance", "[image]") {
  GaussianRng rng(7);
  double sum = 0.0, sumSq = 0.0;
  const int n = 40000;
  for (int i = 0; i < n; ++i) {
    const double v = rng.normal();
    sum += v;
    sumSq += v * v;
  }
  const double mean = sum / n;
  REQUIRE_THAT(mean, WithinAbs(0.0, 0.02));
  REQUIRE_THAT(sumSq / n - mean * mean, WithinAbs(1.0, 0.03));
}

// ---- CLIP's tokenizer ---------------------------------------------------------------------------

TEST_CASE("the CLIP pretokenizer lowercases and collapses whitespace", "[image]") {
  const auto parts = clipPretokenize("  A   Red\tCube \n");
  REQUIRE((parts == std::vector<std::string>{"a", "red", "cube"}));
}

TEST_CASE("the CLIP pretokenizer splits digits one at a time", "[image]") {
  // `[\\p{N}]`, not `[\\p{N}]+` — so "512" is three tokens, which is why a resolution in a prompt
  // costs three positions of the 77.
  const auto parts = clipPretokenize("512px");
  REQUIRE((parts == std::vector<std::string>{"5", "1", "2", "px"}));
}

TEST_CASE("the CLIP pretokenizer keeps contractions and punctuation runs together", "[image]") {
  REQUIRE((clipPretokenize("it's") == std::vector<std::string>{"it", "'s"}));
  REQUIRE((clipPretokenize("wow!!!") == std::vector<std::string>{"wow", "!!!"}));
  REQUIRE((clipPretokenize("<|endoftext|>a") == std::vector<std::string>{"<|endoftext|>", "a"}));
}

TEST_CASE("the byte encoder never emits a space", "[image]") {
  // The merge table is stored as space-separated pairs, so a byte that mapped to a space would
  // make the table ambiguous. GPT-2's stand-in code points exist for exactly this reason.
  std::string all;
  for (int b = 0; b < 256; ++b) all.push_back(static_cast<char>(b));
  const std::string encoded = clipByteEncode(all);
  REQUIRE(encoded.find(' ') == std::string::npos);
  REQUIRE(encoded.find('\n') == std::string::npos);
}

TEST_CASE("the tokenizer pads to the context length and reports truncation", "[image]") {
  // A hand-built vocabulary: one token per letter plus the two specials, and no merges, so every
  // character is its own token and the arithmetic is easy to see.
  std::vector<std::string> tokens{"<|startoftext|>", "<|endoftext|>"};
  for (char c = 'a'; c <= 'z'; ++c) tokens.push_back(std::string(1, c) + "</w>");
  for (char c = 'a'; c <= 'z'; ++c) tokens.push_back(std::string(1, c));
  ClipTokenizer tok(tokens, {}, 0, 1, 1);

  bool truncated = false;
  const auto ids = tok.encodePadded("ab", 8, truncated);
  REQUIRE_FALSE(truncated);
  REQUIRE(ids.size() == 8);
  REQUIRE(ids.front() == 0);
  // "ab" has no merges, so it is 'a' then 'b</w>' — the end-of-word marker is on the LAST symbol.
  REQUIRE(ids[1] == tok.vocabSize() - 26 + 0);  // bare "a"
  REQUIRE(ids[2] == 2 + 1);                     // "b</w>"
  REQUIRE(ids[3] == 1);                         // eos
  for (std::size_t i = 4; i < ids.size(); ++i) REQUIRE(ids[i] == 1);  // padding is eos

  const auto clipped = tok.encodePadded("abcdefgh", 5, truncated);
  REQUIRE(truncated);
  REQUIRE(clipped.size() == 5);
  REQUIRE(clipped.back() == 1);
}

TEST_CASE("merges are applied lowest-rank first", "[image]") {
  std::vector<std::string> tokens{"<|startoftext|>", "<|endoftext|>", "a", "b", "c</w>",
                                  "ab",              "bc</w>",        "abc</w>"};
  // Rank 0 merges a+b before b+c can merge, so "abc" becomes "ab" then "abc</w>".
  ClipTokenizer tok(tokens, {"a b", "b c</w>", "ab c</w>"}, 0, 1, 1);
  const auto ids = tok.encode("abc");
  REQUIRE(ids.size() == 1);
  REQUIRE(tok.idToToken(ids[0]) == "abc</w>");
}

// ---- configuration ------------------------------------------------------------------------------

TEST_CASE("a config whose text encoder does not match the UNet's cross dimension is invalid",
          "[image]") {
  // The pairing this one-file format exists to enforce: a UNet with the wrong text encoder makes
  // a fluent-looking picture of the wrong thing, and nothing errors.
  SdConfig cfg;
  cfg.text.dModel = 768;
  cfg.text.layers = 12;
  cfg.text.heads = 12;
  cfg.text.vocab = 49408;
  cfg.unet.channels = {320, 640};
  cfg.unet.headCounts = {8, 8};
  cfg.unet.transformerDepth = {1, 1};
  cfg.unet.crossDim = 768;
  cfg.vae.channels = {128, 256};
  cfg.vaeScale = 2;
  REQUIRE(cfg.valid());

  cfg.unet.crossDim = 1024;
  REQUIRE_FALSE(cfg.valid());
}

TEST_CASE("a config whose vae stride contradicts its own block count is invalid", "[image]") {
  SdConfig cfg;
  cfg.text.dModel = 768;
  cfg.text.layers = 12;
  cfg.text.heads = 12;
  cfg.text.vocab = 49408;
  cfg.unet.channels = {320};
  cfg.unet.headCounts = {8};
  cfg.unet.transformerDepth = {1};
  cfg.unet.crossDim = 768;
  cfg.vae.channels = {128, 256, 512, 512};
  cfg.vaeScale = 8;
  REQUIRE(cfg.valid());
  cfg.vaeScale = 4;  // four blocks imply a stride of 8
  REQUIRE_FALSE(cfg.valid());
}

// ---- the UNet's skip stack ------------------------------------------------------------------------

TEST_CASE("the UNet consumes exactly the skips its down blocks pushed", "[image]") {
  // No attention and no reference values: this is about the SHAPE of the network. An up block
  // that pops one skip too few finishes with leftovers, one that pops too many runs the stack
  // dry, and both are reported by name rather than crashing or quietly working.
  for (int layers : {1, 2}) {
    for (std::vector<int> channels : {std::vector<int>{4}, std::vector<int>{4, 8},
                                      std::vector<int>{4, 8, 8}}) {
      const SdUnetConfig cfg = tinyUnetConfig(channels, layers);
      const UnetWeights w = tinyUnetWeights(cfg);
      Unet unet(cfg, w);

      FeatureMap latent = ramp(cfg.inChannels, 8, 8);
      std::vector<float> context(static_cast<std::size_t>(2) * cfg.crossDim, 0.1f);
      FeatureMap out;
      std::string err;
      const bool ok = unet.forward(latent, 500, context, 2, out, err);
      INFO("blocks " << channels.size() << " layers " << layers << ": " << err);
      REQUIRE(ok);
      REQUIRE(out.c == cfg.outChannels);
      REQUIRE(out.h == 8);
      REQUIRE(out.w == 8);
    }
  }
}

TEST_CASE("the UNet refuses a latent with the wrong channel count", "[image]") {
  const SdUnetConfig cfg = tinyUnetConfig({4, 8}, 1);
  const UnetWeights w = tinyUnetWeights(cfg);
  Unet unet(cfg, w);
  FeatureMap latent = ramp(3, 8, 8);  // 3 channels, not 4
  std::vector<float> context(static_cast<std::size_t>(2) * cfg.crossDim, 0.0f);
  FeatureMap out;
  std::string err;
  REQUIRE_FALSE(unet.forward(latent, 0, context, 2, out, err));
  REQUIRE(err.find("4 latent channels") != std::string::npos);
}

TEST_CASE("the UNet refuses conditioning of the wrong width", "[image]") {
  const SdUnetConfig cfg = tinyUnetConfig({4}, 1);
  const UnetWeights w = tinyUnetWeights(cfg);
  Unet unet(cfg, w);
  FeatureMap latent = ramp(4, 4, 4);
  std::vector<float> context(11, 0.0f);  // not a multiple of crossDim
  FeatureMap out;
  std::string err;
  REQUIRE_FALSE(unet.forward(latent, 0, context, 2, out, err));
  REQUIRE(err.find("conditioning") != std::string::npos);
}

// ---- residual blocks ------------------------------------------------------------------------------

TEST_CASE("a residual block without a shortcut refuses to change its channel count", "[image]") {
  // The shortcut's presence is derived from the channel counts, so a weight set that disagrees is
  // a loader bug — and adding a residual of the wrong width would read past a buffer.
  ResnetWeights r;
  r.norm1 = unitNorm(4);
  r.conv1 = mat(8, 9 * 4, 0.01f);
  r.norm2 = unitNorm(8);
  r.conv2 = mat(8, 9 * 8, 0.01f);
  // No skip, but 4 -> 8.
  const FeatureMap in = ramp(4, 3, 3);
  FeatureMap out;
  BlockScratch scratch;
  std::string err;
  REQUIRE_FALSE(resnetForward(r, in, nullptr, 1, 1e-5f, out, scratch, err));
  REQUIRE(err.find("shortcut") != std::string::npos);
}

TEST_CASE("the timestep vector enters a residual block per channel", "[image]") {
  ResnetWeights r;
  r.norm1 = unitNorm(2);
  r.conv1 = mat(2, 9 * 2, 0.0f);   // zero convolutions, so only the timestep and the residual move
  r.conv1.b = {0.0f, 0.0f};
  r.timeEmb = mat(2, 3, 0.0f);
  r.timeEmb.b = {1.0f, -2.0f};     // a constant per-channel contribution
  r.norm2 = unitNorm(2);
  r.conv2 = mat(2, 9 * 2, 0.0f);
  r.conv2.b = {0.0f, 0.0f};

  const FeatureMap in = ramp(2, 2, 2);
  const std::vector<float> temb(3, 0.0f);
  FeatureMap out;
  BlockScratch scratch;
  std::string err;
  REQUIRE(resnetForward(r, in, &temb, 1, 1e-5f, out, scratch, err));
  // conv2 is all zeros, so the block's body contributes nothing and the output is the input.
  for (std::size_t i = 0; i < out.size(); ++i) {
    REQUIRE_THAT(out.data[i], WithinAbs(in.data[i], 1e-5));
  }
}
