#include "qorvix/image/nn.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "qorvix/runtime/ops.hpp"

namespace qorvix::image {

namespace {

// How much memory one im2col strip may take. The strip height falls out of this and the row's
// patch width, so a 4-channel first convolution processes the whole image in one strip while a
// 512-channel one at 512x512 walks it a few rows at a time.
constexpr std::size_t kIm2ColBudgetBytes = 64ull << 20;

}  // namespace

bool conv2d(FeatureMap& out, const FeatureMap& in, const runtime::WeightMat& weight,
            const std::vector<float>& bias, int kernel, int stride, int pad,
            std::vector<float>& scratch, std::string& error) {
  if (in.c <= 0 || in.h <= 0 || in.w <= 0) {
    error = "conv2d: empty input";
    return false;
  }
  const int patch = kernel * kernel * in.c;
  if (weight.cols != patch) {
    error = "conv2d: weight expects " + std::to_string(weight.cols) + " inputs per position but a " +
            std::to_string(kernel) + "x" + std::to_string(kernel) + " patch over " +
            std::to_string(in.c) + " channels is " + std::to_string(patch);
    return false;
  }
  const int outH = (in.h + 2 * pad - kernel) / stride + 1;
  const int outW = (in.w + 2 * pad - kernel) / stride + 1;
  if (outH <= 0 || outW <= 0) {
    error = "conv2d: kernel " + std::to_string(kernel) + " stride " + std::to_string(stride) +
            " pad " + std::to_string(pad) + " leaves no output for a " + std::to_string(in.h) +
            "x" + std::to_string(in.w) + " input";
    return false;
  }
  out.resize(weight.rows, outH, outW);

  // One strip is `rowsPerStrip` output rows; each output position contributes `patch` floats.
  const std::size_t bytesPerRow = static_cast<std::size_t>(outW) * patch * sizeof(float);
  int rowsPerStrip = bytesPerRow == 0 ? outH
                                      : static_cast<int>(kIm2ColBudgetBytes / std::max<std::size_t>(bytesPerRow, 1));
  rowsPerStrip = std::clamp(rowsPerStrip, 1, outH);
  scratch.resize(static_cast<std::size_t>(rowsPerStrip) * outW * patch);

  for (int y0 = 0; y0 < outH; y0 += rowsPerStrip) {
    const int rows = std::min(rowsPerStrip, outH - y0);
    // im2col. The inner run over input channels is a contiguous copy in BOTH buffers because the
    // activation is position-major and the weight was permuted to (kh, kw, in) at conversion.
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
      const int oy = y0 + r;
      for (int ox = 0; ox < outW; ++ox) {
        float* col = scratch.data() + (static_cast<std::size_t>(r) * outW + ox) * patch;
        for (int ky = 0; ky < kernel; ++ky) {
          const int iy = oy * stride - pad + ky;
          for (int kx = 0; kx < kernel; ++kx) {
            const int ix = ox * stride - pad + kx;
            float* dst = col + (static_cast<std::size_t>(ky) * kernel + kx) * in.c;
            if (iy < 0 || iy >= in.h || ix < 0 || ix >= in.w) {
              std::memset(dst, 0, static_cast<std::size_t>(in.c) * sizeof(float));
            } else {
              std::memcpy(dst, in.at(iy, ix), static_cast<std::size_t>(in.c) * sizeof(float));
            }
          }
        }
      }
    }
    runtime::wmatmulNBias(out.data.data() + static_cast<std::size_t>(y0) * outW * out.c, weight,
                          scratch.data(), rows * outW, bias);
  }
  return true;
}

bool pointwise(FeatureMap& out, const FeatureMap& in, const runtime::WeightMat& weight,
               const std::vector<float>& bias, std::string& error) {
  if (weight.cols != in.c) {
    error = "pointwise: weight takes " + std::to_string(weight.cols) + " channels, input has " +
            std::to_string(in.c);
    return false;
  }
  out.resize(weight.rows, in.h, in.w);
  runtime::wmatmulNBias(out.data.data(), weight, in.data.data(),
                        static_cast<int>(in.positions()), bias);
  return true;
}

void groupNorm(FeatureMap& x, const std::vector<float>& weight, const std::vector<float>& bias,
               int groups, float eps) {
  if (groups <= 0 || x.c % groups != 0) return;
  const int perGroup = x.c / groups;
  const std::size_t n = x.positions();

  // Per-channel sums first, accumulated while walking the map once, front to back. Summing per
  // GROUP directly would be the same arithmetic but would read the same cache lines `groups`
  // times; this reads them once and folds channels into groups afterwards, which is free.
  std::vector<double> sum(x.c, 0.0), sumSq(x.c, 0.0);
  for (std::size_t p = 0; p < n; ++p) {
    const float* row = x.data.data() + p * x.c;
    for (int i = 0; i < x.c; ++i) {
      sum[i] += row[i];
      sumSq[i] += static_cast<double>(row[i]) * row[i];
    }
  }

  // scale[i], shift[i] fold the group statistics and the affine parameters into one multiply-add
  // per element, so the normalization pass below is a single streaming loop.
  std::vector<float> scale(x.c), shift(x.c);
  const double count = static_cast<double>(n) * perGroup;
  for (int g = 0; g < groups; ++g) {
    double s = 0.0, sq = 0.0;
    for (int i = g * perGroup; i < (g + 1) * perGroup; ++i) {
      s += sum[i];
      sq += sumSq[i];
    }
    const double mean = s / count;
    // Biased (population) variance, which is what torch.nn.GroupNorm computes.
    const double var = sq / count - mean * mean;
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int i = g * perGroup; i < (g + 1) * perGroup; ++i) {
      const float wq = weight.empty() ? 1.0f : weight[i];
      const float bq = bias.empty() ? 0.0f : bias[i];
      scale[i] = static_cast<float>(inv) * wq;
      shift[i] = static_cast<float>(-mean * inv) * wq + bq;
    }
  }

#pragma omp parallel for schedule(static)
  for (std::ptrdiff_t p = 0; p < static_cast<std::ptrdiff_t>(n); ++p) {
    float* row = x.data.data() + static_cast<std::size_t>(p) * x.c;
    for (int i = 0; i < x.c; ++i) row[i] = row[i] * scale[i] + shift[i];
  }
}

void siluInPlace(float* x, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) x[i] = runtime::ops::silu(x[i]);
}

void addInPlace(float* dst, const float* src, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] += src[i];
}

void addPerChannel(FeatureMap& x, const float* perChannel) {
  const std::size_t n = x.positions();
  for (std::size_t p = 0; p < n; ++p) {
    float* row = x.data.data() + p * x.c;
    for (int i = 0; i < x.c; ++i) row[i] += perChannel[i];
  }
}

void upsampleNearest2x(FeatureMap& out, const FeatureMap& in) {
  out.resize(in.c, in.h * 2, in.w * 2);
  const std::size_t rowBytes = static_cast<std::size_t>(in.c) * sizeof(float);
  for (int y = 0; y < in.h; ++y) {
    for (int x = 0; x < in.w; ++x) {
      const float* src = in.at(y, x);
      std::memcpy(out.at(y * 2, x * 2), src, rowBytes);
      std::memcpy(out.at(y * 2, x * 2 + 1), src, rowBytes);
      std::memcpy(out.at(y * 2 + 1, x * 2), src, rowBytes);
      std::memcpy(out.at(y * 2 + 1, x * 2 + 1), src, rowBytes);
    }
  }
}

bool concatChannels(FeatureMap& out, const FeatureMap& a, const FeatureMap& b, std::string& error) {
  if (a.h != b.h || a.w != b.w) {
    error = "skip connection joins a " + std::to_string(a.h) + "x" + std::to_string(a.w) +
            " map to a " + std::to_string(b.h) + "x" + std::to_string(b.w) + " one";
    return false;
  }
  out.resize(a.c + b.c, a.h, a.w);
  const std::size_t n = a.positions();
  for (std::size_t p = 0; p < n; ++p) {
    float* dst = out.data.data() + p * out.c;
    std::memcpy(dst, a.data.data() + p * a.c, static_cast<std::size_t>(a.c) * sizeof(float));
    std::memcpy(dst + a.c, b.data.data() + p * b.c, static_cast<std::size_t>(b.c) * sizeof(float));
  }
  return true;
}

void attention(float* out, const float* q, const float* k, const float* v, int nq, int nkv,
               int heads, int headDim, bool causal) {
  const int dim = heads * headDim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

  // Parallel over query positions rather than heads: a spatial transformer at 64x64 has 4096
  // queries and only 8 heads, and this self-attention is the single most expensive thing in the
  // UNet. Each thread carries its own score row, which is why the scratch is declared inside the
  // parallel region rather than handed in.
#pragma omp parallel
  {
    std::vector<float> row(static_cast<std::size_t>(nkv));
#pragma omp for schedule(static)
    for (int i = 0; i < nq; ++i) {
      const int limit = causal ? std::min(i + 1, nkv) : nkv;
      for (int hd = 0; hd < heads; ++hd) {
        const float* qv = q + static_cast<std::size_t>(i) * dim + hd * headDim;
        for (int j = 0; j < limit; ++j) {
          const float* kv = k + static_cast<std::size_t>(j) * dim + hd * headDim;
          float acc = 0.0f;
          for (int d = 0; d < headDim; ++d) acc += qv[d] * kv[d];
          row[j] = acc * scale;
        }
        runtime::ops::softmax(row.data(), limit);
        float* o = out + static_cast<std::size_t>(i) * dim + hd * headDim;
        for (int d = 0; d < headDim; ++d) o[d] = 0.0f;
        for (int j = 0; j < limit; ++j) {
          const float p = row[j];
          const float* vv = v + static_cast<std::size_t>(j) * dim + hd * headDim;
          for (int d = 0; d < headDim; ++d) o[d] += p * vv[d];
        }
      }
    }
  }
}

}  // namespace qorvix::image
