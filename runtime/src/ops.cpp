#include "qorvix/runtime/ops.hpp"

#include <algorithm>
#include <cmath>

namespace qorvix::runtime::ops {

void rmsnorm(float* out, const float* x, const float* weight, int n, float eps) {
  double sumSq = 0.0;
  for (int i = 0; i < n; ++i) sumSq += static_cast<double>(x[i]) * x[i];
  const float scale = static_cast<float>(1.0 / std::sqrt(sumSq / n + eps));
  for (int i = 0; i < n; ++i) out[i] = x[i] * scale * weight[i];
}

void layernorm(float* out, const float* x, const float* weight, const float* bias, int n,
               float eps) {
  double mean = 0.0;
  for (int i = 0; i < n; ++i) mean += x[i];
  mean /= n;
  double var = 0.0;
  for (int i = 0; i < n; ++i) {
    const double d = x[i] - mean;
    var += d * d;
  }
  var /= n;
  const float invStd = static_cast<float>(1.0 / std::sqrt(var + eps));
  for (int i = 0; i < n; ++i) {
    float v = static_cast<float>((x[i] - mean) * invStd) * weight[i];
    if (bias) v += bias[i];
    out[i] = v;
  }
}

void matmul(float* out, const float* weight, const float* x, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* row = weight + static_cast<std::size_t>(r) * cols;
    double acc = 0.0;
    for (int c = 0; c < cols; ++c) acc += static_cast<double>(row[c]) * x[c];
    out[r] = static_cast<float>(acc);
  }
}

void softmax(float* x, int n) {
  if (n <= 0) return;
  float maxV = x[0];
  for (int i = 1; i < n; ++i) maxV = std::max(maxV, x[i]);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - maxV);
    sum += x[i];
  }
  const float inv = static_cast<float>(1.0 / sum);
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

float silu(float z) { return z / (1.0f + std::exp(-z)); }

void swiglu(float* out, const float* gate, const float* up, int n) {
  for (int i = 0; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}

void add(float* out, const float* x, int n) {
  for (int i = 0; i < n; ++i) out[i] += x[i];
}

void rope(float* vec, int n_heads, int head_dim, int pos, float freqBase, RopeMode mode) {
  const int half = head_dim / 2;
  for (int h = 0; h < n_heads; ++h) {
    float* v = vec + static_cast<std::size_t>(h) * head_dim;
    for (int i = 0; i < half; ++i) {
      const float theta = pos * std::pow(freqBase, -2.0f * i / head_dim);
      const float cosT = std::cos(theta);
      const float sinT = std::sin(theta);
      if (mode == RopeMode::Interleaved) {
        const float a = v[2 * i];
        const float b = v[2 * i + 1];
        v[2 * i] = a * cosT - b * sinT;
        v[2 * i + 1] = a * sinT + b * cosT;
      } else {  // Neox
        const float a = v[i];
        const float b = v[i + half];
        v[i] = a * cosT - b * sinT;
        v[i + half] = a * sinT + b * cosT;
      }
    }
  }
}

int argmax(const float* x, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

}  // namespace qorvix::runtime::ops
