#include "qorvix/runtime/pooling.hpp"

#include <cmath>
#include <cstddef>

#include "qorvix/runtime/cpu_features.hpp"

namespace qorvix::runtime::ops {

namespace {

// Below this the vector is treated as zero. Chosen well above the float denormal floor so that a
// near-zero vector normalizes to something finite or not at all — never to +/-inf.
constexpr float kZeroNorm = 1e-12f;

}  // namespace

void meanPool(float* out, const float* states, int nTokens, int dim) {
  if (nTokens <= 0 || dim <= 0) return;
  for (int i = 0; i < dim; ++i) out[i] = 0.0f;
  for (int t = 0; t < nTokens; ++t) {
    const float* row = states + static_cast<std::size_t>(t) * dim;
    for (int i = 0; i < dim; ++i) out[i] += row[i];
  }
  const float inv = 1.0f / static_cast<float>(nTokens);
  for (int i = 0; i < dim; ++i) out[i] *= inv;
}

void clsPool(float* out, const float* states, int nTokens, int dim) {
  if (nTokens <= 0 || dim <= 0) return;
  for (int i = 0; i < dim; ++i) out[i] = states[i];
}

void lastPool(float* out, const float* states, int nTokens, int dim) {
  if (nTokens <= 0 || dim <= 0) return;
  const float* row = states + static_cast<std::size_t>(nTokens - 1) * dim;
  for (int i = 0; i < dim; ++i) out[i] = row[i];
}

float l2Norm(const float* x, int n) {
  if (n <= 0) return 0.0f;
  return std::sqrt(cpu::dotProductF32(x, x, n));
}

void l2Normalize(float* x, int n) {
  const float norm = l2Norm(x, n);
  if (norm <= kZeroNorm) return;  // see the header: never turn an empty document into NaN
  const float inv = 1.0f / norm;
  for (int i = 0; i < n; ++i) x[i] *= inv;
}

float cosineSimilarity(const float* a, const float* b, int n) {
  if (n <= 0) return 0.0f;
  const float na = l2Norm(a, n);
  const float nb = l2Norm(b, n);
  if (na <= kZeroNorm || nb <= kZeroNorm) return 0.0f;
  return cpu::dotProductF32(a, b, n) / (na * nb);
}

}  // namespace qorvix::runtime::ops
