#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "qorvix/runtime/ops.hpp"

using namespace qorvix::runtime;
using Catch::Matchers::WithinAbs;

TEST_CASE("rmsnorm normalizes by RMS and applies weight", "[ops]") {
  // x = [3,4] -> mean(x^2) = (9+16)/2 = 12.5 -> rms = sqrt(12.5) ~= 3.5355 (eps ~ 0).
  std::vector<float> x{3.0f, 4.0f};
  std::vector<float> w{1.0f, 1.0f};
  std::vector<float> out(2);
  ops::rmsnorm(out.data(), x.data(), w.data(), 2, 0.0f);
  const float rms = std::sqrt(12.5f);
  REQUIRE_THAT(out[0], WithinAbs(3.0f / rms, 1e-5f));
  REQUIRE_THAT(out[1], WithinAbs(4.0f / rms, 1e-5f));

  // Weight scales each channel.
  w = {2.0f, 0.5f};
  ops::rmsnorm(out.data(), x.data(), w.data(), 2, 0.0f);
  REQUIRE_THAT(out[0], WithinAbs(2.0f * 3.0f / rms, 1e-5f));
  REQUIRE_THAT(out[1], WithinAbs(0.5f * 4.0f / rms, 1e-5f));
}

// layernorm was written in Phase 5 for a caller that didn't exist yet and shipped with ZERO callers
// and zero tests — the encoder in Phase 11a is its first user. These cases pin the three choices a
// re-implementation could plausibly get wrong and still look right on a smoke test.
TEST_CASE("layernorm centers and scales by the population variance", "[ops]") {
  // x = [1,2,3,4] -> mean 2.5, population variance ((1.5^2)*2 + (0.5^2)*2)/4 = 1.25 (NOT the
  // sample variance 5/3). Dividing by n-1 instead of n is the single most likely silent bug: the
  // output stays finite, unit-ish and plausible, and only shifts every embedding slightly.
  std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> w{1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> out(4);
  ops::layernorm(out.data(), x.data(), w.data(), nullptr, 4, 0.0f);

  const float invStd = 1.0f / std::sqrt(1.25f);
  REQUIRE_THAT(out[0], WithinAbs(-1.5f * invStd, 1e-5f));
  REQUIRE_THAT(out[1], WithinAbs(-0.5f * invStd, 1e-5f));
  REQUIRE_THAT(out[2], WithinAbs(0.5f * invStd, 1e-5f));
  REQUIRE_THAT(out[3], WithinAbs(1.5f * invStd, 1e-5f));

  // Normalized output has zero mean and unit variance.
  const float mean = (out[0] + out[1] + out[2] + out[3]) / 4.0f;
  REQUIRE_THAT(mean, WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("layernorm applies weight and bias, and a null bias equals a zero bias", "[ops]") {
  std::vector<float> x{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> w{2.0f, 0.5f, -1.0f, 3.0f};
  std::vector<float> b{10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<float> plain(4), scaled(4), biased(4), zeroBias(4);

  ops::layernorm(plain.data(), x.data(), std::vector<float>(4, 1.0f).data(), nullptr, 4, 0.0f);
  ops::layernorm(scaled.data(), x.data(), w.data(), nullptr, 4, 0.0f);
  ops::layernorm(biased.data(), x.data(), w.data(), b.data(), 4, 0.0f);

  for (int i = 0; i < 4; ++i) {
    REQUIRE_THAT(scaled[i], WithinAbs(plain[i] * w[i], 1e-5f));
    REQUIRE_THAT(biased[i], WithinAbs(plain[i] * w[i] + b[i], 1e-5f));
  }

  // A null bias pointer must take the same path as an all-zero bias, not a different one.
  const std::vector<float> zeros(4, 0.0f);
  ops::layernorm(zeroBias.data(), x.data(), w.data(), zeros.data(), 4, 0.0f);
  for (int i = 0; i < 4; ++i) REQUIRE(zeroBias[i] == scaled[i]);
}

TEST_CASE("layernorm on constant input returns the bias without dividing by zero", "[ops]") {
  // Variance is exactly 0, so eps is the only thing standing between this and a NaN. BERT trains
  // at eps=1e-12, which is small enough that a sloppy implementation can still blow up here.
  std::vector<float> x{7.0f, 7.0f, 7.0f, 7.0f};
  std::vector<float> w{1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> b{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out(4);
  ops::layernorm(out.data(), x.data(), w.data(), b.data(), 4, 1e-12f);
  for (int i = 0; i < 4; ++i) {
    REQUIRE(std::isfinite(out[i]));
    REQUIRE_THAT(out[i], WithinAbs(b[i], 1e-4f));
  }
}

TEST_CASE("gelu matches known reference values and saturates at both tails", "[ops]") {
  REQUIRE_THAT(ops::gelu(0.0f), WithinAbs(0.0f, 1e-6f));
  // Reference values from the exact erf form (PyTorch nn.GELU).
  REQUIRE_THAT(ops::gelu(1.0f), WithinAbs(0.8413447f, 1e-5f));
  REQUIRE_THAT(ops::gelu(-1.0f), WithinAbs(-0.1586553f, 1e-5f));
  REQUIRE_THAT(ops::gelu(2.0f), WithinAbs(1.9544997f, 1e-5f));
  // Tails: gelu(z) -> 0 as z -> -inf and -> z as z -> +inf.
  REQUIRE_THAT(ops::gelu(-10.0f), WithinAbs(0.0f, 1e-6f));
  REQUIRE_THAT(ops::gelu(10.0f), WithinAbs(10.0f, 1e-5f));

  // gelu is not monotonic overall (it dips below zero near z ~ -0.75) but is monotonic above 0.
  float prev = ops::gelu(0.0f);
  for (float z = 0.1f; z <= 5.0f; z += 0.1f) {
    const float cur = ops::gelu(z);
    REQUIRE(cur > prev);
    prev = cur;
  }
}

TEST_CASE("the tanh gelu approximation tracks the exact form within 1e-3", "[ops]") {
  // The two variants are close but distinguishable — which is exactly why both exist here. If this
  // bound ever fails, one of the two implementations is wrong, not merely a different convention.
  float maxDiff = 0.0f;
  for (float z = -5.0f; z <= 5.0f; z += 0.05f) {
    maxDiff = std::max(maxDiff, std::abs(ops::gelu(z) - ops::geluTanh(z)));
  }
  REQUIRE(maxDiff < 1e-3f);
  REQUIRE(maxDiff > 1e-5f);  // they are genuinely different functions, not aliases
}

TEST_CASE("geluInPlace applies gelu elementwise", "[ops]") {
  std::vector<float> x{-1.0f, 0.0f, 1.0f, 2.0f};
  const std::vector<float> ref = x;
  ops::geluInPlace(x.data(), 4);
  for (int i = 0; i < 4; ++i) REQUIRE_THAT(x[i], WithinAbs(ops::gelu(ref[i]), 1e-6f));
}

TEST_CASE("matmul computes row dot products", "[ops]") {
  // W = [[1,2,3],[4,5,6]], x = [1,0,-1] -> out = [1-3, 4-6] = [-2,-2].
  std::vector<float> W{1, 2, 3, 4, 5, 6};
  std::vector<float> x{1, 0, -1};
  std::vector<float> out(2);
  ops::matmul(out.data(), W.data(), x.data(), 2, 3);
  REQUIRE_THAT(out[0], WithinAbs(-2.0f, 1e-6f));
  REQUIRE_THAT(out[1], WithinAbs(-2.0f, 1e-6f));
}

TEST_CASE("softmax is stable and sums to one", "[ops]") {
  std::vector<float> x{1.0f, 2.0f, 3.0f};
  ops::softmax(x.data(), 3);
  float sum = x[0] + x[1] + x[2];
  REQUIRE_THAT(sum, WithinAbs(1.0f, 1e-6f));
  REQUIRE(x[2] > x[1]);
  REQUIRE(x[1] > x[0]);

  // Large values must not overflow (stability).
  std::vector<float> big{1000.0f, 1000.0f};
  ops::softmax(big.data(), 2);
  REQUIRE_THAT(big[0], WithinAbs(0.5f, 1e-6f));
}

TEST_CASE("silu and swiglu", "[ops]") {
  REQUIRE_THAT(ops::silu(0.0f), WithinAbs(0.0f, 1e-6f));
  // silu(1) = 1 * sigmoid(1) = 1/(1+e^-1) ~= 0.731059.
  REQUIRE_THAT(ops::silu(1.0f), WithinAbs(0.7310586f, 1e-5f));

  std::vector<float> gate{1.0f, 2.0f}, up{3.0f, 4.0f}, out(2);
  ops::swiglu(out.data(), gate.data(), up.data(), 2);
  REQUIRE_THAT(out[0], WithinAbs(ops::silu(1.0f) * 3.0f, 1e-5f));
  REQUIRE_THAT(out[1], WithinAbs(ops::silu(2.0f) * 4.0f, 1e-5f));
}

TEST_CASE("rope at position 0 is the identity", "[ops]") {
  std::vector<float> v{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> ref = v;
  ops::rope(v.data(), 1, 4, 0, 10000.0f, ops::RopeMode::Neox);
  for (int i = 0; i < 4; ++i) REQUIRE_THAT(v[i], WithinAbs(ref[i], 1e-6f));
}

TEST_CASE("rope rotates the first pair by the base angle", "[ops]") {
  // head_dim=2, pos=1: theta_0 = 1 * base^(0) = 1 rad. Pair (v0,v1) rotates by 1 rad.
  std::vector<float> v{1.0f, 0.0f};
  ops::rope(v.data(), 1, 2, 1, 10000.0f, ops::RopeMode::Interleaved);
  REQUIRE_THAT(v[0], WithinAbs(std::cos(1.0f), 1e-5f));
  REQUIRE_THAT(v[1], WithinAbs(std::sin(1.0f), 1e-5f));

  // rope preserves the norm of each rotated pair.
  std::vector<float> u{0.6f, 0.8f};
  ops::rope(u.data(), 1, 2, 5, 10000.0f, ops::RopeMode::Interleaved);
  REQUIRE_THAT(std::sqrt(u[0] * u[0] + u[1] * u[1]), WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("add and argmax", "[ops]") {
  std::vector<float> a{1, 2, 3}, b{10, 20, 30};
  ops::add(a.data(), b.data(), 3);
  REQUIRE(a[0] == 11.0f);
  REQUIRE(a[2] == 33.0f);

  std::vector<float> logits{0.1f, -3.0f, 5.0f, 4.9f};
  REQUIRE(ops::argmax(logits.data(), 4) == 2);
}
