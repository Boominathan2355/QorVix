#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
