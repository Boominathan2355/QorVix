#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "qorvix/runtime/pooling.hpp"

using namespace qorvix::runtime;
using Catch::Matchers::WithinAbs;

namespace {

// Three tokens, four dims, row-major [nTokens, dim]. The rows are deliberately distinct constants
// so every pooling mode picks out a value no other mode could produce — mean is 20, CLS is 10,
// last is 30, and no two coincide.
const std::vector<float> kStates{
    10.0f, 11.0f, 12.0f, 13.0f,  // token 0 ([CLS])
    20.0f, 21.0f, 22.0f, 23.0f,  // token 1
    30.0f, 31.0f, 32.0f, 33.0f,  // token 2 ([SEP])
};

}  // namespace

TEST_CASE("mean cls and last pooling each select a different reduction", "[ops]") {
  std::vector<float> out(4);

  ops::meanPool(out.data(), kStates.data(), 3, 4);
  for (int i = 0; i < 4; ++i) REQUIRE_THAT(out[i], WithinAbs(20.0f + i, 1e-5f));

  ops::clsPool(out.data(), kStates.data(), 3, 4);
  for (int i = 0; i < 4; ++i) REQUIRE_THAT(out[i], WithinAbs(10.0f + i, 1e-5f));

  ops::lastPool(out.data(), kStates.data(), 3, 4);
  for (int i = 0; i < 4; ++i) REQUIRE_THAT(out[i], WithinAbs(30.0f + i, 1e-5f));
}

TEST_CASE("pooling a single token gives the same vector for all three modes", "[ops]") {
  const std::vector<float> one{5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> mean(4), cls(4), last(4);
  ops::meanPool(mean.data(), one.data(), 1, 4);
  ops::clsPool(cls.data(), one.data(), 1, 4);
  ops::lastPool(last.data(), one.data(), 1, 4);
  for (int i = 0; i < 4; ++i) {
    REQUIRE_THAT(mean[i], WithinAbs(one[i], 1e-5f));
    REQUIRE(cls[i] == one[i]);
    REQUIRE(last[i] == one[i]);
  }
}

TEST_CASE("l2 normalize scales to unit length", "[ops]") {
  std::vector<float> v{3.0f, 4.0f};  // norm 5
  REQUIRE_THAT(ops::l2Norm(v.data(), 2), WithinAbs(5.0f, 1e-5f));
  ops::l2Normalize(v.data(), 2);
  REQUIRE_THAT(v[0], WithinAbs(0.6f, 1e-5f));
  REQUIRE_THAT(v[1], WithinAbs(0.8f, 1e-5f));
  REQUIRE_THAT(ops::l2Norm(v.data(), 2), WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("l2 normalize leaves a zero vector alone instead of producing NaN", "[ops]") {
  // An empty document embeds to (near) zero. Dividing by that norm yields NaN, and a single NaN
  // vector in a store makes every cosine against it NaN — poisoning the index silently, because
  // every comparison with NaN is false and the entry simply never ranks.
  std::vector<float> zero{0.0f, 0.0f, 0.0f};
  ops::l2Normalize(zero.data(), 3);
  for (float x : zero) {
    REQUIRE(std::isfinite(x));
    REQUIRE(x == 0.0f);
  }
}

TEST_CASE("cosine similarity is one for identical and minus one for antiparallel vectors",
          "[ops]") {
  const std::vector<float> a{1.0f, 2.0f, 3.0f};
  const std::vector<float> neg{-1.0f, -2.0f, -3.0f};
  const std::vector<float> orth{0.0f, 3.0f, -2.0f};  // dot(a, orth) = 0 + 6 - 6 = 0

  REQUIRE_THAT(ops::cosineSimilarity(a.data(), a.data(), 3), WithinAbs(1.0f, 1e-5f));
  REQUIRE_THAT(ops::cosineSimilarity(a.data(), neg.data(), 3), WithinAbs(-1.0f, 1e-5f));
  REQUIRE_THAT(ops::cosineSimilarity(a.data(), orth.data(), 3), WithinAbs(0.0f, 1e-5f));

  // Symmetric, and invariant to the magnitude of either input.
  const std::vector<float> scaled{100.0f, 200.0f, 300.0f};
  REQUIRE_THAT(ops::cosineSimilarity(a.data(), orth.data(), 3),
               WithinAbs(ops::cosineSimilarity(orth.data(), a.data(), 3), 1e-6f));
  REQUIRE_THAT(ops::cosineSimilarity(a.data(), scaled.data(), 3), WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("cosine similarity against a zero vector is zero, not NaN", "[ops]") {
  const std::vector<float> a{1.0f, 2.0f, 3.0f};
  const std::vector<float> zero{0.0f, 0.0f, 0.0f};
  const float c = ops::cosineSimilarity(a.data(), zero.data(), 3);
  REQUIRE(std::isfinite(c));
  REQUIRE(c == 0.0f);
}
