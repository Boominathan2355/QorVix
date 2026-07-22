#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "qorvix/runtime/sampler.hpp"

using namespace qorvix::runtime;

TEST_CASE("temperature <= 0 is greedy argmax", "[sampler]") {
  SamplingParams p;
  p.temperature = 0.0f;
  Sampler s(p, 123);
  std::vector<float> logits{0.1f, 3.0f, 0.5f, 2.9f};
  REQUIRE(s.sample(logits, {}) == 1);
}

TEST_CASE("top-k of 1 always returns the max", "[sampler]") {
  SamplingParams p;
  p.temperature = 1.0f;
  p.topK = 1;
  p.topP = 1.0f;
  p.minP = 0.0f;
  p.repetitionPenalty = 1.0f;
  Sampler s(p, 7);
  std::vector<float> logits{0.0f, 1.0f, 5.0f, 2.0f};
  for (int i = 0; i < 20; ++i) {
    std::vector<float> l = logits;
    REQUIRE(s.sample(l, {}) == 2);
  }
}

TEST_CASE("sampling is deterministic for a fixed seed", "[sampler]") {
  SamplingParams p;  // defaults, stochastic
  std::vector<float> base{1.0f, 1.2f, 0.9f, 1.1f, 0.5f};

  Sampler a(p, 42);
  Sampler b(p, 42);
  for (int i = 0; i < 10; ++i) {
    std::vector<float> la = base, lb = base;
    REQUIRE(a.sample(la, {}) == b.sample(lb, {}));
  }
}

TEST_CASE("repetition penalty suppresses recent tokens", "[sampler]") {
  // Token 1 has the top logit; with a strong repetition penalty and token 1 in the recent
  // history, greedy selection should move off it.
  SamplingParams p;
  p.temperature = 0.0f;  // greedy so the effect is deterministic
  p.repetitionPenalty = 5.0f;
  Sampler s(p, 0);
  std::vector<float> logits{1.0f, 2.0f, 1.9f, 0.0f};
  REQUIRE(s.sample(logits, {1}) == 2);  // token 1 penalized -> token 2 wins
}

TEST_CASE("min-p prunes low-probability tail", "[sampler]") {
  // One dominant token; min-p should drop the rest, making the draw deterministic.
  SamplingParams p;
  p.temperature = 1.0f;
  p.topK = 0;
  p.topP = 1.0f;
  p.minP = 0.5f;
  p.repetitionPenalty = 1.0f;
  Sampler s(p, 99);
  std::vector<float> logits{10.0f, 0.0f, 0.0f, 0.0f};  // token 0 dominates
  for (int i = 0; i < 20; ++i) {
    std::vector<float> l = logits;
    REQUIRE(s.sample(l, {}) == 0);
  }
}
