#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/text_model.hpp"

using namespace qorvix::runtime;
using Catch::Matchers::WithinAbs;

namespace {

// A minimal but valid config: d=4, 2 heads (head_dim=2), MHA, 1 layer, ffn=4, vocab=3.
ModelConfig tinyConfig() {
  ModelConfig c;
  c.architecture = "llama";
  c.vocabSize = 3;
  c.contextLength = 16;
  c.embeddingLength = 4;
  c.blockCount = 1;
  c.feedForwardLength = 4;
  c.headCount = 2;
  c.headCountKv = 2;
  c.ropeDimensionCount = 2;
  c.ropeFreqBase = 10000.0f;
  c.normEpsilon = 1e-5f;
  c.ropeMode = ops::RopeMode::Neox;
  return c;
}

LayerWeights zeroLayer(const ModelConfig& c) {
  const std::size_t d = c.embeddingLength, kv = c.kvDim(), ffn = c.feedForwardLength;
  LayerWeights L;
  L.attnNorm.assign(d, 1.0f);
  L.wq.assign(d * d, 0.0f);
  L.wk.assign(kv * d, 0.0f);
  L.wv.assign(kv * d, 0.0f);
  L.wo.assign(d * d, 0.0f);
  L.ffnNorm.assign(d, 1.0f);
  L.ffnGate.assign(ffn * d, 0.0f);
  L.ffnUp.assign(ffn * d, 0.0f);
  L.ffnDown.assign(d * ffn, 0.0f);
  return L;
}

}  // namespace

// With attention and FFN weights all zero, both residual branches contribute nothing, so the
// hidden state is exactly the token embedding and logits = lmHead * rmsnorm(embedding). Fully
// analytic — validates embedding lookup, residual wiring, final norm, and the LM head.
TEST_CASE("forward reduces to embedding -> norm -> lm-head when blocks are zeroed", "[text_model]") {
  ModelConfig c = tinyConfig();
  Weights w;
  w.tokenEmbd = {0, 0, 0, 0,   // token 0
                 1, 2, 3, 4,   // token 1
                 -1, -1, -1, -1};
  w.layers = {zeroLayer(c)};
  w.outputNorm.assign(4, 1.0f);
  // LM head selects individual normalized channels: logits[i] = xf[i].
  w.output = {1, 0, 0, 0, /**/ 0, 1, 0, 0, /**/ 0, 0, 1, 0};

  TextModel model(c, w, /*maxSeqLen=*/16);
  const auto& logits = model.forward(/*token=*/1, /*pos=*/0);

  std::vector<float> xf(4);
  const std::vector<float> emb{1, 2, 3, 4};
  ops::rmsnorm(xf.data(), emb.data(), w.outputNorm.data(), 4, c.normEpsilon);
  REQUIRE(logits.size() == 3);
  REQUIRE_THAT(logits[0], WithinAbs(xf[0], 1e-5f));
  REQUIRE_THAT(logits[1], WithinAbs(xf[1], 1e-5f));
  REQUIRE_THAT(logits[2], WithinAbs(xf[2], 1e-5f));
}

// With Q=K=0, attention scores are all zero, so softmax is uniform and the attention output is
// the mean of the cached value vectors. Building the reference from the (independently verified)
// ops validates the attention/KV path and the residual + FFN wiring.
TEST_CASE("attention with zero Q/K averages the value vectors", "[text_model]") {
  ModelConfig c = tinyConfig();
  c.headCount = 1;  // single head, head_dim = 4
  c.headCountKv = 1;
  c.ropeDimensionCount = 4;

  Weights w;
  w.tokenEmbd = {0, 0, 0, 0, /**/ 2, -1, 0.5f, 3, /**/ 0, 0, 0, 0};
  LayerWeights L = zeroLayer(c);
  // wv = identity so V = normed input; wo = identity so attn output flows to the residual.
  L.wv = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  L.wo = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  w.layers = {L};
  w.outputNorm.assign(4, 1.0f);
  w.output = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};  // 3 x 4

  TextModel model(c, w, 16);
  const auto& logits = model.forward(1, 0);

  // Reference: single position -> attn out = V = rmsnorm(emb). x = emb + V. logits = norm(x).
  const std::vector<float> emb{2, -1, 0.5f, 3};
  std::vector<float> vv(4);
  ops::rmsnorm(vv.data(), emb.data(), L.attnNorm.data(), 4, c.normEpsilon);
  std::vector<float> x = emb;
  ops::add(x.data(), vv.data(), 4);  // FFN is zero -> no second contribution
  std::vector<float> xf(4);
  ops::rmsnorm(xf.data(), x.data(), w.outputNorm.data(), 4, c.normEpsilon);

  for (int i = 0; i < 3; ++i) REQUIRE_THAT(logits[i], WithinAbs(xf[i], 1e-4f));
}

TEST_CASE("greedy generation is deterministic and respects limits", "[text_model]") {
  ModelConfig c = tinyConfig();
  Weights w;
  // Embedding biases the hidden state so token 2 always wins the LM head -> constant output.
  w.tokenEmbd = {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f};
  w.layers = {zeroLayer(c)};
  w.outputNorm.assign(4, 1.0f);
  w.output = {0, 0, 0, 0, /**/ 0, 0, 0, 0, /**/ 1, 1, 1, 1};  // token 2 has the largest logit

  TextModel model(c, w, 16);
  auto a = model.generateGreedy({0, 1}, 3);
  auto b = model.generateGreedy({0, 1}, 3);
  REQUIRE(a.size() == 3);
  REQUIRE(a == b);                 // deterministic
  REQUIRE(a[0] == 2);              // token 2 always wins
  REQUIRE(a[1] == 2);

  // maxNew is honored and the KV cache resets between runs.
  REQUIRE(model.generateGreedy({0}, 1).size() == 1);
}

TEST_CASE("GQA groups query heads onto shared kv heads", "[text_model]") {
  // 4 query heads, 2 kv heads -> group size 2. Just exercise the mapping end to end (no crash,
  // finite logits of the right shape) with a full non-trivial config.
  ModelConfig c = tinyConfig();
  c.embeddingLength = 8;
  c.headCount = 4;
  c.headCountKv = 2;  // kvDim = 2 * 2 = 4
  c.feedForwardLength = 8;

  Weights w;
  w.tokenEmbd.assign(c.vocabSize * c.embeddingLength, 0.05f);
  LayerWeights L = zeroLayer(c);
  for (std::size_t i = 0; i < L.wv.size(); ++i) L.wv[i] = 0.01f * (i % 7);
  w.layers = {L};
  w.outputNorm.assign(c.embeddingLength, 1.0f);
  w.output.assign(c.vocabSize * c.embeddingLength, 0.1f);

  TextModel model(c, w, 16);
  const auto& logits = model.forward(1, 0);
  REQUIRE(logits.size() == c.vocabSize);
  for (float v : logits) REQUIRE(std::isfinite(v));
}
