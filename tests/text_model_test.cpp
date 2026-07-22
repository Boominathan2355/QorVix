#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/runtime/weights.hpp"

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
  const int d = c.embeddingLength, kv = c.kvDim(), ffn = c.feedForwardLength;
  LayerWeights L;
  L.attnNorm.assign(d, 1.0f);
  L.wq = WeightMat::f32(std::vector<float>(d * d, 0.0f), d, d);
  L.wk = WeightMat::f32(std::vector<float>(kv * d, 0.0f), kv, d);
  L.wv = WeightMat::f32(std::vector<float>(kv * d, 0.0f), kv, d);
  L.wo = WeightMat::f32(std::vector<float>(d * d, 0.0f), d, d);
  L.ffnNorm.assign(d, 1.0f);
  L.ffnGate = WeightMat::f32(std::vector<float>(ffn * d, 0.0f), ffn, d);
  L.ffnUp = WeightMat::f32(std::vector<float>(ffn * d, 0.0f), ffn, d);
  L.ffnDown = WeightMat::f32(std::vector<float>(d * ffn, 0.0f), d, ffn);
  return L;
}

}  // namespace

// With attention and FFN weights all zero, both residual branches contribute nothing, so the
// hidden state is exactly the token embedding and logits = lmHead * rmsnorm(embedding).
TEST_CASE("forward reduces to embedding -> norm -> lm-head when blocks are zeroed", "[text_model]") {
  ModelConfig c = tinyConfig();
  Weights w;
  w.tokenEmbd = WeightMat::f32({0, 0, 0, 0, 1, 2, 3, 4, -1, -1, -1, -1}, 3, 4);
  w.layers = {zeroLayer(c)};
  w.outputNorm.assign(4, 1.0f);
  w.output = WeightMat::f32({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 3, 4);

  TextModel model(c, std::move(w), /*maxSeqLen=*/16);
  const auto& logits = model.forward(/*token=*/1, /*pos=*/0);

  std::vector<float> xf(4);
  const std::vector<float> emb{1, 2, 3, 4}, ones(4, 1.0f);
  ops::rmsnorm(xf.data(), emb.data(), ones.data(), 4, c.normEpsilon);
  REQUIRE(logits.size() == 3);
  REQUIRE_THAT(logits[0], WithinAbs(xf[0], 1e-5f));
  REQUIRE_THAT(logits[1], WithinAbs(xf[1], 1e-5f));
  REQUIRE_THAT(logits[2], WithinAbs(xf[2], 1e-5f));
}

// With Q=K=0, attention scores are all zero (uniform softmax) so the attention output is the mean
// of the cached values. Single position -> attn out = V = rmsnorm(emb).
TEST_CASE("attention with zero Q/K averages the value vectors", "[text_model]") {
  ModelConfig c = tinyConfig();
  c.headCount = 1;  // single head, head_dim = 4
  c.headCountKv = 1;
  c.ropeDimensionCount = 4;

  Weights w;
  w.tokenEmbd = WeightMat::f32({0, 0, 0, 0, 2, -1, 0.5f, 3, 0, 0, 0, 0}, 3, 4);
  LayerWeights L = zeroLayer(c);
  L.wv = WeightMat::f32({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, 4, 4);
  L.wo = WeightMat::f32({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, 4, 4);
  w.layers = {std::move(L)};
  w.outputNorm.assign(4, 1.0f);
  w.output = WeightMat::f32({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 3, 4);

  TextModel model(c, std::move(w), 16);
  const auto& logits = model.forward(1, 0);

  const std::vector<float> emb{2, -1, 0.5f, 3}, ones(4, 1.0f);
  std::vector<float> vv(4);
  ops::rmsnorm(vv.data(), emb.data(), ones.data(), 4, c.normEpsilon);
  std::vector<float> x = emb;
  ops::add(x.data(), vv.data(), 4);
  std::vector<float> xf(4);
  ops::rmsnorm(xf.data(), x.data(), ones.data(), 4, c.normEpsilon);

  for (int i = 0; i < 3; ++i) REQUIRE_THAT(logits[i], WithinAbs(xf[i], 1e-4f));
}

TEST_CASE("greedy generation is deterministic and respects limits", "[text_model]") {
  ModelConfig c = tinyConfig();
  Weights w;
  w.tokenEmbd = WeightMat::f32(
      {0.1f, 0.1f, 0.1f, 0.1f, 0.2f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f}, 3, 4);
  w.layers = {zeroLayer(c)};
  w.outputNorm.assign(4, 1.0f);
  w.output = WeightMat::f32({0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1}, 3, 4);  // token 2 wins

  TextModel model(c, std::move(w), 16);
  auto a = model.generateGreedy({0, 1}, 3);
  auto b = model.generateGreedy({0, 1}, 3);
  REQUIRE(a.size() == 3);
  REQUIRE(a == b);
  REQUIRE(a[0] == 2);
  REQUIRE(a[1] == 2);
  REQUIRE(model.generateGreedy({0}, 1).size() == 1);
}

TEST_CASE("GQA groups query heads onto shared kv heads", "[text_model]") {
  ModelConfig c = tinyConfig();
  c.embeddingLength = 8;
  c.headCount = 4;
  c.headCountKv = 2;  // kvDim = 4
  c.feedForwardLength = 8;

  Weights w;
  w.tokenEmbd = WeightMat::f32(std::vector<float>(c.vocabSize * c.embeddingLength, 0.05f),
                               c.vocabSize, c.embeddingLength);
  LayerWeights L = zeroLayer(c);
  std::vector<float> wv(static_cast<std::size_t>(c.kvDim()) * c.embeddingLength);
  for (std::size_t i = 0; i < wv.size(); ++i) wv[i] = 0.01f * (i % 7);
  L.wv = WeightMat::f32(std::move(wv), c.kvDim(), c.embeddingLength);
  w.layers = {std::move(L)};
  w.outputNorm.assign(c.embeddingLength, 1.0f);
  w.output = WeightMat::f32(std::vector<float>(c.vocabSize * c.embeddingLength, 0.1f),
                            c.vocabSize, c.embeddingLength);

  TextModel model(c, std::move(w), 16);
  const auto& logits = model.forward(1, 0);
  REQUIRE(logits.size() == c.vocabSize);
  for (float v : logits) REQUIRE(std::isfinite(v));
}
