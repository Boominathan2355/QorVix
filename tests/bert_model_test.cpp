#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <string>
#include <vector>

#include "qorvix/embeddings/bert_model.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/pooling.hpp"

using namespace qorvix;
using namespace qorvix::embeddings;
using Catch::Matchers::WithinAbs;

namespace {

constexpr int kD = 4, kFfn = 8, kVocab = 8, kCtx = 16, kLayers = 2, kHeads = 2;

runtime::ModelConfig tinyConfig(runtime::PoolingType pooling = runtime::PoolingType::Cls) {
  runtime::ModelConfig cfg;
  cfg.architecture = "bert";
  cfg.family = runtime::ArchFamily::Encoder;
  cfg.vocabSize = kVocab;
  cfg.contextLength = kCtx;
  cfg.embeddingLength = kD;
  cfg.blockCount = kLayers;
  cfg.feedForwardLength = kFfn;
  cfg.headCount = kHeads;
  cfg.headCountKv = kHeads;
  cfg.normEpsilon = 1e-12f;
  cfg.causal = false;
  cfg.pooling = pooling;
  cfg.tokenTypeCount = 0;
  cfg.hasPositionEmbd = true;
  cfg.ffnGated = false;
  cfg.attnBias = false;
  cfg.postNorm = true;
  return cfg;
}

runtime::WeightMat zeros(int rows, int cols) {
  return runtime::WeightMat::f32(std::vector<float>(static_cast<std::size_t>(rows) * cols, 0.0f),
                                 rows, cols);
}

// A synthetic encoder whose weights are all zero except the token and position embedding tables.
// Zero projections mean attention scores are all zero, so softmax is uniform and attention
// degenerates to an average over V. The residual stream therefore carries the embeddings through
// apart from the LayerNorms, making pooling and normalization directly observable.
//
// The embedding rows must point in genuinely DIFFERENT DIRECTIONS, not merely have different
// magnitudes. LayerNorm is invariant to both scale and uniform shift, so a table like
// `row[t][i] = 0.1*(t+1)*(i+1)` — every row a positive multiple of one direction — normalizes
// every token to the identical vector, and any test that varies the input then compares outputs
// silently passes for the wrong reason. The modular pattern below gives each row a distinct sign
// pattern.
runtime::EncoderWeights zeroWeights() {
  runtime::EncoderWeights w;

  std::vector<float> emb(static_cast<std::size_t>(kVocab) * kD, 0.0f);
  for (int t = 0; t < kVocab; ++t) {
    for (int i = 0; i < kD; ++i) {
      emb[static_cast<std::size_t>(t) * kD + i] = static_cast<float>(((t + 1) * (i + 3)) % 7) - 3.0f;
    }
  }
  w.tokenEmbd = runtime::WeightMat::f32(std::move(emb), kVocab, kD);

  std::vector<float> pos(static_cast<std::size_t>(kCtx) * kD, 0.0f);
  for (int p = 0; p < kCtx; ++p) {
    for (int i = 0; i < kD; ++i) {
      pos[static_cast<std::size_t>(p) * kD + i] = 0.05f * static_cast<float>((p + i) % 3);
    }
  }
  w.positionEmbd = runtime::WeightMat::f32(std::move(pos), kCtx, kD);

  w.embdNorm.assign(kD, 1.0f);
  w.embdNormB.assign(kD, 0.0f);

  w.layers.resize(kLayers);
  for (auto& L : w.layers) {
    L.wq = zeros(kD, kD);
    L.wk = zeros(kD, kD);
    L.wv = zeros(kD, kD);
    L.wo = zeros(kD, kD);
    L.attnNorm.assign(kD, 1.0f);
    L.attnNormB.assign(kD, 0.0f);
    L.ffnUp = zeros(kFfn, kD);
    L.ffnDown = zeros(kD, kFfn);
    L.ffnNorm.assign(kD, 1.0f);
    L.ffnNormB.assign(kD, 0.0f);
  }
  return w;
}

BertModel tinyModel(runtime::PoolingType pooling = runtime::PoolingType::Cls) {
  return BertModel(tinyConfig(pooling), zeroWeights(), kCtx);
}

}  // namespace

TEST_CASE("a synthetic encoder produces a finite unit-norm embedding", "[embeddings]") {
  BertModel m = tinyModel();
  std::string err;
  std::vector<float> v;
  REQUIRE(m.embed({1, 2, 3}, v, err));
  REQUIRE(err.empty());
  REQUIRE(v.size() == kD);
  for (float x : v) REQUIRE(std::isfinite(x));
  REQUIRE_THAT(runtime::ops::l2Norm(v.data(), kD), WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("cls pooling takes the first token state and mean pooling the average", "[embeddings]") {
  // Same input, same weights, different pooling — the vectors must differ, which is what proves
  // pooling_type is honoured rather than one mode being hardcoded.
  const std::vector<int> tokens{1, 5, 2};
  std::string err;

  BertModel cls = tinyModel(runtime::PoolingType::Cls);
  BertModel mean = tinyModel(runtime::PoolingType::Mean);
  BertModel last = tinyModel(runtime::PoolingType::Last);

  std::vector<float> vCls, vMean, vLast, states;
  REQUIRE(cls.embedWith(tokens, runtime::PoolingType::Cls, false, vCls, err));
  REQUIRE(mean.embedWith(tokens, runtime::PoolingType::Mean, false, vMean, err));
  REQUIRE(last.embedWith(tokens, runtime::PoolingType::Last, false, vLast, err));
  REQUIRE(cls.embedTokens(tokens, states, err));
  REQUIRE(states.size() == tokens.size() * kD);

  // Each pooled vector must equal the corresponding reduction of the per-token states.
  for (int i = 0; i < kD; ++i) {
    REQUIRE_THAT(vCls[i], WithinAbs(states[i], 1e-5f));
    REQUIRE_THAT(vLast[i], WithinAbs(states[2 * kD + i], 1e-5f));
    const float avg = (states[i] + states[kD + i] + states[2 * kD + i]) / 3.0f;
    REQUIRE_THAT(vMean[i], WithinAbs(avg, 1e-5f));
  }
}

TEST_CASE("bidirectional attention lets the first token see the last", "[embeddings]") {
  // The single most valuable synthetic test in this phase. TextModel::attention bakes its causal
  // mask into the loop bound `t <= pos`; the encoder's bound is `t < n`. If that one character
  // were wrong, position 0 would attend only to itself, the [CLS] state would depend on nothing
  // but [CLS], and this vector would be bit-identical between the two inputs — while remaining
  // finite, unit-norm, and passing every other check here.
  //
  // Needs non-zero attention weights, so V is given real values while Q and K stay zero: scores
  // are then uniform and attention becomes a plain average over every V in the sequence.
  runtime::EncoderWeights w = zeroWeights();
  for (auto& L : w.layers) {
    std::vector<float> v(static_cast<std::size_t>(kD) * kD, 0.0f);
    for (int i = 0; i < kD; ++i) v[static_cast<std::size_t>(i) * kD + i] = 1.0f;  // identity
    L.wv = runtime::WeightMat::f32(std::move(v), kD, kD);
    std::vector<float> o(static_cast<std::size_t>(kD) * kD, 0.0f);
    for (int i = 0; i < kD; ++i) o[static_cast<std::size_t>(i) * kD + i] = 1.0f;
    L.wo = runtime::WeightMat::f32(std::move(o), kD, kD);
  }
  BertModel m(tinyConfig(runtime::PoolingType::Cls), std::move(w), kCtx);

  std::string err;
  std::vector<float> a, b;
  REQUIRE(m.embedWith({1, 2, 3}, runtime::PoolingType::Cls, false, a, err));
  REQUIRE(m.embedWith({1, 2, 7}, runtime::PoolingType::Cls, false, b, err));  // last token differs

  float maxDiff = 0.0f;
  for (int i = 0; i < kD; ++i) maxDiff = std::max(maxDiff, std::abs(a[i] - b[i]));
  REQUIRE(maxDiff > 1e-6f);
}

TEST_CASE("identical token sequences embed identically", "[embeddings]") {
  BertModel m = tinyModel();
  std::string err;
  std::vector<float> a, b;
  REQUIRE(m.embed({1, 2, 3}, a, err));
  REQUIRE(m.embed({1, 2, 3}, b, err));
  REQUIRE(a == b);  // bit-identical: the scratch buffers must not carry state between calls
}

TEST_CASE("normalization is applied only when requested", "[embeddings]") {
  BertModel m = tinyModel();
  std::string err;
  std::vector<float> raw, normed;
  REQUIRE(m.embedWith({1, 2, 3}, runtime::PoolingType::Cls, false, raw, err));
  REQUIRE(m.embedWith({1, 2, 3}, runtime::PoolingType::Cls, true, normed, err));

  const float n = runtime::ops::l2Norm(raw.data(), kD);
  REQUIRE(n > 0.0f);
  REQUIRE_THAT(runtime::ops::l2Norm(normed.data(), kD), WithinAbs(1.0f, 1e-5f));
  for (int i = 0; i < kD; ++i) REQUIRE_THAT(normed[i], WithinAbs(raw[i] / n, 1e-5f));
}

TEST_CASE("an empty or over-long sequence is rejected with an error", "[embeddings]") {
  BertModel m = tinyModel();
  std::string err;
  std::vector<float> v;

  REQUIRE_FALSE(m.embed({}, v, err));
  REQUIRE(err.find("empty") != std::string::npos);

  REQUIRE_FALSE(m.embed(std::vector<int>(kCtx + 1, 1), v, err));
  REQUIRE(err.find("exceeds") != std::string::npos);
}

TEST_CASE("a token id outside the vocabulary is rejected rather than read out of bounds",
          "[embeddings]") {
  BertModel m = tinyModel();
  std::string err;
  std::vector<float> v;
  REQUIRE_FALSE(m.embed({0, kVocab, 1}, v, err));
  REQUIRE(err.find("vocabulary") != std::string::npos);
  REQUIRE_FALSE(m.embed({-1}, v, err));
}

TEST_CASE("embedBatch returns one vector per input in request order", "[embeddings]") {
  BertModel m = tinyModel();
  std::string err;
  std::vector<std::vector<float>> out;
  const std::vector<std::vector<int>> batch{{1, 2}, {3, 4, 5}, {6}};
  REQUIRE(m.embedBatch(batch, out, err));
  REQUIRE(out.size() == 3);

  for (std::size_t i = 0; i < batch.size(); ++i) {
    std::vector<float> single;
    REQUIRE(m.embed(batch[i], single, err));
    REQUIRE(out[i] == single);
  }
}
