#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "qorvix/runtime/generator.hpp"
#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/runtime/weights.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

using namespace qorvix::runtime;
using qorvix::tokenizer::SpecialTokens;
using qorvix::tokenizer::Tokenizer;
using qorvix::tokenizer::TokenizerModel;

namespace {

ModelConfig cfg4() {  // d=4, 2 heads, MHA, 1 layer, vocab=4
  ModelConfig c;
  c.architecture = "llama";
  c.vocabSize = 4;
  c.contextLength = 32;
  c.embeddingLength = 4;
  c.blockCount = 1;
  c.feedForwardLength = 4;
  c.headCount = 2;
  c.headCountKv = 2;
  c.ropeDimensionCount = 2;
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

// Builds a model whose LM head always makes `winner` the argmax (its output row dominates).
TextModel modelPredicting(int winner) {
  ModelConfig c = cfg4();
  const int d = c.embeddingLength, vocab = c.vocabSize;
  Weights w;
  w.tokenEmbd = WeightMat::f32(std::vector<float>(vocab * d, 0.5f), vocab, d);
  w.layers = {zeroLayer(c)};
  w.outputNorm.assign(d, 1.0f);
  std::vector<float> out(vocab * d, 0.0f);
  for (int j = 0; j < d; ++j) out[winner * d + j] = 10.0f;
  w.output = WeightMat::f32(std::move(out), vocab, d);
  return TextModel(c, std::move(w), /*maxSeqLen=*/32);
}

// Vocab: 0=<s>, 1=</s>, 2="A", 3="B"; byte-level BPE so single letters map to themselves.
Tokenizer toyTok() {
  SpecialTokens sp;
  sp.bos = 0;
  sp.eos = 1;
  return Tokenizer(TokenizerModel::Bpe, {"<s>", "</s>", "A", "B"}, {}, {}, sp);
}

}  // namespace

TEST_CASE("greedy generation streams a deterministic sequence", "[generator]") {
  TextModel model = modelPredicting(2);  // always predicts "A"
  Tokenizer tok = toyTok();
  Generator gen(model, tok);

  GenerationConfig cfg;
  cfg.maxNewTokens = 3;
  cfg.sampling.temperature = 0.0f;  // greedy
  cfg.addBos = false;

  std::string streamed;
  auto result = gen.generate("A", cfg, [&](const std::string& p) { streamed += p; });

  REQUIRE(result.tokens == std::vector<int>{2, 2, 2});
  REQUIRE(result.text == "AAA");
  REQUIRE(streamed == result.text);   // streaming matches the final text
  REQUIRE_FALSE(result.hitEos);
  REQUIRE(result.promptTokens == 1);  // "A" -> one token, no BOS
}

TEST_CASE("generation stops on EOS", "[generator]") {
  TextModel model = modelPredicting(1);  // always predicts </s>
  Tokenizer tok = toyTok();
  Generator gen(model, tok);

  GenerationConfig cfg;
  cfg.maxNewTokens = 10;
  cfg.sampling.temperature = 0.0f;
  cfg.addBos = false;

  auto result = gen.generate("A", cfg);
  REQUIRE(result.hitEos);
  REQUIRE(result.tokens.empty());
  REQUIRE(result.text.empty());
}

TEST_CASE("maxNewTokens caps the output", "[generator]") {
  TextModel model = modelPredicting(3);  // predicts "B" forever
  Tokenizer tok = toyTok();
  Generator gen(model, tok);

  GenerationConfig cfg;
  cfg.maxNewTokens = 5;
  cfg.sampling.temperature = 0.0f;
  cfg.addBos = true;  // exercise BOS handling too

  auto result = gen.generate("A", cfg);
  REQUIRE(result.tokens.size() == 5);
  REQUIRE(result.text == "BBBBB");
  REQUIRE(result.promptTokens == 2);  // <s> + "A"
}
