#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "qorvix/runtime/generator.hpp"
#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/text_model.hpp"
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

// Builds a model whose LM head always makes `winner` the argmax (its output row dominates).
TextModel modelPredicting(int winner) {
  ModelConfig c = cfg4();
  Weights w;
  w.tokenEmbd.assign(c.vocabSize * c.embeddingLength, 0.5f);  // all-positive embeddings
  w.layers = {zeroLayer(c)};
  w.outputNorm.assign(c.embeddingLength, 1.0f);
  w.output.assign(c.vocabSize * c.embeddingLength, 0.0f);
  for (int j = 0; j < static_cast<int>(c.embeddingLength); ++j)
    w.output[winner * c.embeddingLength + j] = 10.0f;
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
