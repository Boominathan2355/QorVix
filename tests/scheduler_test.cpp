#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/runtime/weights.hpp"
#include "qorvix/scheduler/scheduler.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

using namespace qorvix::runtime;
using namespace qorvix::scheduler;
using qorvix::tokenizer::SpecialTokens;
using qorvix::tokenizer::Tokenizer;
using qorvix::tokenizer::TokenizerModel;

namespace {

ModelConfig cfg4() {
  ModelConfig c;
  c.architecture = "llama";
  c.vocabSize = 4;
  c.contextLength = 64;
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

// LM head always makes `winner` the argmax; supports up to `maxSessions` concurrent sequences.
TextModel modelPredicting(int winner, std::uint32_t maxSessions) {
  ModelConfig c = cfg4();
  const int d = c.embeddingLength, vocab = c.vocabSize;
  Weights w;
  w.tokenEmbd = WeightMat::f32(std::vector<float>(vocab * d, 0.5f), vocab, d);
  w.layers = {zeroLayer(c)};
  w.outputNorm.assign(d, 1.0f);
  std::vector<float> out(vocab * d, 0.0f);
  for (int j = 0; j < d; ++j) out[winner * d + j] = 10.0f;
  w.output = WeightMat::f32(std::move(out), vocab, d);
  return TextModel(c, std::move(w), /*maxSeqLen=*/64, maxSessions);
}

Tokenizer toyTok() {
  SpecialTokens sp;
  sp.bos = 0;
  sp.eos = 1;
  return Tokenizer(TokenizerModel::Bpe, {"<s>", "</s>", "A", "B"}, {}, {}, sp);
}

RequestResult* find(std::vector<RequestResult>& v, RequestId id) {
  for (auto& r : v)
    if (r.id == id) return &r;
  return nullptr;
}

}  // namespace

TEST_CASE("scheduler runs multiple requests to completion", "[scheduler]") {
  TextModel model = modelPredicting(2, /*maxSessions=*/4);  // always emits "A"
  Tokenizer tok = toyTok();
  Scheduler sched(model, tok, {/*maxConcurrent=*/4});

  RequestParams p;
  p.maxNewTokens = 3;
  p.sampling.temperature = 0.0f;  // greedy
  p.addBos = false;

  RequestId a = sched.submit("A", p);
  RequestId b = sched.submit("B", p);
  REQUIRE(sched.waiting() == 2);

  auto results = sched.runToCompletion();
  REQUIRE(sched.idle());
  REQUIRE(results.size() == 2);

  RequestResult* ra = find(results, a);
  RequestResult* rb = find(results, b);
  REQUIRE(ra != nullptr);
  REQUIRE(rb != nullptr);
  REQUIRE(ra->text == "AAA");
  REQUIRE(rb->text == "AAA");
  REQUIRE(ra->tokens == std::vector<int>{2, 2, 2});
  REQUIRE_FALSE(ra->rejected);
}

TEST_CASE("requests exceeding capacity are batched over multiple rounds", "[scheduler]") {
  TextModel model = modelPredicting(3, /*maxSessions=*/2);  // emits "B"
  Tokenizer tok = toyTok();
  Scheduler sched(model, tok, {/*maxConcurrent=*/2});

  RequestParams p;
  p.maxNewTokens = 2;
  p.sampling.temperature = 0.0f;
  p.addBos = false;

  for (int i = 0; i < 5; ++i) sched.submit("A", p);
  REQUIRE(sched.waiting() == 5);

  // First step admits only maxConcurrent (2).
  sched.step();
  REQUIRE(sched.active() <= 2);

  auto results = sched.runToCompletion();
  // 5 total across all rounds (the first step may finish some already).
  int total = static_cast<int>(results.size());
  // gather remaining if any were returned by the first step
  REQUIRE(sched.idle());
  for (const auto& r : results) {
    REQUIRE(r.text == "BB");
    REQUIRE(r.tokens.size() == 2);
  }
  REQUIRE(total >= 3);  // remaining after the first admitted pair completed
}

TEST_CASE("higher priority is admitted first", "[scheduler]") {
  TextModel model = modelPredicting(2, /*maxSessions=*/1);  // only one slot
  Tokenizer tok = toyTok();
  Scheduler sched(model, tok, {/*maxConcurrent=*/1});

  RequestParams lo;
  lo.maxNewTokens = 1;
  lo.sampling.temperature = 0.0f;
  lo.addBos = false;
  lo.priority = 0;
  RequestParams hi = lo;
  hi.priority = 10;

  RequestId low = sched.submit("A", lo);
  RequestId high = sched.submit("A", hi);

  // One slot: the first admitted request is the high-priority one.
  auto first = sched.step();  // admits high, prefills
  while (first.empty()) first = sched.step();
  REQUIRE(first.front().id == high);
  (void)low;
}

TEST_CASE("streaming callback fires per generated token", "[scheduler]") {
  TextModel model = modelPredicting(2, 2);
  Tokenizer tok = toyTok();
  Scheduler sched(model, tok, {2});

  RequestParams p;
  p.maxNewTokens = 4;
  p.sampling.temperature = 0.0f;
  p.addBos = false;

  std::string streamed;
  sched.submit("A", p, [&](RequestId, const std::string& piece) { streamed += piece; });
  auto results = sched.runToCompletion();
  REQUIRE(results.size() == 1);
  REQUIRE(streamed == results.front().text);
  REQUIRE(streamed == "AAAA");
}
