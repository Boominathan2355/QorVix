#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "qorvix/runtime/benchmark.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/text_model.hpp"
#include "qorvix/runtime/weights.hpp"

// Performance-regression tests. These drive the SAME runBenchmark() that `qorvix bench` uses, on a
// small synthetic CPU model, and assert:
//   1. Structural correctness — the harness actually ran, produced the expected number of timed
//      runs, and its median lies within [min,max]. If this breaks, the benchmark itself is wrong and
//      every perf number it prints is suspect.
//   2. A *catastrophic*-regression floor on decode throughput. It is intentionally very loose (a
//      tiny model on CPU does thousands of forwards/sec; the floor is a few dozen), so it never
//      flakes on normal CI noise or a loaded runner — it only trips on an order-of-magnitude
//      slowdown, exactly the class of bug worth failing the build for (e.g. the nested-OpenMP
//      regression that once made the CPU forward ~30x slower). Tight, machine-specific numbers are
//      the job of `qorvix bench` on real hardware, not of a portable unit test.
using namespace qorvix::runtime;

namespace {

ModelConfig benchConfig() {
  ModelConfig c;
  c.architecture = "llama";
  c.vocabSize = 256;
  c.contextLength = 128;
  c.embeddingLength = 64;  // d_model
  c.blockCount = 2;
  c.feedForwardLength = 128;
  c.headCount = 4;      // head_dim = 16
  c.headCountKv = 2;    // GQA
  c.ropeDimensionCount = 16;
  c.ropeFreqBase = 10000.0f;
  c.normEpsilon = 1e-5f;
  c.ropeMode = ops::RopeMode::Neox;
  return c;
}

// Small deterministic non-zero fill so the forward does real arithmetic (all-zero weights would
// short-circuit the residual branches and under-measure).
std::vector<float> filled(std::size_t n, int seed) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i)
    v[i] = 0.01f * static_cast<float>((static_cast<int>(i) * 7 + seed * 13) % 17 - 8);
  return v;
}

WeightMat mat(int rows, int cols, int seed) {
  return WeightMat::f32(filled(static_cast<std::size_t>(rows) * cols, seed), rows, cols);
}

TextModel makeModel(const ModelConfig& c, std::uint32_t maxSeq) {
  const int d = static_cast<int>(c.embeddingLength), kv = c.kvDim(),
            ffn = static_cast<int>(c.feedForwardLength), vocab = static_cast<int>(c.vocabSize);
  Weights w;
  w.tokenEmbd = mat(vocab, d, 1);
  w.outputNorm = filled(d, 2);
  for (auto& x : w.outputNorm) x += 1.0f;  // norms centered near 1
  w.output = mat(vocab, d, 3);
  w.layers.resize(c.blockCount);
  for (std::uint32_t l = 0; l < c.blockCount; ++l) {
    auto& L = w.layers[l];
    L.attnNorm = filled(d, 10 + l);
    L.ffnNorm = filled(d, 20 + l);
    for (auto& x : L.attnNorm) x += 1.0f;
    for (auto& x : L.ffnNorm) x += 1.0f;
    L.wq = mat(d, d, 30 + l);
    L.wk = mat(kv, d, 40 + l);
    L.wv = mat(kv, d, 50 + l);
    L.wo = mat(d, d, 60 + l);
    L.ffnGate = mat(ffn, d, 70 + l);
    L.ffnUp = mat(ffn, d, 80 + l);
    L.ffnDown = mat(d, ffn, 90 + l);
  }
  return TextModel(c, std::move(w), maxSeq);
}

}  // namespace

TEST_CASE("benchmark harness runs and reports a consistent structure", "[benchmark]") {
  const auto cfg = benchConfig();
  TextModel model = makeModel(cfg, 128);

  BenchConfig bc;
  bc.promptTokens = 16;
  bc.genTokens = 32;
  bc.warmupRuns = 1;
  bc.timedRuns = 3;
  const BenchResult r = runBenchmark(model, bc);

  REQUIRE(r.ran);
  REQUIRE(r.backend == "cpu");
  REQUIRE(r.promptTokens == 16);
  REQUIRE(r.genTokens == 32);
  REQUIRE(r.timedRuns == 3);
  REQUIRE(r.decodeRunTokPerSec.size() == 3);

  // Every rate is positive and finite; the median sits within the observed spread.
  REQUIRE(r.prefillTokPerSec > 0.0);
  REQUIRE(r.decodeTokPerSec > 0.0);
  REQUIRE(r.decodeMsPerTokMedian > 0.0);
  REQUIRE(r.decodeMsPerTokMin <= r.decodeMsPerTokMedian + 1e-9);
  REQUIRE(r.decodeMsPerTokMedian <= r.decodeMsPerTokMax + 1e-9);
}

TEST_CASE("benchmark clamps the workload to the engine context", "[benchmark]") {
  const auto cfg = benchConfig();
  TextModel model = makeModel(cfg, 64);  // maxSeq = 64

  BenchConfig bc;
  bc.promptTokens = 1000;  // absurd — must be clamped to < maxSeq
  bc.genTokens = 1000;
  bc.warmupRuns = 0;
  bc.timedRuns = 1;
  const BenchResult r = runBenchmark(model, bc);

  REQUIRE(r.ran);
  REQUIRE(r.promptTokens < 64);
  REQUIRE(r.promptTokens + r.genTokens <= 64);
}

TEST_CASE("decode throughput clears the catastrophic-regression floor", "[benchmark]") {
  const auto cfg = benchConfig();
  TextModel model = makeModel(cfg, 128);

  BenchConfig bc;
  bc.promptTokens = 8;
  bc.genTokens = 24;
  bc.warmupRuns = 1;
  bc.timedRuns = 3;
  const BenchResult r = runBenchmark(model, bc);

  REQUIRE(r.ran);
  // Deliberately loose: a 2-layer d=64 model on CPU does hundreds+ decode tok/s even on a slow
  // shared runner. A failure here means an order-of-magnitude regression, not measurement noise.
  const double kFloorTokPerSec = 20.0;
  INFO("decode tok/s = " << r.decodeTokPerSec << " (floor " << kFloorTokPerSec << ")");
  REQUIRE(r.decodeTokPerSec > kFloorTokPerSec);
}
