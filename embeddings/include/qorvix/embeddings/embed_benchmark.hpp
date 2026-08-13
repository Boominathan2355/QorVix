#pragma once

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include "qorvix/embeddings/embedding_engine.hpp"

// Throughput benchmark for the embedding seam — the encoder twin of runtime::runBenchmark.
//
// A sibling core rather than a parameter, because runBenchmark takes an IInferenceEngine& and its
// BenchResult is decode-shaped (decodeTokPerSec, ms/token). Two seams, two measurement cores, one
// CLI: `qorvix bench` dispatches on cfg.isEncoder(), so there is still one tool and one
// BENCHMARKS.md. Same warmup / median-of-runs discipline, so a number here means the same thing
// across commits.
namespace qorvix::embeddings {

struct EmbedBenchConfig {
  int seqTokens = 256;  // sequence length, INCLUDING [CLS]/[SEP]
  int batch = 1;        // sequences per timed run
  int warmupRuns = 1;
  int timedRuns = 3;
};

struct EmbedBenchResult {
  std::string backend;
  int seqTokens = 0;
  int batch = 0;
  int timedRuns = 0;
  bool ran = false;

  double loadSec = 0.0;  // filled by the caller

  // The headline numbers. embedTokPerSec is the one to compare against a decoder's prefill rate,
  // since both measure "how fast can this model consume input".
  double embedSeqPerSec = 0.0;
  double embedTokPerSec = 0.0;
  double msPerSeqMedian = 0.0;
  double msPerSeqMin = 0.0;
  double msPerSeqMax = 0.0;

  std::vector<double> runSeqPerSec;
};

namespace embed_bench_detail {
inline double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace embed_bench_detail

inline EmbedBenchResult runEmbedBenchmark(IEmbeddingEngine& engine, const EmbedBenchConfig& cfg) {
  using clock = std::chrono::steady_clock;

  EmbedBenchResult r;
  r.backend = engine.backendName();
  const int vocab = static_cast<int>(engine.config().vocabSize);
  const int maxSeq = static_cast<int>(engine.maxSeqLen());
  if (vocab <= 2 || maxSeq < 2) return r;  // ran stays false

  const int n = std::max(2, std::min(cfg.seqTokens, maxSeq));
  const int b = std::max(1, cfg.batch);
  r.seqTokens = n;
  r.batch = b;

  // Synthetic token ids spread across the vocabulary. Content does not affect cost — the encoder
  // does identical work for any ids — but spreading them avoids an unrealistically cache-friendly
  // embedding-table access pattern.
  std::vector<std::vector<int>> batch(static_cast<std::size_t>(b));
  for (int s = 0; s < b; ++s) {
    batch[s].resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) batch[s][i] = (i * 7919 + s * 104729) % vocab;
  }

  std::string err;
  std::vector<std::vector<float>> out;
  for (int w = 0; w < std::max(0, cfg.warmupRuns); ++w) {
    if (!engine.embedBatch(batch, out, err)) return r;
  }

  std::vector<double> msPerSeq;
  for (int t = 0; t < std::max(1, cfg.timedRuns); ++t) {
    const auto t0 = clock::now();
    if (!engine.embedBatch(batch, out, err)) return r;
    const double sec = std::chrono::duration<double>(clock::now() - t0).count();
    if (sec <= 0.0) continue;
    msPerSeq.push_back(sec * 1000.0 / b);
    r.runSeqPerSec.push_back(b / sec);
    ++r.timedRuns;
  }
  if (msPerSeq.empty()) return r;

  r.msPerSeqMedian = embed_bench_detail::median(msPerSeq);
  r.msPerSeqMin = *std::min_element(msPerSeq.begin(), msPerSeq.end());
  r.msPerSeqMax = *std::max_element(msPerSeq.begin(), msPerSeq.end());
  r.embedSeqPerSec = r.msPerSeqMedian > 0.0 ? 1000.0 / r.msPerSeqMedian : 0.0;
  r.embedTokPerSec = r.embedSeqPerSec * n;
  r.ran = true;
  return r;
}

}  // namespace qorvix::embeddings
