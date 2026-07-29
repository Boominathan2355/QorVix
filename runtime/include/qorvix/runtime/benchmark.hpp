#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "qorvix/runtime/inference_engine.hpp"

// Backend-agnostic throughput benchmark. Drives ANY runtime::IInferenceEngine (CPU / CUDA / Vulkan)
// through the seam — prefill P tokens, then decode G tokens — with warmup runs discarded and several
// timed runs reduced to a median, so a number here means the same thing on every backend and is
// stable enough to compare across commits. This is the single measurement core: `qorvix bench` and
// the performance-regression tests both call runBenchmark(), so the CLI and CI can never drift.
namespace qorvix::runtime {

struct BenchConfig {
  int promptTokens = 64;  // prefill length (positions 0..P-1)
  int genTokens = 64;     // decode steps timed separately
  int warmupRuns = 1;     // discarded (page-in, clocks spin up, caches warm)
  int timedRuns = 3;      // measured; reported as median
};

struct BenchResult {
  std::string backend;
  int promptTokens = 0;
  int genTokens = 0;
  int timedRuns = 0;
  bool ran = false;

  double loadSec = 0.0;  // engine build/load time — filled in by the caller, not runBenchmark

  // Prefill: throughput of the prompt pass (tokens / prefill seconds), median across timed runs.
  double prefillTokPerSec = 0.0;
  // Decode: the headline number — steady-state per-token generation throughput.
  double decodeTokPerSec = 0.0;       // median
  double decodeMsPerTokMedian = 0.0;  // median ms / decoded token
  double decodeMsPerTokMin = 0.0;
  double decodeMsPerTokMax = 0.0;

  std::vector<double> decodeRunTokPerSec;  // one entry per timed run (for spread / debugging)
};

namespace bench_detail {
inline double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
}  // namespace bench_detail

inline BenchResult runBenchmark(IInferenceEngine& engine, const BenchConfig& cfg) {
  using clock = std::chrono::steady_clock;
  auto secs = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };

  BenchResult r;
  r.backend = engine.backendName();
  const int vocab = static_cast<int>(engine.config().vocabSize);
  const int maxSeq = static_cast<int>(engine.maxSeqLen());
  if (vocab <= 0 || maxSeq <= 1) return r;  // ran stays false

  // Clamp the workload to the engine's context so we never index past maxSeq.
  const int P = std::max(1, std::min(cfg.promptTokens, maxSeq - 1));
  const int G = std::max(0, std::min(cfg.genTokens, maxSeq - P));
  r.promptTokens = P;
  r.genTokens = G;
  r.timedRuns = cfg.timedRuns;

  // Deterministic, backend-independent token stream (content is irrelevant to timing, but keeping it
  // fixed makes runs reproducible). Spread across the vocab rather than a single hot token.
  auto tokenAt = [vocab](int i) { return static_cast<int>((static_cast<unsigned>(i) * 2654435761u) % static_cast<unsigned>(vocab)); };

  std::vector<double> prefillRates, decodeRates, msPerTok;
  const int totalRuns = cfg.warmupRuns + cfg.timedRuns;
  for (int run = 0; run < totalRuns; ++run) {
    const auto session = engine.openSession();
    if (session == memory::kInvalidSession) return r;  // engine at capacity / no slot

    int pos = 0;
    const auto tPrefill0 = clock::now();
    for (int i = 0; i < P && pos < maxSeq; ++i, ++pos) engine.forward(session, tokenAt(i), pos);
    const double prefillSec = secs(tPrefill0, clock::now());

    int decoded = 0;
    const auto tDecode0 = clock::now();
    for (int i = 0; i < G && pos < maxSeq; ++i, ++pos) {
      engine.forward(session, tokenAt(P + i), pos);
      ++decoded;
    }
    const double decodeSec = secs(tDecode0, clock::now());

    engine.closeSession(session);

    if (run >= cfg.warmupRuns) {
      prefillRates.push_back(prefillSec > 0 ? P / prefillSec : 0.0);
      decodeRates.push_back(decoded > 0 && decodeSec > 0 ? decoded / decodeSec : 0.0);
      msPerTok.push_back(decoded > 0 ? 1000.0 * decodeSec / decoded : 0.0);
    }
  }

  r.decodeRunTokPerSec = decodeRates;
  r.prefillTokPerSec = bench_detail::median(prefillRates);
  r.decodeTokPerSec = bench_detail::median(decodeRates);
  r.decodeMsPerTokMedian = bench_detail::median(msPerTok);
  if (!msPerTok.empty()) {
    r.decodeMsPerTokMin = *std::min_element(msPerTok.begin(), msPerTok.end());
    r.decodeMsPerTokMax = *std::max_element(msPerTok.begin(), msPerTok.end());
  }
  r.ran = true;
  return r;
}

}  // namespace qorvix::runtime
