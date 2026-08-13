#pragma once

#include <string>
#include <vector>

#include "qorvix/rag/bm25.hpp"
#include "qorvix/rag/document.hpp"
#include "qorvix/rag/vector_store.hpp"

namespace qorvix::rag {

struct HybridOptions {
  int k = 5;             // results to return
  float alpha = 0.5f;    // 1.0 = dense only, 0.0 = lexical only
  int candidates = 50;   // per-arm pool size, fused down to k
  int rrfK = 60;         // RRF smoothing constant
};

// Fuses two ranked lists by Reciprocal Rank Fusion: score = sum over arms of weight / (K + rank).
//
// RRF rather than min-max normalization, deliberately. The two arms produce incomparable
// quantities — a cosine bounded in [-1, 1] and an unbounded BM25 score — so combining them by
// value needs a calibration that does not exist. Min-max normalization is the obvious
// alternative and is unstable exactly when it matters: with a small or degenerate candidate pool
// (every score equal, or a single hit) the normalizer collapses and the fused ranking becomes
// arbitrary. RRF uses only RANK, so it has nothing to calibrate.
std::vector<SearchHit> fuseRrf(const std::vector<SearchHit>& dense,
                               const std::vector<SearchHit>& lexical, const HybridOptions& opt);

// Runs both arms and fuses them. `queryVec` must already be embedded and normalized the same way
// the stored vectors were.
std::vector<SearchHit> hybridSearch(const VectorStore& store, const Bm25Index& index,
                                    const std::vector<float>& queryVec, const std::string& query,
                                    const HybridOptions& opt = {});

}  // namespace qorvix::rag
