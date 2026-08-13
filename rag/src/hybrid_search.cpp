#include "qorvix/rag/hybrid_search.hpp"

#include <algorithm>
#include <unordered_map>

namespace qorvix::rag {

std::vector<SearchHit> fuseRrf(const std::vector<SearchHit>& dense,
                               const std::vector<SearchHit>& lexical, const HybridOptions& opt) {
  const float alpha = std::clamp(opt.alpha, 0.0f, 1.0f);
  const float k = static_cast<float>(opt.rrfK > 0 ? opt.rrfK : 60);

  std::unordered_map<std::size_t, float> fused;
  for (std::size_t r = 0; r < dense.size(); ++r) {
    fused[dense[r].index] += alpha / (k + static_cast<float>(r) + 1.0f);
  }
  for (std::size_t r = 0; r < lexical.size(); ++r) {
    fused[lexical[r].index] += (1.0f - alpha) / (k + static_cast<float>(r) + 1.0f);
  }

  std::vector<SearchHit> out;
  out.reserve(fused.size());
  for (const auto& [index, score] : fused) out.push_back({index, score});
  // Tiebreak on index: unordered_map iteration order is unspecified, so equal fused scores would
  // otherwise rank differently between runs.
  std::sort(out.begin(), out.end(), [](const SearchHit& a, const SearchHit& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;
  });
  if (opt.k > 0 && out.size() > static_cast<std::size_t>(opt.k)) {
    out.resize(static_cast<std::size_t>(opt.k));
  }
  return out;
}

std::vector<SearchHit> hybridSearch(const VectorStore& store, const Bm25Index& index,
                                    const std::vector<float>& queryVec, const std::string& query,
                                    const HybridOptions& opt) {
  const int pool = std::max(opt.k, opt.candidates);
  const float alpha = std::clamp(opt.alpha, 0.0f, 1.0f);

  // Skip an arm entirely when it carries no weight — running it would only cost time, and at
  // alpha 0 or 1 the caller has explicitly asked for one retrieval mode.
  std::vector<SearchHit> dense;
  if (alpha > 0.0f) dense = store.searchDense(queryVec, pool);
  std::vector<SearchHit> lexical;
  if (alpha < 1.0f) lexical = index.searchText(query, pool);

  return fuseRrf(dense, lexical, opt);
}

}  // namespace qorvix::rag
