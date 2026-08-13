#include "qorvix/rag/bm25.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>

namespace qorvix::rag {

namespace {

void putU32(std::ostream& out, std::uint32_t v) {
  char b[4];
  for (int i = 0; i < 4; ++i) b[i] = static_cast<char>((v >> (8 * i)) & 0xFF);
  out.write(b, 4);
}
bool getU32(std::istream& in, std::uint32_t& v) {
  char b[4];
  if (!in.read(b, 4)) return false;
  v = 0;
  for (int i = 0; i < 4; ++i)
    v |= static_cast<std::uint32_t>(static_cast<unsigned char>(b[i])) << (8 * i);
  return true;
}
void putStr(std::ostream& out, const std::string& s) {
  putU32(out, static_cast<std::uint32_t>(s.size()));
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
}
bool getStr(std::istream& in, std::string& s) {
  std::uint32_t n = 0;
  if (!getU32(in, n) || n > 4096) return false;  // a single term is never this long
  s.resize(n);
  return n == 0 || static_cast<bool>(in.read(s.data(), n));
}

}  // namespace

std::vector<std::string> bm25Terms(const std::string& text) {
  std::vector<std::string> terms;
  std::string cur;
  for (unsigned char c : text) {
    if (std::isalnum(c)) {
      cur.push_back(static_cast<char>(std::tolower(c)));
    } else if (c >= 0x80) {
      // Keep multibyte UTF-8 bytes in the term rather than splitting on them; without this every
      // accented or CJK word fragments into meaningless pieces.
      cur.push_back(static_cast<char>(c));
    } else if (!cur.empty()) {
      terms.push_back(cur);
      cur.clear();
    }
  }
  if (!cur.empty()) terms.push_back(cur);
  return terms;
}

void Bm25Index::add(const std::vector<std::string>& terms) {
  const auto doc = static_cast<std::uint32_t>(docLengths_.size());
  std::unordered_map<std::string, std::uint32_t> tf;
  for (const auto& t : terms) ++tf[t];
  for (const auto& [term, freq] : tf) postings_[term].emplace_back(doc, freq);
  docLengths_.push_back(static_cast<std::uint32_t>(terms.size()));
  finalized_ = false;  // a stale idf must not be searchable
}

void Bm25Index::finalize() {
  const auto n = static_cast<float>(docLengths_.size());
  double total = 0.0;
  for (std::uint32_t l : docLengths_) total += l;
  avgDocLength_ = docLengths_.empty() ? 0.0f : static_cast<float>(total / docLengths_.size());

  idf_.clear();
  idf_.reserve(postings_.size());
  for (const auto& [term, list] : postings_) {
    const auto df = static_cast<float>(list.size());
    // Robertson/Sparck-Jones idf with the +1 that keeps it non-negative. Without the +1 a term
    // appearing in more than half the corpus scores NEGATIVE, so a document containing the query
    // term ranks below one that does not.
    idf_[term] = std::log(1.0f + (n - df + 0.5f) / (df + 0.5f));
  }
  finalized_ = true;
}

std::vector<SearchHit> Bm25Index::search(const std::vector<std::string>& queryTerms, int k) const {
  std::vector<SearchHit> hits;
  if (k <= 0 || docLengths_.empty() || !finalized_) return hits;

  std::unordered_map<std::uint32_t, float> scores;
  for (const auto& term : queryTerms) {
    const auto p = postings_.find(term);
    if (p == postings_.end()) continue;
    const auto w = idf_.find(term);
    if (w == idf_.end()) continue;
    for (const auto& [doc, freq] : p->second) {
      const float f = static_cast<float>(freq);
      const float len = static_cast<float>(docLengths_[doc]);
      const float norm = avgDocLength_ > 0.0f ? len / avgDocLength_ : 1.0f;
      scores[doc] += w->second * (f * (kK1 + 1.0f)) / (f + kK1 * (1.0f - kB + kB * norm));
    }
  }

  hits.reserve(scores.size());
  for (const auto& [doc, score] : scores) hits.push_back({doc, score});
  // Sort by score, then by index: unordered_map iteration order is unspecified, so without the
  // tiebreak two runs over the same data could return different orderings.
  std::sort(hits.begin(), hits.end(), [](const SearchHit& a, const SearchHit& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;
  });
  if (hits.size() > static_cast<std::size_t>(k)) hits.resize(static_cast<std::size_t>(k));
  return hits;
}

bool Bm25Index::save(std::ostream& out) const {
  putU32(out, static_cast<std::uint32_t>(docLengths_.size()));
  for (std::uint32_t l : docLengths_) putU32(out, l);
  putU32(out, static_cast<std::uint32_t>(postings_.size()));
  // Sorted term order so the file is byte-identical for identical input, despite the hash map.
  std::vector<const std::string*> terms;
  terms.reserve(postings_.size());
  for (const auto& [term, _] : postings_) terms.push_back(&term);
  std::sort(terms.begin(), terms.end(),
            [](const std::string* a, const std::string* b) { return *a < *b; });
  for (const std::string* term : terms) {
    const auto& list = postings_.at(*term);
    putStr(out, *term);
    putU32(out, static_cast<std::uint32_t>(list.size()));
    for (const auto& [doc, freq] : list) {
      putU32(out, doc);
      putU32(out, freq);
    }
  }
  return static_cast<bool>(out);
}

bool Bm25Index::load(std::istream& in, Bm25Index& out) {
  out = Bm25Index{};
  std::uint32_t nDocs = 0;
  if (!getU32(in, nDocs)) return false;
  out.docLengths_.resize(nDocs);
  for (std::uint32_t i = 0; i < nDocs; ++i) {
    if (!getU32(in, out.docLengths_[i])) return false;
  }
  std::uint32_t nTerms = 0;
  if (!getU32(in, nTerms)) return false;
  for (std::uint32_t i = 0; i < nTerms; ++i) {
    std::string term;
    std::uint32_t listLen = 0;
    if (!getStr(in, term) || !getU32(in, listLen)) return false;
    auto& list = out.postings_[term];
    list.reserve(listLen);
    for (std::uint32_t j = 0; j < listLen; ++j) {
      std::uint32_t doc = 0, freq = 0;
      if (!getU32(in, doc) || !getU32(in, freq)) return false;
      if (doc >= nDocs) return false;  // a corrupt posting must not index out of range later
      list.emplace_back(doc, freq);
    }
  }
  out.finalize();
  return true;
}

}  // namespace qorvix::rag
