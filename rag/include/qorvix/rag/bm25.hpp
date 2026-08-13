#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "qorvix/rag/document.hpp"

namespace qorvix::rag {

// Whole-word terms: lowercased, split on non-alphanumeric.
//
// Deliberately NOT the WordPiece tokenizer. BM25's whole premise is term-document statistics, and
// "##ation" is not a term anyone searches for — splitting a rare word into common subwords would
// hand it the idf of its pieces instead of its own, which is exactly backwards for the lexical
// arm. The two retrieval arms wanting different tokenizations is the point of running both.
std::vector<std::string> bm25Terms(const std::string& text);

// Okapi BM25 with the standard Robertson parameters.
class Bm25Index {
 public:
  // Adds one document's terms. Order matters only in that hit indices follow insertion order,
  // matching VectorStore so the two arms can be fused by index.
  void add(const std::vector<std::string>& terms);
  void addText(const std::string& text) { add(bm25Terms(text)); }

  // Computes idf and the average document length. Must be called after the last add() and before
  // search(); adding afterwards invalidates it (add() clears the flag, so a stale index cannot be
  // searched by accident).
  void finalize();

  std::vector<SearchHit> search(const std::vector<std::string>& queryTerms, int k) const;
  std::vector<SearchHit> searchText(const std::string& query, int k) const {
    return search(bm25Terms(query), k);
  }

  std::size_t size() const noexcept { return docLengths_.size(); }
  bool finalized() const noexcept { return finalized_; }

  bool save(std::ostream& out) const;
  static bool load(std::istream& in, Bm25Index& out);

  static constexpr float kK1 = 1.2f;
  static constexpr float kB = 0.75f;

 private:
  // term -> (doc index, term frequency). A posting list, so scoring touches only documents that
  // actually contain a query term instead of every document.
  std::unordered_map<std::string, std::vector<std::pair<std::uint32_t, std::uint32_t>>> postings_;
  std::unordered_map<std::string, float> idf_;
  std::vector<std::uint32_t> docLengths_;
  float avgDocLength_ = 0.0f;
  bool finalized_ = false;
};

}  // namespace qorvix::rag
