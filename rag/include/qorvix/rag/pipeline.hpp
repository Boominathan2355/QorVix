#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "qorvix/embeddings/embedding_engine.hpp"
#include "qorvix/rag/bm25.hpp"
#include "qorvix/rag/chunker.hpp"
#include "qorvix/rag/document.hpp"
#include "qorvix/rag/hybrid_search.hpp"
#include "qorvix/rag/vector_store.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

namespace qorvix::rag {

// One index: the dense store and the lexical index, kept in step so a hit index means the same
// chunk in both. They are saved to one file so the two halves can never drift apart on disk.
struct Index {
  VectorStore store;
  Bm25Index lexical;

  bool save(const std::filesystem::path& path, std::string& error) const;
  static bool load(const std::filesystem::path& path, Index& out, std::string& error);
};

struct IndexStats {
  int documents = 0;
  int chunks = 0;
  int tokens = 0;
  std::vector<std::string> skipped;
};

// Loads, chunks, embeds and indexes every supported file under `path`.
//
// `progress` (optional) is called once per document with (done, total, source) so a CLI can show
// movement — indexing a directory is minutes of work on the CPU encoder, and a silent process
// looks hung.
bool buildIndex(const std::filesystem::path& path, embeddings::IEmbeddingEngine& engine,
                const tokenizer::Tokenizer& tok, const ChunkOptions& chunkOpt, Index& out,
                IndexStats& stats, std::string& error,
                const std::function<void(int, int, const std::string&)>& progress = {});

// Embeds `query` and runs hybrid retrieval over an existing index.
bool queryIndex(const Index& index, embeddings::IEmbeddingEngine& engine,
                const tokenizer::Tokenizer& tok, const std::string& query,
                const HybridOptions& opt, std::vector<SearchHit>& hits, std::string& error);

}  // namespace qorvix::rag
