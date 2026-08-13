#include "qorvix/rag/pipeline.hpp"

#include <algorithm>
#include <fstream>

#include "qorvix/rag/loaders.hpp"

namespace fs = std::filesystem;

namespace qorvix::rag {

namespace {

// Truncates to the encoder's limit while keeping the closing [SEP]. Dropping the tail outright
// would leave the sequence unterminated, which shifts the vector on a CLS-pooled model.
std::vector<int> fitToEncoder(std::vector<int> ids, int cap, int sepId) {
  if (static_cast<int>(ids.size()) <= cap) return ids;
  ids.resize(cap);
  if (sepId >= 0) ids.back() = sepId;
  return ids;
}

constexpr std::uint32_t kIndexMagic = 0x58444951;  // "QIDX"
constexpr std::uint32_t kIndexVersion = 1;

}  // namespace

bool Index::save(const fs::path& path, std::string& error) const {
  // The vector store owns the file format; the lexical index is appended after it, so the two
  // halves cannot be separated and cannot disagree about how many chunks exist.
  if (!store.save(path, error)) return false;
  std::ofstream out(path, std::ios::binary | std::ios::app);
  if (!out) {
    error = "cannot reopen '" + path.string() + "' to append the lexical index";
    return false;
  }
  char b[4];
  auto putU32 = [&](std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b[i] = static_cast<char>((v >> (8 * i)) & 0xFF);
    out.write(b, 4);
  };
  putU32(kIndexMagic);
  putU32(kIndexVersion);
  if (!lexical.save(out)) {
    error = "failed writing the lexical index to '" + path.string() + "'";
    return false;
  }
  return true;
}

bool Index::load(const fs::path& path, Index& out, std::string& error) {
  auto store = VectorStore::load(path, error);
  if (!store) return false;
  out.store = std::move(*store);

  // Re-open and skip to the lexical block. VectorStore::load stops exactly at its own end, but it
  // does not report the offset, so scan for the trailer magic from the end instead of guessing.
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot reopen '" + path.string() + "'";
    return false;
  }
  std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const char magic[4] = {0x51, 0x49, 0x44, 0x58};  // "QIDX" little-endian
  const auto pos = all.rfind(std::string(magic, 4));
  if (pos == std::string::npos) {
    // A store written by `qorvix rag index` always has one. Its absence means the file predates
    // the lexical half or was truncated — rebuild rather than search with a dense-only index that
    // silently returns worse results.
    error = "'" + path.string() + "' has no lexical index — rebuild it with `qorvix rag index`";
    return false;
  }
  std::istringstream tail(all.substr(pos + 4));
  std::uint32_t version = 0;
  char vb[4];
  if (!tail.read(vb, 4)) {
    error = "'" + path.string() + "' is truncated at the lexical index";
    return false;
  }
  for (int i = 0; i < 4; ++i)
    version |= static_cast<std::uint32_t>(static_cast<unsigned char>(vb[i])) << (8 * i);
  if (version != kIndexVersion) {
    error = "'" + path.string() + "' lexical index is version " + std::to_string(version) +
            ", this build reads " + std::to_string(kIndexVersion);
    return false;
  }
  if (!Bm25Index::load(tail, out.lexical)) {
    error = "'" + path.string() + "' has a corrupt lexical index";
    return false;
  }
  if (out.lexical.size() != out.store.size()) {
    error = "index halves disagree: " + std::to_string(out.store.size()) + " vectors vs " +
            std::to_string(out.lexical.size()) + " lexical documents";
    return false;
  }
  return true;
}

bool buildIndex(const fs::path& path, embeddings::IEmbeddingEngine& engine,
                const tokenizer::Tokenizer& tok, const ChunkOptions& chunkOpt, Index& out,
                IndexStats& stats, std::string& error,
                const std::function<void(int, int, const std::string&)>& progress) {
  error.clear();
  std::vector<Document> docs = loadDirectory(path, stats.skipped);
  if (docs.empty()) {
    error = "no supported documents found under '" + path.string() + "'";
    return false;
  }

  // Cap chunks at the encoder's own limit whatever the caller asked for: a longer chunk is
  // silently truncated at embed time, so its tail would never reach the index.
  ChunkOptions opt = chunkOpt;
  opt.maxTokens = std::min(opt.maxTokens, static_cast<int>(engine.maxSeqLen()));

  out.store = VectorStore(static_cast<int>(engine.dim()));
  const int cap = static_cast<int>(engine.maxSeqLen());
  const int total = static_cast<int>(docs.size());
  int done = 0;

  for (const auto& doc : docs) {
    if (progress) progress(done, total, doc.source);
    ++done;
    const auto chunks = chunkDocument(doc, tok, opt);
    for (const auto& c : chunks) {
      std::vector<float> vec;
      const auto ids = fitToEncoder(tok.encode(c.text, true), cap, tok.special().sep);
      if (!engine.embed(ids, vec, error)) {
        error = "embedding failed for " + doc.source + " chunk " + std::to_string(c.index) + ": " +
                error;
        return false;
      }
      // Both halves must receive every chunk, in the same order, or a fused hit index would point
      // at different chunks in the two arms.
      if (!out.store.add(c, vec)) {
        error = "vector dimension mismatch while indexing " + doc.source;
        return false;
      }
      out.lexical.addText(c.text);
      ++stats.chunks;
      stats.tokens += c.tokenCount;
    }
    ++stats.documents;
  }
  if (progress) progress(done, total, "");

  out.lexical.finalize();
  return true;
}

bool queryIndex(const Index& index, embeddings::IEmbeddingEngine& engine,
                const tokenizer::Tokenizer& tok, const std::string& query,
                const HybridOptions& opt, std::vector<SearchHit>& hits, std::string& error) {
  error.clear();
  if (index.store.empty()) {
    error = "the index is empty";
    return false;
  }
  std::vector<float> qv;
  const auto ids =
      fitToEncoder(tok.encode(query, true), static_cast<int>(engine.maxSeqLen()), tok.special().sep);
  if (!engine.embed(ids, qv, error)) return false;
  if (static_cast<int>(qv.size()) != index.store.dim()) {
    error = "query dimension " + std::to_string(qv.size()) + " does not match the index's " +
            std::to_string(index.store.dim()) + " — was it built with a different model?";
    return false;
  }
  hits = hybridSearch(index.store, index.lexical, qv, query, opt);
  return true;
}

}  // namespace qorvix::rag
