#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "qorvix/rag/bm25.hpp"
#include "qorvix/rag/chunker.hpp"
#include "qorvix/rag/hybrid_search.hpp"
#include "qorvix/rag/loaders.hpp"
#include "qorvix/rag/vector_store.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

namespace fs = std::filesystem;
using namespace qorvix::rag;
using qorvix::tokenizer::SpecialTokens;
using qorvix::tokenizer::Tokenizer;
using qorvix::tokenizer::TokenizerModel;
using Catch::Matchers::WithinAbs;

namespace {

// A WordPiece tokenizer whose vocabulary is single lowercase letters plus a handful of words, so
// token counts in the chunker tests are predictable without pulling in a real model. Everything
// unknown becomes [UNK], which still counts as one token — which is all the chunker needs.
Tokenizer letterTokenizer() {
  std::vector<std::string> vocab{"[PAD]", "[UNK]", "[CLS]", "[SEP]"};
  for (char c = 'a'; c <= 'z'; ++c) vocab.push_back(std::string("\xE2\x96\x81") + c);
  SpecialTokens sp;
  sp.pad = 0;
  sp.unk = 1;
  sp.cls = 2;
  sp.sep = 3;
  sp.addBos = true;
  sp.addEos = true;
  return Tokenizer(TokenizerModel::WordPiece, vocab, {}, {}, sp, true);
}

fs::path tempDir(const std::string& name) {
  const fs::path dir = fs::temp_directory_path() / ("qorvix_rag_test_" + name);
  fs::remove_all(dir);
  fs::create_directories(dir);
  return dir;
}

void writeText(const fs::path& p, const std::string& s) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out << s;
}

std::vector<float> unitVec(int dim, int hot) {
  std::vector<float> v(dim, 0.0f);
  v[hot % dim] = 1.0f;
  return v;
}

}  // namespace

// ---- chunker --------------------------------------------------------------------------------

TEST_CASE("chunking respects the token budget", "[rag]") {
  const Tokenizer tok = letterTokenizer();
  std::string text;
  for (int i = 0; i < 200; ++i) text += "a b c d e ";

  ChunkOptions opt;
  opt.maxTokens = 32;
  opt.overlapTokens = 4;
  const auto chunks = chunkText(text, "doc", "doc.txt", tok, opt);

  REQUIRE(chunks.size() > 1);
  for (const auto& c : chunks) {
    // The budget is what the ENCODER will see, wrappers included — a chunk that fits "except for
    // [CLS] and [SEP]" is still truncated at embed time, silently losing its tail.
    REQUIRE(static_cast<int>(tok.encode(c.text, true).size()) <= opt.maxTokens);
    REQUIRE_FALSE(c.text.empty());
  }
}

TEST_CASE("chunks are contiguous, ordered, and cite their source span", "[rag]") {
  const Tokenizer tok = letterTokenizer();
  std::string text;
  for (int i = 0; i < 100; ++i) text += "a b c d e f g h ";

  ChunkOptions opt;
  opt.maxTokens = 24;
  opt.overlapTokens = 0;
  const auto chunks = chunkText(text, "doc", "doc.txt", tok, opt);

  REQUIRE(chunks.size() > 2);
  for (std::size_t i = 0; i < chunks.size(); ++i) {
    REQUIRE(chunks[i].index == static_cast<int>(i));
    REQUIRE(chunks[i].byteEnd > chunks[i].byteStart);
    REQUIRE(chunks[i].byteEnd <= text.size());
    // With no overlap the spans must advance and not go backwards.
    if (i > 0) REQUIRE(chunks[i].byteStart >= chunks[i - 1].byteStart);
  }
}

TEST_CASE("overlap makes adjacent chunks share text", "[rag]") {
  // Without overlap a fact spanning a boundary is in neither chunk in full and is retrievable
  // from neither, which is the whole reason the option exists.
  const Tokenizer tok = letterTokenizer();
  std::string text;
  for (int i = 0; i < 100; ++i) text += "a b c d e f g h ";

  ChunkOptions none;
  none.maxTokens = 32;
  none.overlapTokens = 0;
  ChunkOptions with;
  with.maxTokens = 32;
  with.overlapTokens = 8;

  const auto a = chunkText(text, "d", "d.txt", tok, none);
  const auto b = chunkText(text, "d", "d.txt", tok, with);
  REQUIRE(b.size() > a.size());  // overlap means more chunks for the same text
  REQUIRE(b[1].byteStart < a[1].byteStart);
}

TEST_CASE("chunking terminates on text with no whitespace at all", "[rag]") {
  // A single unbroken token stream has no boundary to snap to; the loop must still advance rather
  // than spin forever looking for one.
  const Tokenizer tok = letterTokenizer();
  const std::string text(2000, 'a');
  ChunkOptions opt;
  opt.maxTokens = 16;
  const auto chunks = chunkText(text, "d", "d.txt", tok, opt);
  REQUIRE_FALSE(chunks.empty());
}

TEST_CASE("chunking empty text yields no chunks", "[rag]") {
  const Tokenizer tok = letterTokenizer();
  REQUIRE(chunkText("", "d", "d.txt", tok, {}).empty());
  REQUIRE(chunkText("   \n\n  ", "d", "d.txt", tok, {}).empty());
}

// ---- loaders --------------------------------------------------------------------------------

TEST_CASE("markdown loading keeps prose and drops syntax", "[rag]") {
  const fs::path dir = tempDir("md");
  writeText(dir / "a.md",
            "# Heading\n\nSome **bold** text with a [link](http://example.com) here.\n\n"
            "- a list item\n\n```\ncode stays\n```\n");
  std::string err;
  const auto doc = loadDocument(dir / "a.md", err);
  REQUIRE(doc.has_value());

  REQUIRE(doc->text.find("Heading") != std::string::npos);
  REQUIRE(doc->text.find("bold") != std::string::npos);
  REQUIRE(doc->text.find("link") != std::string::npos);   // link TEXT is content
  REQUIRE(doc->text.find("a list item") != std::string::npos);
  REQUIRE(doc->text.find("code stays") != std::string::npos);  // in technical docs, code IS content
  REQUIRE(doc->text.find("http://example.com") == std::string::npos);  // the URL is not
  REQUIRE(doc->text.find("**") == std::string::npos);
  REQUIRE(doc->text.find("# ") == std::string::npos);
}

TEST_CASE("csv loading handles quoted fields with separators and newlines", "[rag]") {
  const fs::path dir = tempDir("csv");
  writeText(dir / "a.csv", "name,note\nalpha,\"has, a comma\"\nbeta,\"says \"\"hi\"\"\"\n");
  std::string err;
  const auto doc = loadDocument(dir / "a.csv", err);
  REQUIRE(doc.has_value());
  REQUIRE(doc->text.find("has, a comma") != std::string::npos);
  REQUIRE(doc->text.find("says \"hi\"") != std::string::npos);
  REQUIRE(doc->text.find("alpha | has, a comma") != std::string::npos);
}

TEST_CASE("pdf and docx report that they are not implemented, rather than returning garbage",
          "[rag]") {
  // A missing loader errors; a bad loader lies. Naive PDF scraping produces text that silently
  // poisons every embedding derived from it, which is strictly worse than refusing.
  const fs::path dir = tempDir("pdf");
  writeText(dir / "a.pdf", "%PDF-1.4 not really");
  std::string err;
  REQUIRE_FALSE(loadDocument(dir / "a.pdf", err).has_value());
  REQUIRE(err.find("not implemented") != std::string::npos);
}

TEST_CASE("directory loading skips unsupported files and is deterministic", "[rag]") {
  const fs::path dir = tempDir("dir");
  writeText(dir / "b.txt", "beta");
  writeText(dir / "a.txt", "alpha");
  writeText(dir / "c.png", "not text");
  fs::create_directories(dir / "sub");
  writeText(dir / "sub" / "d.md", "delta");

  std::vector<std::string> skipped;
  const auto docs = loadDirectory(dir, skipped);
  REQUIRE(docs.size() == 3);
  // Sorted, so an index built twice from the same tree is identical.
  REQUIRE(docs[0].text.find("alpha") != std::string::npos);
  REQUIRE(docs[1].text.find("beta") != std::string::npos);
  REQUIRE(skipped.empty());  // an unsupported extension is skipped silently, not reported
}

// ---- vector store ---------------------------------------------------------------------------

TEST_CASE("vector store returns exact cosine top-k, best first", "[rag]") {
  VectorStore store(4);
  for (int i = 0; i < 4; ++i) {
    Chunk c;
    c.docId = "d";
    c.text = "chunk " + std::to_string(i);
    c.index = i;
    REQUIRE(store.add(c, unitVec(4, i)));
  }
  const auto hits = store.searchDense(unitVec(4, 2), 2);
  REQUIRE(hits.size() == 2);
  REQUIRE(hits[0].index == 2);
  REQUIRE_THAT(hits[0].score, WithinAbs(1.0f, 1e-5f));
  REQUIRE(hits[1].score < hits[0].score);
}

TEST_CASE("vector store rejects a mismatched dimension instead of storing it", "[rag]") {
  VectorStore store(4);
  Chunk c;
  REQUIRE(store.add(c, {1.0f, 0.0f, 0.0f, 0.0f}));
  REQUIRE_FALSE(store.add(c, {1.0f, 0.0f}));
  REQUIRE(store.size() == 1);
}

TEST_CASE("vector store round-trips through a file", "[rag]") {
  const fs::path dir = tempDir("store");
  VectorStore store(4);
  for (int i = 0; i < 3; ++i) {
    Chunk c;
    c.docId = "doc" + std::to_string(i);
    c.source = "src.txt";
    c.text = "text with \"quotes\" and \n newlines";
    c.index = i;
    c.byteStart = static_cast<std::size_t>(i) * 10;
    c.byteEnd = c.byteStart + 9;
    c.tokenCount = i + 1;
    REQUIRE(store.add(c, unitVec(4, i)));
  }

  std::string err;
  const fs::path path = dir / "s.qvx";
  REQUIRE(store.save(path, err));

  const auto back = VectorStore::load(path, err);
  REQUIRE(back.has_value());
  REQUIRE(back->size() == 3);
  REQUIRE(back->dim() == 4);
  for (std::size_t i = 0; i < 3; ++i) {
    REQUIRE(back->chunk(i).docId == store.chunk(i).docId);
    REQUIRE(back->chunk(i).text == store.chunk(i).text);
    REQUIRE(back->chunk(i).byteStart == store.chunk(i).byteStart);
    REQUIRE(back->chunk(i).tokenCount == store.chunk(i).tokenCount);
    for (int d = 0; d < 4; ++d) REQUIRE(back->vector(i)[d] == store.vector(i)[d]);
  }
}

TEST_CASE("a store with a bad magic or unknown version is refused, not reinterpreted", "[rag]") {
  // Reading a stale layout would return plausible nonsense — vectors that are really chunk text.
  const fs::path dir = tempDir("badstore");
  const fs::path path = dir / "bad.qvx";
  writeText(path, std::string(64, 'x'));
  std::string err;
  REQUIRE_FALSE(VectorStore::load(path, err).has_value());
  REQUIRE(err.find("magic") != std::string::npos);

  VectorStore store(2);
  Chunk c;
  REQUIRE(store.add(c, {1.0f, 0.0f}));
  REQUIRE(store.save(path, err));
  // Corrupt the version field (bytes 4..7).
  {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(4);
    const char bogus[4] = {99, 0, 0, 0};
    f.write(bogus, 4);
  }
  REQUIRE_FALSE(VectorStore::load(path, err).has_value());
  REQUIRE(err.find("version") != std::string::npos);
}

// ---- BM25 -----------------------------------------------------------------------------------

TEST_CASE("bm25 terms are whole words, lowercased", "[rag]") {
  // Deliberately NOT WordPiece: "##ation" is not a term anyone searches for, and splitting a rare
  // word into common subwords would hand it the idf of its pieces instead of its own.
  const auto terms = bm25Terms("Hello, World! It's 2026.");
  REQUIRE(terms == std::vector<std::string>{"hello", "world", "it", "s", "2026"});
}

TEST_CASE("bm25 ranks documents containing more of the query", "[rag]") {
  Bm25Index idx;
  idx.addText("the cat sat on the mat");
  idx.addText("the dog sat on the log");
  idx.addText("quantum chromodynamics");
  idx.finalize();

  const auto hits = idx.searchText("cat mat", 3);
  REQUIRE_FALSE(hits.empty());
  REQUIRE(hits[0].index == 0);
}

TEST_CASE("bm25 scores a common term non-negatively", "[rag]") {
  // Without the +1 in the idf formula, a term appearing in more than half the corpus scores
  // NEGATIVE, so a document containing the query term ranks BELOW one that does not.
  Bm25Index idx;
  for (int i = 0; i < 5; ++i) idx.addText("common word here");
  idx.addText("something else entirely");
  idx.finalize();

  const auto hits = idx.searchText("common", 6);
  REQUIRE_FALSE(hits.empty());
  for (const auto& h : hits) REQUIRE(h.score >= 0.0f);
}

TEST_CASE("bm25 cannot be searched before finalize", "[rag]") {
  Bm25Index idx;
  idx.addText("alpha beta");
  REQUIRE_FALSE(idx.finalized());
  REQUIRE(idx.searchText("alpha", 3).empty());
  idx.finalize();
  REQUIRE_FALSE(idx.searchText("alpha", 3).empty());
  // Adding invalidates it again — a stale idf must not be searchable.
  idx.addText("gamma");
  REQUIRE_FALSE(idx.finalized());
}

TEST_CASE("bm25 round-trips through a stream", "[rag]") {
  Bm25Index idx;
  idx.addText("the cat sat on the mat");
  idx.addText("the dog sat on the log");
  idx.finalize();

  std::stringstream ss;
  REQUIRE(idx.save(ss));
  Bm25Index back;
  REQUIRE(Bm25Index::load(ss, back));
  REQUIRE(back.size() == idx.size());
  REQUIRE(back.searchText("cat", 2)[0].index == idx.searchText("cat", 2)[0].index);
}

// ---- hybrid fusion --------------------------------------------------------------------------

TEST_CASE("rrf fusion ranks a document both arms agree on above either alone", "[rag]") {
  const std::vector<SearchHit> dense{{5, 0.9f}, {1, 0.8f}, {2, 0.7f}};
  const std::vector<SearchHit> lexical{{1, 12.0f}, {9, 8.0f}, {5, 3.0f}};

  HybridOptions opt;
  opt.k = 4;
  const auto fused = fuseRrf(dense, lexical, opt);
  REQUIRE_FALSE(fused.empty());
  // Document 1 is rank 2 dense and rank 1 lexical; document 5 is rank 1 dense and rank 3 lexical.
  // RRF uses only rank, so the incomparable score SCALES (0.9 vs 12.0) never enter the decision —
  // which is the entire reason for choosing it over min-max normalization.
  REQUIRE(fused[0].index == 1);
}

TEST_CASE("alpha 1 reduces to dense and alpha 0 to lexical", "[rag]") {
  const std::vector<SearchHit> dense{{5, 0.9f}, {1, 0.8f}};
  const std::vector<SearchHit> lexical{{9, 12.0f}, {2, 8.0f}};

  HybridOptions denseOnly;
  denseOnly.alpha = 1.0f;
  denseOnly.k = 4;
  const auto d = fuseRrf(dense, lexical, denseOnly);
  REQUIRE(d[0].index == 5);
  REQUIRE(d[1].index == 1);
  // The lexical arm contributes exactly zero, so its documents rank below every dense one.
  REQUIRE(d[2].score == 0.0f);

  HybridOptions lexOnly;
  lexOnly.alpha = 0.0f;
  lexOnly.k = 4;
  const auto l = fuseRrf(dense, lexical, lexOnly);
  REQUIRE(l[0].index == 9);
  REQUIRE(l[1].index == 2);
}

TEST_CASE("rrf fusion is deterministic when scores tie", "[rag]") {
  // Fused scores accumulate in an unordered_map, so equal scores would otherwise rank differently
  // between runs depending on hash iteration order.
  const std::vector<SearchHit> dense{{3, 0.5f}, {7, 0.5f}};
  HybridOptions opt;
  opt.k = 2;
  opt.alpha = 1.0f;
  const auto a = fuseRrf(dense, {}, opt);
  const auto b = fuseRrf(dense, {}, opt);
  REQUIRE(a[0].index == b[0].index);
  REQUIRE(a[1].index == b[1].index);
}
