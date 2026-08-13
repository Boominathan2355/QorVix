#pragma once

#include <string>
#include <vector>

#include "qorvix/rag/document.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

namespace qorvix::rag {

struct ChunkOptions {
  // Must not exceed the embedding model's maxSeqLen. A chunk longer than the encoder's limit is
  // silently truncated at embed time, so its tail never reaches the index and no error is raised.
  int maxTokens = 256;
  int overlapTokens = 32;
  bool splitOnSentences = true;
};

// Splits `text` into overlapping chunks of at most opt.maxTokens tokens.
//
// Token-aware by construction, using the EMBEDDING MODEL'S OWN tokenizer: a character- or
// word-count chunker cannot honour a token budget, and the failure is silent (see above).
//
// Boundaries are chosen by descending preference — paragraph, then sentence, then whitespace,
// then a hard split — so a chunk breaks mid-sentence only when a single sentence exceeds the
// budget on its own.
std::vector<Chunk> chunkText(const std::string& text, const std::string& docId,
                             const std::string& source, const tokenizer::Tokenizer& tok,
                             const ChunkOptions& opt = {});

// Convenience wrapper over a loaded document.
std::vector<Chunk> chunkDocument(const Document& doc, const tokenizer::Tokenizer& tok,
                                 const ChunkOptions& opt = {});

}  // namespace qorvix::rag
