#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Shared value types for the RAG pipeline (SPEC "RAG SYSTEM").
namespace qorvix::rag {

// One loaded source document, before chunking.
struct Document {
  std::string id;      // stable identifier, usually the source path
  std::string source;  // where it came from, for citation
  std::string text;    // extracted plain text
};

// One indexed span of a document.
struct Chunk {
  std::string docId;
  std::string source;
  std::string text;
  int index = 0;             // position within the document, 0-based
  std::size_t byteStart = 0; // offset into Document::text, so a hit can cite its exact span
  std::size_t byteEnd = 0;
  int tokenCount = 0;
};

// A ranked result. `index` refers to the store's insertion order.
struct SearchHit {
  std::size_t index = 0;
  float score = 0.0f;
};

}  // namespace qorvix::rag
