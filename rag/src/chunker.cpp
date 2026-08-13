#include "qorvix/rag/chunker.hpp"

#include <algorithm>
#include <cctype>

namespace qorvix::rag {

namespace {

// A candidate split point, with the preference tier it came from. Lower tier = better boundary.
struct Boundary {
  std::size_t pos;
  int tier;  // 0 paragraph, 1 sentence, 2 whitespace
};

bool isSentenceEnd(const std::string& s, std::size_t i) {
  const char c = s[i];
  if (c != '.' && c != '!' && c != '?') return false;
  // Require following whitespace so decimals ("3.14"), abbreviations run together, and URLs do
  // not split. A trailing terminator at end-of-text is a boundary too.
  if (i + 1 >= s.size()) return true;
  return std::isspace(static_cast<unsigned char>(s[i + 1])) != 0;
}

// Collects every plausible split point in [from, to), best tier first.
std::vector<Boundary> boundariesIn(const std::string& s, std::size_t from, std::size_t to,
                                   bool sentences) {
  std::vector<Boundary> out;
  for (std::size_t i = from; i < to && i < s.size(); ++i) {
    if (s[i] == '\n' && i + 1 < s.size() && s[i + 1] == '\n') {
      out.push_back({i + 2, 0});
    } else if (sentences && isSentenceEnd(s, i)) {
      out.push_back({i + 1, 1});
    } else if (std::isspace(static_cast<unsigned char>(s[i]))) {
      out.push_back({i + 1, 2});
    }
  }
  return out;
}

int tokenCount(const tokenizer::Tokenizer& tok, const std::string& s) {
  // Count the way the encoder will see it, wrappers included — a chunk that fits "except for
  // [CLS] and [SEP]" still gets truncated.
  return static_cast<int>(tok.encode(s, true).size());
}

}  // namespace

std::vector<Chunk> chunkText(const std::string& text, const std::string& docId,
                             const std::string& source, const tokenizer::Tokenizer& tok,
                             const ChunkOptions& opt) {
  std::vector<Chunk> chunks;
  if (text.empty() || opt.maxTokens <= 0) return chunks;

  const int overlap = std::clamp(opt.overlapTokens, 0, std::max(0, opt.maxTokens - 1));
  std::size_t start = 0;
  int index = 0;

  while (start < text.size()) {
    // Skip leading whitespace so a chunk never begins with a blank run.
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
    if (start >= text.size()) break;

    // Grow the window until it exceeds the budget, then back off to the best boundary inside it.
    // Bytes-per-token varies by language and vocabulary, so this measures rather than assuming a
    // ratio: start from an optimistic guess and correct.
    std::size_t end = std::min(text.size(), start + static_cast<std::size_t>(opt.maxTokens) * 6);
    while (end > start && tokenCount(tok, text.substr(start, end - start)) > opt.maxTokens) {
      // Shrink proportionally to how far over budget we are, so this converges in a few steps
      // instead of one byte at a time.
      const int got = tokenCount(tok, text.substr(start, end - start));
      const double ratio = static_cast<double>(opt.maxTokens) / static_cast<double>(got);
      std::size_t next = start + static_cast<std::size_t>((end - start) * ratio * 0.95);
      if (next >= end) next = end - 1;
      if (next <= start) next = start + 1;
      end = next;
    }
    // Grow back if the guess was too pessimistic and there is text left.
    while (end < text.size() &&
           tokenCount(tok, text.substr(start, std::min(text.size(), end + 32) - start)) <=
               opt.maxTokens) {
      end = std::min(text.size(), end + 32);
    }

    // Prefer a clean boundary, but only in the back half of the window — snapping to an early
    // boundary would produce chunks far under budget and inflate the index.
    std::size_t cut = end;
    if (end < text.size()) {
      const std::size_t searchFrom = start + (end - start) / 2;
      const auto cands = boundariesIn(text, searchFrom, end, opt.splitOnSentences);
      for (int tier = 0; tier <= 2 && cut == end; ++tier) {
        for (auto it = cands.rbegin(); it != cands.rend(); ++it) {
          if (it->tier == tier) {
            cut = it->pos;
            break;
          }
        }
      }
    }
    if (cut <= start) cut = std::min(text.size(), start + 1);

    std::string body = text.substr(start, cut - start);
    // Trim trailing whitespace from the stored text but keep byteEnd pointing at the real cut, so
    // the offsets still delimit the source span exactly.
    std::size_t trimmed = body.size();
    while (trimmed > 0 && std::isspace(static_cast<unsigned char>(body[trimmed - 1]))) --trimmed;
    body.resize(trimmed);

    if (!body.empty()) {
      Chunk c;
      c.docId = docId;
      c.source = source;
      c.text = body;
      c.index = index++;
      c.byteStart = start;
      c.byteEnd = cut;
      c.tokenCount = tokenCount(tok, body);
      chunks.push_back(std::move(c));
    }

    if (cut >= text.size()) break;

    // Step back by roughly `overlap` tokens so adjacent chunks share context. Without it, a fact
    // spanning a boundary is in neither chunk in full and is retrievable from neither.
    std::size_t nextStart = cut;
    if (overlap > 0) {
      const std::size_t span = cut - start;
      const int chunkTokens = std::max(1, chunks.empty() ? 1 : chunks.back().tokenCount);
      const std::size_t back =
          static_cast<std::size_t>(span * static_cast<double>(overlap) / chunkTokens);
      nextStart = cut > back ? cut - back : start + 1;
      if (nextStart <= start) nextStart = start + 1;  // always make progress
    }
    start = nextStart;
  }

  return chunks;
}

std::vector<Chunk> chunkDocument(const Document& doc, const tokenizer::Tokenizer& tok,
                                 const ChunkOptions& opt) {
  return chunkText(doc.text, doc.id, doc.source, tok, opt);
}

}  // namespace qorvix::rag
