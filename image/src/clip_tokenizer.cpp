#include "qorvix/image/clip_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>

#include "qorvix/gguf/gguf_file.hpp"

namespace qorvix::image {

namespace {

const std::string kEmpty;
const std::string kEow = "</w>";

// GPT-2's byte -> code point table, which CLIP inherited verbatim. The 188 bytes that are already
// printable non-space characters map to themselves; the other 68 are lifted into U+0100 and above,
// in byte order, so that no byte can produce a space, a control character or a newline. That
// property is what lets a merge table be stored as space-separated pairs.
struct ByteTable {
  std::array<std::string, 256> encode;      // byte -> UTF-8 of its stand-in code point
  std::unordered_map<std::string, int> decode;

  ByteTable() {
    auto printable = [](int b) {
      return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      const int cp = printable(b) ? b : 256 + extra;
      if (!printable(b)) ++extra;
      std::string s;
      if (cp < 0x80) {
        s.push_back(static_cast<char>(cp));
      } else if (cp < 0x800) {
        s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      } else {
        s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
      }
      encode[static_cast<std::size_t>(b)] = s;
      decode[s] = b;
    }
  }
};

const ByteTable& byteTable() {
  static const ByteTable t;
  return t;
}

bool isAsciiLetter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool isAsciiDigit(unsigned char c) { return c >= '0' && c <= '9'; }
bool isSpace(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

// See the header's KNOWN DIVERGENCE note: everything above ASCII counts as a letter.
bool isLetterByte(unsigned char c) { return isAsciiLetter(c) || c >= 0x80; }

bool startsWith(const std::string& s, std::size_t i, const char* lit) {
  const std::size_t n = std::char_traits<char>::length(lit);
  return s.compare(i, n, lit) == 0;
}

// `re.sub(r"\s+", " ", text).strip()` followed by `.lower()`, which is what CLIP's tokenizer does
// before its regex ever runs. Lowercasing is ASCII-only here, deliberately: a Unicode case fold
// needs the same category tables the header already flags as absent, and getting it half right
// would be worse than saying so.
std::string normalize(const std::string& text) {
  std::string out;
  out.reserve(text.size());
  bool pendingSpace = false;
  for (unsigned char c : text) {
    if (isSpace(c)) {
      pendingSpace = !out.empty();
      continue;
    }
    if (pendingSpace) {
      out.push_back(' ');
      pendingSpace = false;
    }
    out.push_back(static_cast<char>(c >= 'A' && c <= 'Z' ? c + ('a' - 'A') : c));
  }
  return out;
}

}  // namespace

std::vector<std::string> clipPretokenize(const std::string& text) {
  const std::string s = normalize(text);
  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (isSpace(c)) {
      ++i;
      continue;
    }
    // The two specials are matched before anything else, exactly as the alternation orders them:
    // otherwise `<|endoftext|>` in a prompt would split into punctuation and letters.
    if (startsWith(s, i, "<|startoftext|>")) {
      out.emplace_back("<|startoftext|>");
      i += 15;
      continue;
    }
    if (startsWith(s, i, "<|endoftext|>")) {
      out.emplace_back("<|endoftext|>");
      i += 13;
      continue;
    }
    if (c == '\'' && i + 1 < s.size()) {
      static const char* kTwo[] = {"'s", "'t", "'m", "'d"};
      static const char* kThree[] = {"'re", "'ve", "'ll"};
      bool matched = false;
      for (const char* lit : kThree) {
        if (startsWith(s, i, lit)) {
          out.emplace_back(lit);
          i += 3;
          matched = true;
          break;
        }
      }
      if (matched) continue;
      for (const char* lit : kTwo) {
        if (startsWith(s, i, lit)) {
          out.emplace_back(lit);
          i += 2;
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }
    if (isLetterByte(c)) {
      const std::size_t start = i;
      while (i < s.size() && isLetterByte(static_cast<unsigned char>(s[i]))) ++i;
      out.push_back(s.substr(start, i - start));
      continue;
    }
    if (isAsciiDigit(c)) {
      // `[\p{N}]`, not `[\p{N}]+`: one digit per token, so "512" is three tokens.
      out.push_back(s.substr(i, 1));
      ++i;
      continue;
    }
    const std::size_t start = i;
    while (i < s.size()) {
      const unsigned char d = static_cast<unsigned char>(s[i]);
      if (isSpace(d) || isLetterByte(d) || isAsciiDigit(d)) break;
      ++i;
    }
    out.push_back(s.substr(start, i - start));
  }
  return out;
}

std::string clipByteEncode(const std::string& bytes) {
  const ByteTable& t = byteTable();
  std::string out;
  out.reserve(bytes.size() * 2);
  for (unsigned char c : bytes) out += t.encode[c];
  return out;
}

ClipTokenizer::ClipTokenizer(std::vector<std::string> tokens, const std::vector<std::string>& merges,
                             int bos, int eos, int pad)
    : tokens_(std::move(tokens)), bos_(bos), eos_(eos), pad_(pad) {
  tokenIndex_.reserve(tokens_.size() * 2);
  for (std::size_t i = 0; i < tokens_.size(); ++i) tokenIndex_.emplace(tokens_[i], static_cast<int>(i));
  mergeRank_.reserve(merges.size() * 2);
  for (std::size_t r = 0; r < merges.size(); ++r) {
    // Stored as "left right"; the rank IS the priority, so a later duplicate must not win.
    mergeRank_.emplace(merges[r], static_cast<int>(r));
  }
}

std::optional<ClipTokenizer> ClipTokenizer::fromGguf(const gguf::GgufFile& file, std::string& error) {
  error.clear();
  const auto model = file.getString("tokenizer.ggml.model").value_or("");
  if (model != "clip") {
    error = "tokenizer.ggml.model is '" + model + "', expected 'clip'";
    return std::nullopt;
  }
  auto readStrings = [&](const char* key, std::vector<std::string>& dst) {
    const gguf::GgufValue* v = file.find(key);
    if (!v || !v->isArray()) return false;
    dst.reserve(v->array().size());
    for (const auto& e : v->array()) {
      const std::string* s = e.asString();
      if (!s) return false;
      dst.push_back(*s);
    }
    return true;
  };
  std::vector<std::string> tokens, merges;
  if (!readStrings("tokenizer.ggml.tokens", tokens)) {
    error = "tokenizer.ggml.tokens is missing or not a string array";
    return std::nullopt;
  }
  if (!readStrings("tokenizer.ggml.merges", merges)) {
    error = "tokenizer.ggml.merges is missing or not a string array";
    return std::nullopt;
  }
  auto id = [&](const char* key) {
    const gguf::GgufValue* v = file.find(key);
    if (!v) return -1;
    const auto n = v->asI64();
    return n ? static_cast<int>(*n) : -1;
  };
  const int bos = id("tokenizer.ggml.bos_token_id");
  const int eos = id("tokenizer.ggml.eos_token_id");
  int pad = id("tokenizer.ggml.padding_token_id");
  if (bos < 0 || eos < 0) {
    error = "the file carries no bos/eos token ids";
    return std::nullopt;
  }
  // CLIP pads with <|endoftext|>. Defaulting rather than failing is safe here precisely because
  // it is not a guess: it is the same token, and every CLIP tokenizer says so.
  if (pad < 0) pad = eos;
  return ClipTokenizer(std::move(tokens), merges, bos, eos, pad);
}

void ClipTokenizer::bpe(const std::string& word, std::vector<int>& out) const {
  if (word.empty()) return;

  // Split into UTF-8 code points, then append the end-of-word marker to the LAST one. The marker
  // travels with that symbol through every merge, which is how CLIP's table distinguishes "in" at
  // the end of a word from "in" in the middle of one.
  std::vector<std::string> parts;
  for (std::size_t i = 0; i < word.size();) {
    const unsigned char c = static_cast<unsigned char>(word[i]);
    std::size_t len = 1;
    if ((c & 0xE0) == 0xC0) len = 2;
    else if ((c & 0xF0) == 0xE0) len = 3;
    else if ((c & 0xF8) == 0xF0) len = 4;
    len = std::min(len, word.size() - i);
    parts.push_back(word.substr(i, len));
    i += len;
  }
  parts.back() += kEow;

  // Merge the adjacent pair with the lowest rank until none of the remaining pairs is in the
  // table. Rescanning every pair each round is O(n^2) in the length of a word, which for prompts
  // is a handful of characters.
  while (parts.size() > 1) {
    int bestRank = std::numeric_limits<int>::max();
    std::size_t bestAt = 0;
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
      const auto it = mergeRank_.find(parts[i] + " " + parts[i + 1]);
      if (it != mergeRank_.end() && it->second < bestRank) {
        bestRank = it->second;
        bestAt = i;
      }
    }
    if (bestRank == std::numeric_limits<int>::max()) break;
    parts[bestAt] += parts[bestAt + 1];
    parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(bestAt) + 1);
  }

  for (const auto& p : parts) {
    const auto it = tokenIndex_.find(p);
    // A subword the merges produced but the vocabulary lacks cannot happen for a well-formed
    // file; dropping it silently would shift every position after it, so it is skipped loudly
    // enough to be visible in the token count rather than substituted with an unk the model has
    // never seen.
    if (it != tokenIndex_.end()) out.push_back(it->second);
  }
}

std::vector<int> ClipTokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  for (const auto& word : clipPretokenize(text)) {
    const auto special = tokenIndex_.find(word);
    if ((word == "<|startoftext|>" || word == "<|endoftext|>") && special != tokenIndex_.end()) {
      ids.push_back(special->second);
      continue;
    }
    bpe(clipByteEncode(word), ids);
  }
  return ids;
}

std::vector<int> ClipTokenizer::encodePadded(const std::string& text, int contextLength,
                                             bool& truncated) const {
  truncated = false;
  std::vector<int> ids = encode(text);
  const int room = contextLength - 2;  // bos and eos
  if (room < 0) return {};
  if (static_cast<int>(ids.size()) > room) {
    ids.resize(static_cast<std::size_t>(room));
    truncated = true;
  }
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(contextLength));
  out.push_back(bos_);
  out.insert(out.end(), ids.begin(), ids.end());
  out.push_back(eos_);
  out.resize(static_cast<std::size_t>(contextLength), pad_);
  return out;
}

const std::string& ClipTokenizer::idToToken(int id) const {
  if (id < 0 || id >= static_cast<int>(tokens_.size())) return kEmpty;
  return tokens_[static_cast<std::size_t>(id)];
}

std::string ClipTokenizer::decode(const std::vector<int>& ids, bool skipSpecial) const {
  const ByteTable& t = byteTable();
  std::string out;
  for (int id : ids) {
    if (skipSpecial && (id == bos_ || id == eos_ || id == pad_)) continue;
    std::string piece = idToToken(id);
    bool endOfWord = false;
    if (piece.size() >= kEow.size() && piece.compare(piece.size() - kEow.size(), kEow.size(), kEow) == 0) {
      piece.resize(piece.size() - kEow.size());
      endOfWord = true;
    }
    // Undo the byte table one code point at a time; anything that is not in it is passed through,
    // which is what happens to the specials when skipSpecial is false.
    for (std::size_t i = 0; i < piece.size();) {
      const unsigned char c = static_cast<unsigned char>(piece[i]);
      std::size_t len = 1;
      if ((c & 0xE0) == 0xC0) len = 2;
      else if ((c & 0xF0) == 0xE0) len = 3;
      else if ((c & 0xF8) == 0xF0) len = 4;
      len = std::min(len, piece.size() - i);
      const auto it = t.decode.find(piece.substr(i, len));
      if (it != t.decode.end()) {
        out.push_back(static_cast<char>(it->second));
      } else {
        out.append(piece, i, len);
      }
      i += len;
    }
    if (endOfWord) out.push_back(' ');
  }
  return out;
}

}  // namespace qorvix::image
