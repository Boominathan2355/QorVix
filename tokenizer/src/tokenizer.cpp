#include "qorvix/tokenizer/tokenizer.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <queue>

#include "qorvix/gguf/gguf_file.hpp"

namespace qorvix::tokenizer {

namespace {

// ---- UTF-8 helpers -------------------------------------------------------------------------

int utf8Len(unsigned char lead) {
  if (lead < 0x80) return 1;
  if ((lead >> 5) == 0x6) return 2;
  if ((lead >> 4) == 0xE) return 3;
  if ((lead >> 3) == 0x1E) return 4;
  return 1;
}

// Splits a UTF-8 string into whole-character substrings.
std::vector<std::string> utf8Chars(const std::string& s) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i < s.size();) {
    const int len = utf8Len(static_cast<unsigned char>(s[i]));
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

// ---- GPT-2 byte<->unicode alphabet (byte-level BPE) ----------------------------------------

std::string codepointToUtf8(std::uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

struct ByteUnicode {
  std::array<std::string, 256> byteToChar;         // raw byte -> unicode char (utf8)
  std::unordered_map<std::string, std::uint8_t> charToByte;

  ByteUnicode() {
    auto inPrintable = [](int b) {
      return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    };
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      std::uint32_t cp;
      if (inPrintable(b)) {
        cp = static_cast<std::uint32_t>(b);
      } else {
        cp = static_cast<std::uint32_t>(256 + n);
        ++n;
      }
      std::string ch = codepointToUtf8(cp);
      byteToChar[b] = ch;
      charToByte[ch] = static_cast<std::uint8_t>(b);
    }
  }
};

const ByteUnicode& byteUnicode() {
  static const ByteUnicode table;
  return table;
}

enum class CharCat { Space, Letter, Digit, Other };
CharCat category(unsigned char c) {
  if (std::isspace(c)) return CharCat::Space;
  if (std::isdigit(c)) return CharCat::Digit;
  if (std::isalpha(c) || c >= 0x80) return CharCat::Letter;
  return CharCat::Other;
}

// GPT-2-style pretokenization: ` ?\p{L}+ | ?\p{N}+ | ?[^\s\p{L}\p{N}]+ | \s+` over ASCII
// categories (bytes >= 0x80 treated as letters so multibyte UTF-8 stays grouped).
std::vector<std::string> pretokenize(const std::string& text) {
  std::vector<std::string> words;
  const std::size_t n = text.size();
  std::size_t i = 0;
  while (i < n) {
    const bool sp = text[i] == ' ';
    const std::size_t j = i + (sp ? 1 : 0);
    if (j < n && category(static_cast<unsigned char>(text[j])) != CharCat::Space) {
      const CharCat cat = category(static_cast<unsigned char>(text[j]));
      std::size_t k = j + 1;
      while (k < n && category(static_cast<unsigned char>(text[k])) == cat) ++k;
      words.push_back(text.substr(i, k - i));
      i = k;
    } else {
      std::size_t k = i;
      while (k < n && category(static_cast<unsigned char>(text[k])) == CharCat::Space) ++k;
      if (k == i) ++k;  // safety: always advance
      words.push_back(text.substr(i, k - i));
      i = k;
    }
  }
  return words;
}

std::string byteToken(std::uint8_t b) {
  static const char* hex = "0123456789ABCDEF";
  std::string s = "<0x";
  s.push_back(hex[b >> 4]);
  s.push_back(hex[b & 0xF]);
  s.push_back('>');
  return s;
}

}  // namespace

Tokenizer::Tokenizer(TokenizerModel model, std::vector<std::string> tokens,
                     std::vector<float> scores, std::vector<std::string> merges,
                     SpecialTokens special)
    : model_(model),
      tokens_(std::move(tokens)),
      scores_(std::move(scores)),
      special_(special) {
  for (int i = 0; i < static_cast<int>(tokens_.size()); ++i) tokenIndex_[tokens_[i]] = i;
  for (int r = 0; r < static_cast<int>(merges.size()); ++r) mergeRank_[merges[r]] = r;
  if (scores_.size() < tokens_.size()) scores_.resize(tokens_.size(), 0.0f);
}

int Tokenizer::tokenToId(const std::string& token) const {
  auto it = tokenIndex_.find(token);
  return it == tokenIndex_.end() ? -1 : it->second;
}

const std::string& Tokenizer::idToToken(int id) const {
  static const std::string kEmpty;
  if (id < 0 || id >= static_cast<int>(tokens_.size())) return kEmpty;
  return tokens_[id];
}

std::vector<int> Tokenizer::encode(const std::string& text, bool addBos) const {
  std::vector<int> ids;
  if (addBos && special_.bos >= 0) ids.push_back(special_.bos);
  std::vector<int> body = (model_ == TokenizerModel::Bpe) ? encodeBpe(text) : encodeSpm(text);
  ids.insert(ids.end(), body.begin(), body.end());
  if (special_.addEos && special_.eos >= 0) ids.push_back(special_.eos);
  return ids;
}

std::vector<int> Tokenizer::encodeSpm(const std::string& text) const {
  // SentencePiece treats spaces as the ▁ marker and prefixes a leading one.
  std::string norm = "\xE2\x96\x81";  // ▁
  for (char c : text) {
    if (c == ' ') norm += "\xE2\x96\x81";
    else norm.push_back(c);
  }

  struct Symbol {
    int prev, next;
    std::string text;
  };
  std::vector<Symbol> syms;
  for (auto& ch : utf8Chars(norm)) {
    const int idx = static_cast<int>(syms.size());
    syms.push_back({idx - 1, idx + 1, ch});
  }
  if (!syms.empty()) syms.back().next = -1;

  struct Bigram {
    int left, right;
    float score;
    std::size_t size;
  };
  auto cmp = [](const Bigram& a, const Bigram& b) {
    return a.score < b.score || (a.score == b.score && a.left > b.left);
  };
  std::priority_queue<Bigram, std::vector<Bigram>, decltype(cmp)> pq(cmp);

  auto tryAdd = [&](int left, int right) {
    if (left < 0 || right < 0) return;
    const std::string merged = syms[left].text + syms[right].text;
    const int id = tokenToId(merged);
    if (id < 0) return;
    pq.push({left, right, scores_[id], merged.size()});
  };
  for (int i = 1; i < static_cast<int>(syms.size()); ++i) tryAdd(i - 1, i);

  while (!pq.empty()) {
    const Bigram b = pq.top();
    pq.pop();
    Symbol& l = syms[b.left];
    if (l.text.empty() || l.next != b.right) continue;
    Symbol& r = syms[b.right];
    if (r.text.empty() || l.text.size() + r.text.size() != b.size) continue;

    l.text += r.text;
    l.next = r.next;
    if (r.next >= 0) syms[r.next].prev = b.left;
    r.text.clear();
    tryAdd(l.prev, b.left);
    tryAdd(b.left, l.next);
  }

  std::vector<int> ids;
  for (int i = 0; i >= 0 && i < static_cast<int>(syms.size()); i = syms[i].next) {
    if (syms[i].text.empty()) continue;
    const int id = tokenToId(syms[i].text);
    if (id >= 0) {
      ids.push_back(id);
    } else {
      for (unsigned char byte : syms[i].text) {  // byte fallback
        const int bid = tokenToId(byteToken(byte));
        ids.push_back(bid >= 0 ? bid : special_.unk);
      }
    }
  }
  return ids;
}

std::vector<int> Tokenizer::encodeBpe(const std::string& text) const {
  std::vector<int> ids;
  for (const std::string& word : pretokenize(text)) {
    // Map each raw byte to its byte-level unicode char, then split into symbols.
    std::vector<std::string> syms;
    for (unsigned char b : word) syms.push_back(byteUnicode().byteToChar[b]);

    // Greedily merge the adjacent pair with the lowest merge rank until none remain.
    for (;;) {
      int bestI = -1, bestRank = 0;
      for (int i = 0; i + 1 < static_cast<int>(syms.size()); ++i) {
        auto it = mergeRank_.find(syms[i] + " " + syms[i + 1]);
        if (it != mergeRank_.end() && (bestI < 0 || it->second < bestRank)) {
          bestI = i;
          bestRank = it->second;
        }
      }
      if (bestI < 0) break;
      syms[bestI] += syms[bestI + 1];
      syms.erase(syms.begin() + bestI + 1);
    }

    for (const std::string& s : syms) {
      const int id = tokenToId(s);
      ids.push_back(id >= 0 ? id : special_.unk);
    }
  }
  return ids;
}

void Tokenizer::appendTokenText(int id, std::string& out) const {
  const std::string& tok = idToToken(id);
  if (tok.empty()) return;

  // Byte tokens "<0xNN>" decode to the raw byte in both families.
  if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok.back() == '>') {
    auto hex = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      return -1;
    };
    const int hi = hex(tok[3]), lo = hex(tok[4]);
    if (hi >= 0 && lo >= 0) {
      out.push_back(static_cast<char>((hi << 4) | lo));
      return;
    }
  }

  if (model_ == TokenizerModel::Bpe) {
    // Reverse the byte-level alphabet: each unicode char maps back to one raw byte.
    for (const std::string& ch : utf8Chars(tok)) {
      auto it = byteUnicode().charToByte.find(ch);
      if (it != byteUnicode().charToByte.end()) out.push_back(static_cast<char>(it->second));
      else out += ch;
    }
  } else {
    // SPM: ▁ marks a space.
    for (const std::string& ch : utf8Chars(tok)) {
      if (ch == "\xE2\x96\x81") out.push_back(' ');
      else out += ch;
    }
  }
}

std::string Tokenizer::decodeToken(int id) const {
  std::string out;
  appendTokenText(id, out);
  return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skipSpecial) const {
  std::string out;
  for (int id : ids) {
    if (skipSpecial && (id == special_.bos || id == special_.eos || id == special_.pad)) continue;
    appendTokenText(id, out);
  }
  return out;
}

std::optional<Tokenizer> Tokenizer::fromGguf(const gguf::GgufFile& file, std::string& error) {
  error.clear();
  const std::string modelName = file.getString("tokenizer.ggml.model").value_or("");
  TokenizerModel model = TokenizerModel::Unknown;
  if (modelName == "llama" || modelName == "spm") model = TokenizerModel::Spm;
  else if (modelName == "gpt2" || modelName == "bpe" || modelName == "llama-bpe")
    model = TokenizerModel::Bpe;
  else {
    error = "unsupported tokenizer model '" + modelName + "'";
    return std::nullopt;
  }

  const gguf::GgufValue* toks = file.find("tokenizer.ggml.tokens");
  if (!toks || !toks->isArray()) {
    error = "missing tokenizer.ggml.tokens";
    return std::nullopt;
  }
  std::vector<std::string> tokens;
  tokens.reserve(toks->array().size());
  for (const auto& v : toks->array()) {
    if (const std::string* s = v.asString()) tokens.push_back(*s);
    else tokens.emplace_back();
  }

  std::vector<float> scores;
  if (const gguf::GgufValue* sc = file.find("tokenizer.ggml.scores"); sc && sc->isArray()) {
    scores.reserve(sc->array().size());
    for (const auto& v : sc->array()) scores.push_back(static_cast<float>(v.asF64().value_or(0.0)));
  }

  std::vector<std::string> merges;
  if (const gguf::GgufValue* mg = file.find("tokenizer.ggml.merges"); mg && mg->isArray()) {
    merges.reserve(mg->array().size());
    for (const auto& v : mg->array()) {
      if (const std::string* s = v.asString()) merges.push_back(*s);
    }
  }

  SpecialTokens special;
  auto id = [&](const char* key, int fallback) {
    if (auto v = file.getU64(key)) return static_cast<int>(*v);
    return fallback;
  };
  special.bos = id("tokenizer.ggml.bos_token_id", -1);
  special.eos = id("tokenizer.ggml.eos_token_id", -1);
  special.unk = id("tokenizer.ggml.unknown_token_id", -1);
  special.pad = id("tokenizer.ggml.padding_token_id", -1);
  special.addBos = file.getBool("tokenizer.ggml.add_bos_token").value_or(model == TokenizerModel::Spm);
  special.addEos = file.getBool("tokenizer.ggml.add_eos_token").value_or(false);

  return Tokenizer(model, std::move(tokens), std::move(scores), std::move(merges), special);
}

}  // namespace qorvix::tokenizer
