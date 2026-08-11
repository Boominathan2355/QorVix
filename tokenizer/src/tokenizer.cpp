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
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    // The byte-level BPE alphabet never exceeds U+01FF, so this branch was unreachable — and
    // therefore absent — until WordPiece started round-tripping arbitrary text, where CJK
    // Extension B (U+20000+) and emoji reach it.
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

// Decodes UTF-8 to codepoints. Malformed bytes are passed through as-is (as U+00XX) rather than
// rejected — the tokenizer must never throw on user text.
std::vector<std::uint32_t> utf8Decode(const std::string& s) {
  std::vector<std::uint32_t> out;
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char lead = static_cast<unsigned char>(s[i]);
    const int len = utf8Len(lead);
    if (len == 1 || i + len > s.size()) {
      out.push_back(lead);
      ++i;
      continue;
    }
    std::uint32_t cp = lead & (0xFF >> (len + 1));
    bool ok = true;
    for (int k = 1; k < len; ++k) {
      const unsigned char c = static_cast<unsigned char>(s[i + k]);
      if ((c & 0xC0) != 0x80) {
        ok = false;
        break;
      }
      cp = (cp << 6) | (c & 0x3F);
    }
    if (!ok) {
      out.push_back(lead);
      ++i;
    } else {
      out.push_back(cp);
      i += len;
    }
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

// ---- WordPiece / BERT normalization ---------------------------------------------------------
//
// This mirrors HuggingFace's BasicTokenizer, which runs before the subword matcher. Every step
// here is load-bearing for vector quality rather than cosmetic: a capitalized or accented word
// that fails to fold falls through to [UNK], and an [UNK] still produces a perfectly finite,
// unit-norm embedding — just the wrong one. There is no error to observe, which is exactly why
// these are pinned by tests.

bool wpmIsWhitespace(std::uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
         cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
         cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// Cc/Cf controls, which BERT drops outright (tab/newline are handled as whitespace above).
bool wpmIsControl(std::uint32_t cp) {
  if (cp == '\t' || cp == '\n' || cp == '\r') return false;
  return cp < 0x20 || cp == 0x7F || (cp >= 0x80 && cp <= 0x9F) || (cp >= 0x200B && cp <= 0x200F) ||
         (cp >= 0x2060 && cp <= 0x2064) || cp == 0xFEFF;
}

// Punctuation is split into standalone tokens. ASCII follows BERT's rule exactly ("all
// non-alphanumeric ASCII is punctuation"); beyond ASCII this covers the blocks that actually
// occur in text — Latin-1 marks, General Punctuation, CJK punctuation, and the fullwidth forms.
// Full Unicode category-P coverage would need the category table this file deliberately avoids.
bool wpmIsPunct(std::uint32_t cp) {
  if (cp < 0x80) {
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) ||
           (cp >= 123 && cp <= 126);
  }
  return cp == 0xA1 || cp == 0xAB || cp == 0xB6 || cp == 0xB7 || cp == 0xBB || cp == 0xBF ||
         (cp >= 0x2010 && cp <= 0x2027) || (cp >= 0x2030 && cp <= 0x205E) ||
         (cp >= 0x3001 && cp <= 0x303F) || (cp >= 0xFE30 && cp <= 0xFE4F) ||
         (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
         (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65);
}

// CJK ideographs get one token per character (they are not space-separated, so the whitespace
// split alone would swallow a whole sentence into one "word").
bool wpmIsCjk(std::uint32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
         (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) ||
         (cp >= 0x2B820 && cp <= 0x2CEAF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// Case-folds and strips accents in one pass, returning the ASCII form — do_lower_case in BERT is
// a single switch controlling both, not two independent ones.
//
// Coverage is ASCII plus Latin-1 Supplement (U+00C0-U+00FF) and Latin Extended-A (U+0100-U+017F),
// which spans essentially all accented European text. Codepoints outside that range are returned
// unchanged: a real Unicode NFD decomposition plus the Mn category table is a large amount of
// data for a benefit English-and-European retrieval does not see, so the limit is documented here
// rather than discovered later.
const char* wpmFold(std::uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z') {
    static const char kLower[26][2] = {"a", "b", "c", "d", "e", "f", "g", "h", "i",
                                       "j", "k", "l", "m", "n", "o", "p", "q", "r",
                                       "s", "t", "u", "v", "w", "x", "y", "z"};
    return kLower[cp - 'A'];
  }
  if (cp < 0xC0 || cp > 0x17F) return nullptr;

  // Latin-1 Supplement.
  if (cp <= 0xFF) {
    switch (cp) {
      case 0xC6: case 0xE6: return "ae";
      case 0xC7: case 0xE7: return "c";
      case 0xD0: case 0xF0: return "d";
      case 0xD1: case 0xF1: return "n";
      case 0xD7: case 0xF7: return nullptr;  // x and / signs are not letters
      case 0xD8: case 0xF8: return "o";
      case 0xDD: case 0xFD: case 0xFF: return "y";
      case 0xDE: case 0xFE: return "th";
      case 0xDF: return "ss";
      default: break;
    }
    if ((cp >= 0xC0 && cp <= 0xC5) || (cp >= 0xE0 && cp <= 0xE5)) return "a";
    if ((cp >= 0xC8 && cp <= 0xCB) || (cp >= 0xE8 && cp <= 0xEB)) return "e";
    if ((cp >= 0xCC && cp <= 0xCF) || (cp >= 0xEC && cp <= 0xEF)) return "i";
    if ((cp >= 0xD2 && cp <= 0xD6) || (cp >= 0xF2 && cp <= 0xF6)) return "o";
    if ((cp >= 0xD9 && cp <= 0xDC) || (cp >= 0xF9 && cp <= 0xFC)) return "u";
    return nullptr;
  }

  // Latin Extended-A, laid out as consecutive upper/lower pairs per base letter.
  if (cp == 0x132 || cp == 0x133) return "ij";
  if (cp == 0x152 || cp == 0x153) return "oe";
  if (cp <= 0x105) return "a";
  if (cp <= 0x10D) return "c";
  if (cp <= 0x111) return "d";
  if (cp <= 0x11B) return "e";
  if (cp <= 0x123) return "g";
  if (cp <= 0x127) return "h";
  if (cp <= 0x131) return "i";
  if (cp <= 0x135) return "j";
  if (cp <= 0x138) return "k";
  if (cp <= 0x142) return "l";
  if (cp <= 0x14B) return "n";
  if (cp <= 0x151) return "o";
  if (cp <= 0x159) return "r";
  if (cp <= 0x161) return "s";
  if (cp <= 0x167) return "t";
  if (cp <= 0x173) return "u";
  if (cp <= 0x175) return "w";
  if (cp <= 0x178) return "y";
  if (cp <= 0x17E) return "z";
  return "s";  // U+017F long s
}

// BERT's BasicTokenizer: clean controls, split on whitespace, isolate punctuation and CJK
// characters, and (when `lowercase`) case-fold + strip accents.
std::vector<std::string> wpmBasicTokenize(const std::string& text, bool lowercase) {
  std::vector<std::string> words;
  std::string cur;
  auto flush = [&] {
    if (!cur.empty()) {
      words.push_back(cur);
      cur.clear();
    }
  };
  for (std::uint32_t cp : utf8Decode(text)) {
    if (cp == 0 || cp == 0xFFFD || wpmIsControl(cp)) continue;
    if (wpmIsWhitespace(cp)) {
      flush();
      continue;
    }
    if (wpmIsPunct(cp) || wpmIsCjk(cp)) {
      flush();
      words.push_back(codepointToUtf8(cp));
      continue;
    }
    if (lowercase) {
      if (const char* folded = wpmFold(cp)) {
        cur += folded;
        continue;
      }
    }
    cur += codepointToUtf8(cp);
  }
  flush();
  return words;
}

bool isUtf8Continuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

// HuggingFace's WordpieceTokenizer.max_input_chars_per_word. Longer "words" (URLs, base64 blobs)
// go straight to [UNK] rather than costing O(len^2) vocab lookups.
constexpr std::size_t kMaxWordChars = 100;

}  // namespace

Tokenizer::Tokenizer(TokenizerModel model, std::vector<std::string> tokens,
                     std::vector<float> scores, std::vector<std::string> merges,
                     SpecialTokens special, bool lowercase)
    : model_(model),
      tokens_(std::move(tokens)),
      scores_(std::move(scores)),
      special_(special),
      lowercase_(lowercase) {
  for (int i = 0; i < static_cast<int>(tokens_.size()); ++i) tokenIndex_[tokens_[i]] = i;
  for (int r = 0; r < static_cast<int>(merges.size()); ++r) mergeRank_[merges[r]] = r;
  if (scores_.size() < tokens_.size()) scores_.resize(tokens_.size(), 0.0f);

  // Backfill the WordPiece specials from the vocabulary. GGUF writers are inconsistent about
  // which of these keys they emit, and a missing [UNK] in particular would turn every
  // out-of-vocabulary word into id -1 rather than a real token.
  if (model_ == TokenizerModel::WordPiece) {
    // Detect the vocabulary convention (see encodeWordPiece). llama.cpp's GGUF conversion rewrites
    // a WordPiece vocab into SentencePiece shape — word-initial pieces gain a U+2581 marker and
    // continuations lose their "##" — so a real bert GGUF holds "▁the", not "the".
    for (const std::string& t : tokens_) {
      if (t.size() >= 3 && t.compare(0, 3, "\xE2\x96\x81") == 0) {
        wpmSpaceMarker_ = true;
        break;
      }
    }
    auto resolve = [&](int current, int alias, const char* piece) {
      if (current >= 0) return current;
      if (alias >= 0) return alias;
      return tokenToId(piece);
    };
    special_.cls = resolve(special_.cls, special_.bos, "[CLS]");
    special_.sep = resolve(special_.sep, special_.eos, "[SEP]");
    special_.unk = resolve(special_.unk, -1, "[UNK]");
    special_.pad = resolve(special_.pad, -1, "[PAD]");
    special_.mask = resolve(special_.mask, -1, "[MASK]");
  }
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
  // WordPiece wraps in [CLS] ... [SEP]; SPM/BPE wrap in BOS/EOS. On most BERT GGUFs bos == cls
  // and eos == sep, so routing WordPiece through the bos/eos path would produce the right ids by
  // coincidence — and produce silently wrong ones on a file that omits bos_token_id.
  const bool wpm = model_ == TokenizerModel::WordPiece;
  const int open = wpm && special_.cls >= 0 ? special_.cls : special_.bos;
  const int close = wpm && special_.sep >= 0 ? special_.sep : special_.eos;

  std::vector<int> ids;
  if (addBos && open >= 0) ids.push_back(open);
  std::vector<int> body = wpm                            ? encodeWordPiece(text)
                          : model_ == TokenizerModel::Bpe ? encodeBpe(text)
                                                          : encodeSpm(text);
  ids.insert(ids.end(), body.begin(), body.end());
  if (special_.addEos && close >= 0) ids.push_back(close);
  return ids;
}

std::vector<int> Tokenizer::encodeWordPiece(const std::string& text) const {
  std::vector<int> ids;
  for (const std::string& rawWord : wpmBasicTokenize(text, lowercase_)) {
    if (rawWord.size() > kMaxWordChars) {
      ids.push_back(special_.unk);
      continue;
    }

    // Two vocabulary conventions exist, and picking the wrong one is silent rather than fatal:
    // every word still resolves to SOMETHING (usually [UNK], sometimes a real-but-wrong id), so
    // the embedding stays finite and unit-norm while meaning nothing.
    //
    //   HuggingFace: word-initial "the", continuation "##ing".
    //   llama.cpp GGUF: word-initial "▁the", continuation bare "ing" — the conversion rewrites a
    //     WordPiece vocab into SentencePiece shape. This is what real bert GGUFs on disk contain.
    //
    // With the marker convention the whole word gains a leading ▁ and the match runs over that
    // string, so continuation pieces start mid-string and are naturally unprefixed — exactly how
    // llama.cpp does it.
    const std::string word = wpmSpaceMarker_ ? "\xE2\x96\x81" + rawWord : rawWord;

    // Greedy longest-match-first over `word`.
    std::vector<int> pieces;
    std::size_t start = 0;
    bool covered = true;
    while (start < word.size()) {
      std::size_t end = word.size();
      int found = -1;
      while (start < end) {
        std::string sub = word.substr(start, end - start);
        if (!wpmSpaceMarker_ && start > 0) sub.insert(0, "##");
        if (const int id = tokenToId(sub); id >= 0) {
          found = id;
          break;
        }
        // Step back one whole UTF-8 character, not one byte — otherwise a multibyte character
        // (including the 3-byte ▁ marker) gets cut mid-sequence and every subsequent lookup is
        // against invalid UTF-8.
        --end;
        while (end > start && isUtf8Continuation(word[end])) --end;
      }
      if (found < 0) {
        covered = false;
        break;
      }
      pieces.push_back(found);
      start = end;
    }

    // A word with any uncoverable position becomes [UNK] AS A WHOLE — HuggingFace's behaviour.
    // Emitting the pieces found so far would change the token count, which changes both the
    // reported usage and (for mean-pooled models) the embedding itself.
    if (covered) ids.insert(ids.end(), pieces.begin(), pieces.end());
    else ids.push_back(special_.unk);
  }
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

  if (model_ == TokenizerModel::WordPiece) {
    // Mirror of the encode convention. Normalization is lossy (case and accents are gone), so
    // this reconstructs the token stream, not the original input.
    if (wpmSpaceMarker_) {
      // ▁ marks a word start, exactly as in SPM; continuations are bare and simply concatenate.
      for (const std::string& ch : utf8Chars(tok)) {
        if (ch == "\xE2\x96\x81") {
          if (!out.empty()) out.push_back(' ');
        } else {
          out += ch;
        }
      }
    } else if (tok.size() > 2 && tok[0] == '#' && tok[1] == '#') {
      out.append(tok, 2, std::string::npos);
    } else {
      if (!out.empty()) out.push_back(' ');
      out += tok;
    }
  } else if (model_ == TokenizerModel::Bpe) {
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
    if (skipSpecial && (id == special_.bos || id == special_.eos || id == special_.pad ||
                        id == special_.cls || id == special_.sep || id == special_.mask)) {
      continue;
    }
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
  else if (modelName == "bert" || modelName == "wpm")
    model = TokenizerModel::WordPiece;
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
  const bool wpm = model == TokenizerModel::WordPiece;
  special.bos = id("tokenizer.ggml.bos_token_id", -1);
  special.eos = id("tokenizer.ggml.eos_token_id", -1);
  special.unk = id("tokenizer.ggml.unknown_token_id", -1);
  special.pad = id("tokenizer.ggml.padding_token_id", -1);
  special.cls = id("tokenizer.ggml.cls_token_id", -1);
  // llama.cpp's GGUF writer misspells this key. Read the misspelling first because that is what
  // real files on disk contain, with the correct spelling as a fallback for writers that fix it.
  special.sep = id("tokenizer.ggml.seperator_token_id", id("tokenizer.ggml.separator_token_id", -1));
  special.mask = id("tokenizer.ggml.mask_token_id", -1);
  special.addBos =
      file.getBool("tokenizer.ggml.add_bos_token").value_or(model == TokenizerModel::Spm || wpm);
  // BERT always closes with [SEP]; SPM/BPE decoders default to no trailing EOS.
  special.addEos = file.getBool("tokenizer.ggml.add_eos_token").value_or(wpm);

  return Tokenizer(model, std::move(tokens), std::move(scores), std::move(merges), special);
}

}  // namespace qorvix::tokenizer
