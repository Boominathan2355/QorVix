#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::image {

// CLIP's byte-pair encoder — the one Stable Diffusion conditions on.
//
// WHY THIS IS NOT `tokenizer::Tokenizer`'s BPE. The repo already has a byte-level BPE, and it is
// the wrong one in three ways that all change the ids:
//
//   * CLIP marks the END of a word (`dog</w>`); GPT-2 marks the start of one (`Ġdog`). That is
//     not a notation difference — the merge tables are built around it, so applying one table with
//     the other convention merges nothing and every word falls apart into characters.
//   * CLIP normalizes first: whitespace collapses to single spaces and the text is lowercased.
//     GPT-2's tokenizer is case-sensitive by design.
//   * CLIP's pretokenizer splits digits ONE AT A TIME (`[\p{N}]`, not `[\p{N}]+`), so "2024" is
//     four tokens.
//
// Threading a second convention through the shared encoder would have put a branch in every step
// of a hot path that the text models depend on, to serve one caller. This is 150 lines that
// nothing else has to read.
//
// KNOWN DIVERGENCE, stated rather than discovered later: the pretokenizer's `\p{L}` and `\p{N}`
// classes are approximated as "ASCII letter, or any non-ASCII code point" and "ASCII digit".
// That matches CLIP exactly for ASCII and for alphabetic scripts; it differs for non-ASCII
// punctuation and symbols (an ellipsis, a CJK comma, an emoji), which this splits as letters and
// CLIP splits as symbols. A full Unicode category table is the fix, and the vocabulary these
// models ship is overwhelmingly ASCII.
class ClipTokenizer {
 public:
  ClipTokenizer(std::vector<std::string> tokens, const std::vector<std::string>& merges, int bos,
                int eos, int pad);

  static std::optional<ClipTokenizer> fromGguf(const gguf::GgufFile& file, std::string& error);

  // The prompt as the text encoder consumes it: exactly `contextLength` ids, opening with BOS,
  // closing with EOS, then padded out with the padding token.
  //
  // The padding is NOT masked. CLIP's text encoder in a diffusion pipeline runs with no attention
  // mask at all, so those trailing tokens are attended to and are part of the conditioning that
  // produced the model's training distribution. Masking them "correctly" would change every
  // image.
  std::vector<int> encodePadded(const std::string& text, int contextLength, bool& truncated) const;

  // The prompt's own tokens, with no specials and no padding. Exposed for tests and for the gate.
  std::vector<int> encode(const std::string& text) const;

  std::string decode(const std::vector<int>& ids, bool skipSpecial = true) const;

  int bos() const { return bos_; }
  int eos() const { return eos_; }
  int pad() const { return pad_; }
  int vocabSize() const { return static_cast<int>(tokens_.size()); }
  const std::string& idToToken(int id) const;

 private:
  // One pretokenized word (already byte-encoded) through the merge table.
  void bpe(const std::string& word, std::vector<int>& out) const;

  std::vector<std::string> tokens_;
  std::unordered_map<std::string, int> tokenIndex_;
  std::unordered_map<std::string, int> mergeRank_;  // "left right" -> rank
  int bos_ = -1, eos_ = -1, pad_ = -1;
};

// The pretokenizer, split out so a test can pin it without a vocabulary. Returns the raw
// substrings CLIP's regex would match, after whitespace collapsing and lowercasing.
std::vector<std::string> clipPretokenize(const std::string& text);

// GPT-2's byte <-> code-point table, which CLIP reuses unchanged. Maps a raw byte to the UTF-8
// encoding of its stand-in code point, so that every byte sequence becomes printable text with no
// spaces in it.
std::string clipByteEncode(const std::string& bytes);

}  // namespace qorvix::image
