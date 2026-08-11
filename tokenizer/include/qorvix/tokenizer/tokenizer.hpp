#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::tokenizer {

enum class TokenizerModel {
  Spm,        // SentencePiece / llama (score-based unigram merges, ▁ space marker)
  Bpe,        // byte-level BPE / gpt2, qwen2, llama3 (merge-rank)
  WordPiece,  // BERT / embedding encoders (greedy longest-match-first, ## continuations)
  Unknown,
};

struct SpecialTokens {
  int bos = -1;
  int eos = -1;
  int unk = -1;
  int pad = -1;
  // BERT wraps every sequence in [CLS] ... [SEP]. Most BERT GGUFs also set bos_token_id to [CLS]
  // and eos_token_id to [SEP], so the bos/eos machinery above would wrap correctly by coincidence
  // — until a file omits those keys. These are read explicitly so the wrapping is by design.
  int cls = -1;
  int sep = -1;
  int mask = -1;
  bool addBos = true;
  bool addEos = false;
};

// Token vocabulary + encode/decode for the two GGUF tokenizer families. Built from GGUF
// metadata (tokenizer.ggml.*) or constructed directly for tests.
//
// Scope note: the merge/score algorithms follow llama.cpp, but exact parity on real models also
// depends on pretokenization (the byte-level BPE split regex) and normalization; the current
// BPE pretokenizer is a practical GPT-2-style splitter. Bit-exact parity is validated against a
// real GGUF model in the next step.
class Tokenizer {
 public:
  // `lowercase` applies to WordPiece only: it case-folds AND strips accents, which is what
  // BERT's do_lower_case does (the two are one switch, not two). GGUF carries no do_lower_case
  // key, and llama.cpp's WPM path folds unconditionally, so fromGguf() defaults it to true —
  // correct for every uncased model, which is what the embedding models in use are.
  Tokenizer(TokenizerModel model, std::vector<std::string> tokens, std::vector<float> scores,
            std::vector<std::string> merges, SpecialTokens special, bool lowercase = true);

  static std::optional<Tokenizer> fromGguf(const gguf::GgufFile& file, std::string& error);

  std::vector<int> encode(const std::string& text, bool addBos) const;
  std::string decode(const std::vector<int>& ids, bool skipSpecial = true) const;
  std::string decodeToken(int id) const;

  int tokenToId(const std::string& token) const;
  const std::string& idToToken(int id) const;
  int vocabSize() const noexcept { return static_cast<int>(tokens_.size()); }
  TokenizerModel model() const noexcept { return model_; }
  const SpecialTokens& special() const noexcept { return special_; }
  bool lowercase() const noexcept { return lowercase_; }

 private:
  std::vector<int> encodeSpm(const std::string& text) const;
  std::vector<int> encodeBpe(const std::string& text) const;
  std::vector<int> encodeWordPiece(const std::string& text) const;
  void appendTokenText(int id, std::string& out) const;

  TokenizerModel model_;
  std::vector<std::string> tokens_;
  std::vector<float> scores_;
  std::unordered_map<std::string, int> tokenIndex_;
  std::unordered_map<std::string, int> mergeRank_;  // "left right" -> rank (BPE)
  SpecialTokens special_;
  bool lowercase_ = true;
  // Which WordPiece vocabulary convention this model uses; detected from the vocab at
  // construction. See encodeWordPiece() for why there are two.
  bool wpmSpaceMarker_ = false;
};

}  // namespace qorvix::tokenizer
