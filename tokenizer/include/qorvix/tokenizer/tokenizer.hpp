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
  Spm,      // SentencePiece / llama (score-based unigram merges, ▁ space marker)
  Bpe,      // byte-level BPE / gpt2, qwen2, llama3 (merge-rank)
  Unknown,
};

struct SpecialTokens {
  int bos = -1;
  int eos = -1;
  int unk = -1;
  int pad = -1;
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
  Tokenizer(TokenizerModel model, std::vector<std::string> tokens, std::vector<float> scores,
            std::vector<std::string> merges, SpecialTokens special);

  static std::optional<Tokenizer> fromGguf(const gguf::GgufFile& file, std::string& error);

  std::vector<int> encode(const std::string& text, bool addBos) const;
  std::string decode(const std::vector<int>& ids, bool skipSpecial = true) const;
  std::string decodeToken(int id) const;

  int tokenToId(const std::string& token) const;
  const std::string& idToToken(int id) const;
  int vocabSize() const noexcept { return static_cast<int>(tokens_.size()); }
  TokenizerModel model() const noexcept { return model_; }
  const SpecialTokens& special() const noexcept { return special_; }

 private:
  std::vector<int> encodeSpm(const std::string& text) const;
  std::vector<int> encodeBpe(const std::string& text) const;
  void appendTokenText(int id, std::string& out) const;

  TokenizerModel model_;
  std::vector<std::string> tokens_;
  std::vector<float> scores_;
  std::unordered_map<std::string, int> tokenIndex_;
  std::unordered_map<std::string, int> mergeRank_;  // "left right" -> rank (BPE)
  SpecialTokens special_;
};

}  // namespace qorvix::tokenizer
