#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/weights.hpp"

namespace qorvix::runtime {

// CPU reference forward pass for a Llama-family decoder. Holds the (dequantized) weights and a
// contiguous KV cache sized to maxSeqLen. Single-sequence, one token at a time — the batched,
// paged, GPU-accelerated path is built on top of this ground truth in later phases.
class TextModel {
 public:
  // In-memory construction (tests / synthetic models). Weights may be owned-F32 WeightMats.
  TextModel(ModelConfig config, Weights weights, std::uint32_t maxSeqLen = 4096);

  // Builds from an opened GGUF file, taking ownership of it so the mmap stays alive for the
  // borrowed quantized weights. Returns nullopt with `error` set on failure.
  static std::optional<TextModel> fromGguf(gguf::GgufFile file, std::string& error,
                                           std::uint32_t maxSeqLen = 4096);

  const ModelConfig& config() const noexcept { return cfg_; }
  std::uint32_t maxSeqLen() const noexcept { return maxSeq_; }

  // Clears the KV cache (start a fresh sequence).
  void reset() noexcept { filled_ = 0; }

  // Runs the transformer for `token` at absolute position `pos` (0-based), updating the KV cache,
  // and returns logits over the vocabulary ([vocabSize]). `pos` must equal the number of tokens
  // already cached (sequential decode) and be < maxSeqLen.
  const std::vector<float>& forward(int token, int pos);

  // Feeds `prompt` (positions 0..N-1) then greedily appends up to `maxNew` argmax tokens.
  // Returns only the generated tokens. Resets the KV cache first.
  std::vector<int> generateGreedy(const std::vector<int>& prompt, int maxNew);

 private:
  void attention(const LayerWeights& L, int layer, int pos);

  ModelConfig cfg_;
  Weights w_;
  // Keeps the mmap alive when weights borrow quantized bytes from it (null for in-memory models).
  std::unique_ptr<gguf::GgufFile> file_;
  std::uint32_t maxSeq_;
  int filled_ = 0;  // number of positions currently in the KV cache

  // KV cache: [blockCount][maxSeq][kvDim], flattened.
  std::vector<float> kCache_, vCache_;

  // Per-call scratch (sized once in the constructor).
  std::vector<float> x_, xn_, q_, k_, v_, attn_, ffnGate_, ffnUp_, ffnAct_, tmpDModel_, logits_;
};

}  // namespace qorvix::runtime
