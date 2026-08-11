#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/embeddings/embedding_engine.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/encoder_weights.hpp"
#include "qorvix/runtime/model_config.hpp"

namespace qorvix::embeddings {

// CPU BERT-family encoder: the reference implementation of the embedding seam, mirroring how
// runtime::TextModel is the reference for generation.
//
// Not thread-safe: the per-call scratch (hidden states, K/V, attention scores) lives in members so
// it is allocated once at construction rather than per layer per token. Callers that share one
// instance across threads must serialize it, exactly as serve() does for the generation engine.
class BertModel final : public IEmbeddingEngine {
 public:
  // Takes ownership of the GgufFile so its mmap — and the quantized weight pointers borrowed from
  // it — outlive the call. `maxSeq` is clamped to the model's own context length.
  static std::optional<BertModel> fromGguf(gguf::GgufFile file, std::string& error,
                                           std::uint32_t maxSeq = 0);

  // In-memory construction, for tests and synthetic models.
  BertModel(runtime::ModelConfig cfg, runtime::EncoderWeights weights, std::uint32_t maxSeq);

  BertModel(BertModel&&) noexcept = default;
  BertModel& operator=(BertModel&&) noexcept = default;

  bool embed(const std::vector<int>& tokens, std::vector<float>& out, std::string& error) override;
  bool embedWith(const std::vector<int>& tokens, PoolingType pooling, bool normalize,
                 std::vector<float>& out, std::string& error) override;
  bool embedTokens(const std::vector<int>& tokens, std::vector<float>& out,
                   std::string& error) override;

  std::uint32_t dim() const override { return cfg_.embeddingLength; }
  std::uint32_t maxSeqLen() const override { return maxSeq_; }
  PoolingType defaultPooling() const override { return cfg_.pooling; }
  bool defaultNormalize() const override { return true; }
  const runtime::ModelConfig& config() const override { return cfg_; }
  std::string backendName() const override { return "cpu"; }

 private:
  // Runs the full encoder over `tokens`, leaving [n, d] hidden states in states_.
  bool encode(const std::vector<int>& tokens, std::string& error);
  void attention(const runtime::EncoderLayerWeights& L, int n);

  runtime::ModelConfig cfg_;
  runtime::EncoderWeights w_;
  std::uint32_t maxSeq_ = 0;
  std::unique_ptr<gguf::GgufFile> file_;  // keeps the mmap alive behind the borrowed weights

  // Scratch, all sized at construction. TextModel::attention heap-allocates its score buffer per
  // layer per token; at encoder scale that would be nLayers * nHeads * nTokens allocations per
  // document, so none of it is allocated inside the forward pass here.
  std::vector<float> states_;   // [maxSeq, d] — the residual stream
  std::vector<float> norm_;     // [maxSeq, d] — LayerNorm output
  std::vector<float> q_, k_, v_;  // [maxSeq, d]
  std::vector<float> attn_;     // [maxSeq, d]
  std::vector<float> scores_;   // [nHeads, maxSeq]
  std::vector<float> ffn_;      // [maxSeq, ffn]
  std::vector<float> ffnGate_;  // [maxSeq, ffn] — only used when cfg_.ffnGated
  std::vector<float> tmp_;      // [maxSeq, d]
};

}  // namespace qorvix::embeddings
