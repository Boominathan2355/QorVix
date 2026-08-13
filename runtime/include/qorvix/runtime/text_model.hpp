#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/memory/kv_cache.hpp"
#include "qorvix/runtime/inference_engine.hpp"
#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/weights.hpp"

namespace qorvix::runtime {

// CPU reference forward pass for a Llama-family decoder. Holds the (dequantized) weights and a
// contiguous KV cache sized to maxSeqLen. Single-sequence, one token at a time — the batched,
// paged, GPU-accelerated path is built on top of this ground truth in later phases.
class TextModel final : public IInferenceEngine {
 public:
  // In-memory construction (tests / synthetic models). Weights may be owned-F32 WeightMats.
  // The KV pool is sized to hold `maxSessions` concurrent sequences of up to `maxSeqLen` tokens.
  TextModel(ModelConfig config, Weights weights, std::uint32_t maxSeqLen = 4096,
            std::uint32_t maxSessions = 1);

  // Builds from an opened GGUF file, taking ownership of it so the mmap stays alive for the
  // borrowed quantized weights. Returns nullopt with `error` set on failure.
  static std::optional<TextModel> fromGguf(gguf::GgufFile file, std::string& error,
                                           std::uint32_t maxSeqLen = 4096,
                                           std::uint32_t maxSessions = 1);

  const ModelConfig& config() const noexcept override { return cfg_; }
  std::uint32_t maxSeqLen() const noexcept override { return maxSeq_; }
  std::string backendName() const override { return "cpu"; }

  // --- multi-session API (the scheduler drives many sessions against the shared KV pool) ---
  memory::SessionId openSession() override { return kv_.open(); }
  void closeSession(memory::SessionId s) override { kv_.close(s); }
  void resetSession(memory::SessionId s) override { kv_.reset(s); }
  int sessionLength(memory::SessionId s) const { return kv_.length(s); }

  // Runs the transformer for `token` at position `pos` of `session`, updating that session's KV
  // cache, and returns logits ([vocabSize]). `pos` must equal the session's current length.
  const std::vector<float>& forward(memory::SessionId session, int token, int pos) override;

  // Input embeddings (Phase 11b-2): the CPU reference is the multimodal backend, so projected
  // image patches enter the decoder here. The only difference from forward() is where x_ comes
  // from — everything after the embedding lookup is the identical code path, which is what
  // `qorvix vlm-check` tier 1 asserts by feeding a token's own embedding row back in.
  bool acceptsInputEmbeddings() const override { return true; }
  const std::vector<float>& forwardEmbedding(memory::SessionId session, const float* embedding,
                                             int pos) override;

  // --- single-sequence convenience (uses a default session opened at construction) ---
  void reset() { kv_.reset(session_); }
  const std::vector<float>& forward(int token, int pos) { return forward(session_, token, pos); }

  // Feeds `prompt` (positions 0..N-1) then greedily appends up to `maxNew` argmax tokens.
  // Returns only the generated tokens. Resets the KV cache first.
  std::vector<int> generateGreedy(const std::vector<int>& prompt, int maxNew);

 private:
  void attention(memory::SessionId session, const LayerWeights& L, int layer, int pos);
  // The transformer stack proper: runs from an already-populated x_ and returns logits. Both
  // forward() and forwardEmbedding() are just the two ways of filling x_.
  const std::vector<float>& forwardFromState(memory::SessionId session, int pos);

  ModelConfig cfg_;
  Weights w_;
  // Keeps the mmap alive when weights borrow quantized bytes from it (null for in-memory models).
  std::unique_ptr<gguf::GgufFile> file_;
  std::uint32_t maxSeq_;

  // Paged KV cache for this model's single sequence (the scheduler will hold many sessions
  // against a shared cache in Phase 7c; here it's one session sized to maxSeq).
  memory::GlobalKvCache kv_;
  memory::SessionId session_ = memory::kInvalidSession;

  // Per-call scratch (sized once in the constructor).
  std::vector<float> x_, xn_, q_, k_, v_, attn_, ffnGate_, ffnUp_, ffnAct_, tmpDModel_, logits_;
};

}  // namespace qorvix::runtime
