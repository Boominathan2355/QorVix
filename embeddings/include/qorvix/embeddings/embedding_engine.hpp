#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qorvix/runtime/model_config.hpp"

// The embedding seam — the encoder-side twin of runtime::IInferenceEngine (SPEC "EMBEDDINGS
// ENGINE").
//
// Deliberately NOT a subclass of it. Phase 8.5's principle is one seam per task with one factory
// per seam; the invariant it protects is "no parallel path" (the gap it closed was `generate --gpu`
// bypassing the seam with its own loop), not "one C++ interface for every model in the process".
// IInferenceEngine has three properties an encoder cannot honour:
//
//   - its sessions ARE KV-cache allocations (memory::SessionId), and an encoder has no KV cache;
//     K and V for the whole sequence are scratch buffers discarded when embed() returns;
//   - forward(session, token, pos) is one autoregressive step where pos must equal the session
//     length, while an encoder consumes all N tokens at once and has no incremental state;
//   - it returns logits[vocabSize], while an embedding model has no LM head at all (bge-small
//     ships neither output.weight nor output_norm.weight).
//
// Subclassing would mean three stub methods and a forward() that doesn't return logits — a fake
// seam, which is the anti-pattern 8.5 removed. Two seams, two factories, still zero parallel paths.
namespace qorvix::embeddings {

using runtime::PoolingType;

class IEmbeddingEngine {
 public:
  virtual ~IEmbeddingEngine() = default;

  // Encodes one token sequence (the caller supplies [CLS]/[SEP]) using the model's own pooling and
  // normalization, writing dim() floats into `out`. False with `error` set on an empty sequence or
  // one longer than maxSeqLen().
  //
  // Writes into a CALLER-OWNED buffer rather than returning a reference to a reused member. That
  // is the lesson from IInferenceEngine::forward, whose single shared buffer forced forwardBatch()
  // to copy and forced serve() to hold a mutex across an entire generation.
  virtual bool embed(const std::vector<int>& tokens, std::vector<float>& out,
                     std::string& error) = 0;

  // Same, with the pooling and normalization overridden — reranking and Matryoshka truncation
  // want unnormalized or differently-pooled output from the same loaded model.
  virtual bool embedWith(const std::vector<int>& tokens, PoolingType pooling, bool normalize,
                         std::vector<float>& out, std::string& error) = 0;

  // Per-token hidden states, [tokens.size() * dim()] row-major, unpooled and unnormalized.
  // Late-interaction retrieval needs this; pooling is a pure function of it. Optional — the
  // default reports that it is unsupported rather than returning something plausible and wrong.
  virtual bool embedTokens(const std::vector<int>& tokens, std::vector<float>& out,
                           std::string& error) {
    (void)tokens;
    (void)out;
    error = "per-token hidden states are not supported by this engine";
    return false;
  }

  // Encodes a batch in request order. The default runs embed() sequentially; a batched
  // implementation can override it without changing any caller.
  virtual bool embedBatch(const std::vector<std::vector<int>>& batch,
                          std::vector<std::vector<float>>& out, std::string& error);

  virtual std::uint32_t dim() const = 0;            // embedding width (== d_model for BERT)
  virtual std::uint32_t maxSeqLen() const = 0;      // hard truncation limit
  virtual PoolingType defaultPooling() const = 0;   // from GGUF <arch>.pooling_type
  virtual bool defaultNormalize() const = 0;
  virtual const runtime::ModelConfig& config() const = 0;
  virtual std::string backendName() const = 0;      // "cpu" | "cuda" | "vulkan" — for logs
};

}  // namespace qorvix::embeddings
