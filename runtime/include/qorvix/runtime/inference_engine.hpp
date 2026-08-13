#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "qorvix/memory/kv_cache.hpp"
#include "qorvix/runtime/model_config.hpp"

namespace qorvix::runtime {

struct ForwardStep {
  memory::SessionId session = memory::kInvalidSession;
  int token = 0;
  int pos = 0;
};

// One prefill position, which is EITHER a token id or a ready-made input embedding.
//
// Phase 11b-2 exists because of this distinction: a projected image patch is a [d_model] vector
// that was never in the vocabulary, so it has no id to pass to `forward`. Rather than inventing a
// fake id (which would collide with a real token and poison the sampler's repetition history), the
// sequence carries the vector itself and the engine skips its embedding lookup for that step.
//
// `embedding`, when non-null, points at d_model floats owned by the caller and must outlive the
// forward call. `token` is then ignored for compute — it is kept only so callers have something
// to log; it is deliberately NOT fed to the sampler's history.
struct InputToken {
  int token = 0;
  const float* embedding = nullptr;

  bool isEmbedding() const { return embedding != nullptr; }
};

class IInferenceEngine {
 public:
  virtual ~IInferenceEngine() = default;

  // Opens an independent sequence. Returns memory::kInvalidSession when capacity is exhausted —
  // the scheduler treats that as "cannot admit yet", not as an error.
  virtual memory::SessionId openSession() = 0;
  virtual void closeSession(memory::SessionId s) = 0;
  virtual void resetSession(memory::SessionId s) = 0;

  // Runs the transformer for `token` at position `pos` of `session`, updating that session's KV
  // cache. Returns logits [vocabSize]. `pos` must equal the session's current length.
  //
  // The returned reference is only valid until the next call on this engine — implementations are
  // free to hand back a single reused buffer (both current ones do).
  virtual const std::vector<float>& forward(memory::SessionId session, int token, int pos) = 0;

  // Batched forward pass over multiple active sessions/tokens — one decode round for every
  // currently-decoding request. Returns one logits vector per step, in step order.
  //
  // Returns BY VALUE, deliberately. `forward` hands back a reference to a single reused buffer,
  // so collecting pointers across sequential calls would leave every entry aliasing the same
  // buffer holding only the LAST step's logits — every request in the batch would then sample
  // from the last request's distribution. The copy is what makes the sequential fallback correct;
  // a genuinely batched implementation can fill the vectors directly and pay no more.
  virtual std::vector<std::vector<float>> forwardBatch(const std::vector<ForwardStep>& steps) {
    std::vector<std::vector<float>> result;
    result.reserve(steps.size());
    for (const auto& step : steps) result.push_back(forward(step.session, step.token, step.pos));
    return result;
  }

  // Whether this backend can consume a precomputed input embedding (see InputToken). Callers MUST
  // check before calling forwardEmbedding — a backend that cannot do it says so instead of
  // silently degrading, the same rule createEmbeddingEngine follows for GPU encoders.
  //
  // The device backends hold the embedding table in VRAM and look rows up on-device, so accepting
  // a host vector means a new upload path and kernel entry point, not a wrapper. Until that
  // exists, CPU is the multimodal backend and `serve --mmproj --gpu` is refused, not faked.
  virtual bool acceptsInputEmbeddings() const { return false; }

  // Like forward(), but starts from `embedding` ([config().embeddingLength] floats) instead of an
  // embedding-table row. Same contract otherwise: `pos` must equal the session's current length,
  // and the returned reference lives only until the next call on this engine.
  //
  // Throws std::logic_error when !acceptsInputEmbeddings() — reaching it is a caller bug, not a
  // runtime condition, so it is loud rather than a zero vector that would generate plausible junk.
  virtual const std::vector<float>& forwardEmbedding(memory::SessionId session,
                                                     const float* embedding, int pos);

  // Dispatches on InputToken: embedding steps go to forwardEmbedding, id steps to forward.
  const std::vector<float>& forwardInput(memory::SessionId session, const InputToken& in, int pos) {
    return in.isEmbedding() ? forwardEmbedding(session, in.embedding, pos)
                            : forward(session, in.token, pos);
  }

  virtual std::uint32_t maxSeqLen() const = 0;
  virtual const ModelConfig& config() const = 0;

  // For logs and /v1/models — e.g. "cpu" or "cuda". Not used for dispatch.
  virtual std::string backendName() const = 0;
};

inline const std::vector<float>& IInferenceEngine::forwardEmbedding(memory::SessionId, const float*,
                                                                    int) {
  throw std::logic_error(backendName() +
                         " engine does not accept input embeddings; check acceptsInputEmbeddings()");
}

}  // namespace qorvix::runtime
