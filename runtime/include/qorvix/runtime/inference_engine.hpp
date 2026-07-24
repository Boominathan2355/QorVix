#pragma once

#include <cstdint>
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

  virtual std::uint32_t maxSeqLen() const = 0;
  virtual const ModelConfig& config() const = 0;

  // For logs and /v1/models — e.g. "cpu" or "cuda". Not used for dispatch.
  virtual std::string backendName() const = 0;
};

}  // namespace qorvix::runtime
