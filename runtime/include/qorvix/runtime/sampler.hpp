#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace qorvix::runtime {

// Token sampling configuration (SPEC "Sampling"). Disabled-value conventions match llama.cpp:
// temperature <= 0 is greedy; topK == 0, topP >= 1, minP <= 0 disable that filter; penalty
// factors of 1.0 (repetition) / 0.0 (frequency, presence) are no-ops.
struct SamplingParams {
  float temperature = 0.8f;
  int topK = 40;
  float topP = 0.95f;
  float minP = 0.05f;
  float repetitionPenalty = 1.1f;
  float frequencyPenalty = 0.0f;
  float presencePenalty = 0.0f;
  int penaltyLastN = 64;  // window of recent tokens the penalties consider
};

// Turns a logits vector into a sampled token id. Deterministic given the seed and inputs.
class Sampler {
 public:
  explicit Sampler(SamplingParams params, std::uint64_t seed = 0)
      : params_(params), rng_(seed) {}

  // Samples the next token. `recent` is the recently generated/prompt history used for penalties
  // (only the last penaltyLastN entries are considered). `logits` is modified in place.
  int sample(std::vector<float>& logits, const std::vector<int>& recent);

  const SamplingParams& params() const noexcept { return params_; }

 private:
  SamplingParams params_;
  std::mt19937_64 rng_;
};

}  // namespace qorvix::runtime
