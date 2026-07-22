#include "qorvix/runtime/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "qorvix/runtime/ops.hpp"

namespace qorvix::runtime {

namespace {

void applyPenalties(std::vector<float>& logits, const std::vector<int>& recent,
                    const SamplingParams& p) {
  if (p.penaltyLastN == 0) return;
  const bool anyPenalty =
      p.repetitionPenalty != 1.0f || p.frequencyPenalty != 0.0f || p.presencePenalty != 0.0f;
  if (!anyPenalty) return;

  const std::size_t start =
      recent.size() > static_cast<std::size_t>(p.penaltyLastN) ? recent.size() - p.penaltyLastN : 0;
  std::unordered_map<int, int> counts;
  for (std::size_t i = start; i < recent.size(); ++i) ++counts[recent[i]];

  const int vocab = static_cast<int>(logits.size());
  for (const auto& [tok, count] : counts) {
    if (tok < 0 || tok >= vocab) continue;
    float& l = logits[tok];
    // Repetition penalty is multiplicative and sign-aware (llama.cpp convention).
    if (p.repetitionPenalty != 1.0f) l = l > 0 ? l / p.repetitionPenalty : l * p.repetitionPenalty;
    l -= count * p.frequencyPenalty;
    l -= (count > 0 ? 1.0f : 0.0f) * p.presencePenalty;
  }
}

struct Candidate {
  int id;
  float prob;
};

}  // namespace

int Sampler::sample(std::vector<float>& logits, const std::vector<int>& recent) {
  const int vocab = static_cast<int>(logits.size());
  if (vocab == 0) return -1;

  applyPenalties(logits, recent, params_);

  // Greedy path.
  if (params_.temperature <= 0.0f) return ops::argmax(logits.data(), vocab);

  // Temperature, then softmax to probabilities.
  for (float& l : logits) l /= params_.temperature;
  ops::softmax(logits.data(), vocab);

  std::vector<Candidate> cand;
  cand.reserve(vocab);
  for (int i = 0; i < vocab; ++i) cand.push_back({i, logits[i]});
  std::sort(cand.begin(), cand.end(),
            [](const Candidate& a, const Candidate& b) { return a.prob > b.prob; });

  // top-k: keep the k highest-probability candidates.
  if (params_.topK > 0 && params_.topK < static_cast<int>(cand.size())) {
    cand.resize(params_.topK);
  }

  // top-p (nucleus): smallest prefix whose cumulative probability reaches p.
  if (params_.topP < 1.0f) {
    float cum = 0.0f;
    std::size_t keep = cand.size();
    for (std::size_t i = 0; i < cand.size(); ++i) {
      cum += cand[i].prob;
      if (cum >= params_.topP) {
        keep = i + 1;
        break;
      }
    }
    cand.resize(keep);
  }

  // min-p: drop candidates below minP * max_prob (cand[0] is the max after sorting).
  if (params_.minP > 0.0f && !cand.empty()) {
    const float threshold = params_.minP * cand.front().prob;
    cand.erase(std::remove_if(cand.begin(), cand.end(),
                              [threshold](const Candidate& c) { return c.prob < threshold; }),
               cand.end());
  }

  if (cand.empty()) return ops::argmax(logits.data(), vocab);

  // Renormalize the survivors and draw.
  float total = 0.0f;
  for (const auto& c : cand) total += c.prob;
  std::uniform_real_distribution<float> dist(0.0f, total);
  float r = dist(rng_);
  for (const auto& c : cand) {
    r -= c.prob;
    if (r <= 0.0f) return c.id;
  }
  return cand.back().id;
}

}  // namespace qorvix::runtime
