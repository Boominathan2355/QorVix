#include "qorvix/runtime/generator.hpp"

#include "qorvix/runtime/text_model.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

namespace qorvix::runtime {

GenerationResult Generator::generate(const std::string& prompt, const GenerationConfig& cfg,
                                     const std::function<void(const std::string&)>& onToken) {
  GenerationResult result;
  const std::vector<int> promptIds = tok_.encode(prompt, cfg.addBos);
  result.promptTokens = static_cast<int>(promptIds.size());
  if (promptIds.empty()) return result;

  const int maxSeq = static_cast<int>(model_.maxSeqLen());
  const int eos = tok_.special().eos;

  Sampler sampler(cfg.sampling, cfg.seed);
  std::vector<int> history = promptIds;  // penalty context
  std::vector<float> logits;

  model_.reset();
  int pos = 0;

  // Prefill: run every prompt token, keeping the logits after the last one.
  for (std::size_t i = 0; i < promptIds.size() && pos < maxSeq; ++i, ++pos) {
    logits = model_.forward(promptIds[i], pos);
  }

  // Decode loop.
  for (int n = 0; n < cfg.maxNewTokens && pos < maxSeq; ++n) {
    const int next = sampler.sample(logits, history);
    if (next < 0) break;
    if (next == eos) {
      result.hitEos = true;
      break;
    }

    const std::string piece = tok_.decodeToken(next);
    result.text += piece;
    result.tokens.push_back(next);
    history.push_back(next);
    if (onToken) onToken(piece);

    logits = model_.forward(next, pos);
    ++pos;
  }
  return result;
}

}  // namespace qorvix::runtime
