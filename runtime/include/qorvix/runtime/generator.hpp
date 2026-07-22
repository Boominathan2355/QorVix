#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "qorvix/runtime/sampler.hpp"

namespace qorvix::tokenizer {
class Tokenizer;
}

namespace qorvix::runtime {

class TextModel;

struct GenerationConfig {
  int maxNewTokens = 128;
  SamplingParams sampling;
  std::uint64_t seed = 0;
  bool addBos = true;
};

struct GenerationResult {
  std::string text;
  std::vector<int> tokens;   // generated token ids (excludes the prompt)
  int promptTokens = 0;
  bool hitEos = false;
};

// Drives end-to-end text generation: tokenize prompt -> prefill KV cache -> sample/decode loop.
// Ties together a TextModel, a Tokenizer, and a Sampler. Single-sequence.
class Generator {
 public:
  Generator(TextModel& model, const tokenizer::Tokenizer& tok) : model_(model), tok_(tok) {}

  // Generates from `prompt`. `onToken`, if set, receives each new token's decoded text as it is
  // produced (streaming). Stops on EOS, maxNewTokens, or the model's max sequence length.
  GenerationResult generate(const std::string& prompt, const GenerationConfig& cfg,
                            const std::function<void(const std::string&)>& onToken = {});

 private:
  TextModel& model_;
  const tokenizer::Tokenizer& tok_;
};

}  // namespace qorvix::runtime
