#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace qorvix::audio {

// Parsed reference trace from scripts/capture_whisper_reference.py (transformers, fp32).
//
// Tiered for the reason Phase 11b-1 established the hard way — one aggregate verdict names the
// wrong culprit — and Whisper has three culprits whose failures look identical at the end, a
// transcript that is fluent and not what other runtimes produce:
//
//   * `encFrame0` / `encDimMeans` — the encoder alone. Position 0 in full catches a wrong
//     convolution or a wrong position row; the per-dimension means over all positions catch a
//     frame SHIFT, which leaves position 0 intact and is exactly what a stride-2 off-by-one does.
//   * `argmax0` / `logitsTop` / `logitProbes` — one decoder step over the forced prefix, RAW: no
//     suppression applied. This is where cross-attention lives, and cross-attention reading the
//     wrong keys still produces confident logits.
//   * `tokens` / `text` — the greedy loop WITH the model's suppression lists. Two runtimes that
//     agree on logits still disagree here if one ignores those lists (on the probe, the raw argmax
//     is a suppressed token, so this tier fails loudly rather than subtly if they are skipped).
struct WhisperReference {
  std::string model;
  std::string language;
  int dModel = 0;
  int encCtx = 0;
  int vocab = 0;
  int maxNewTokens = 0;

  std::vector<int> prompt;  // the forced prefix transformers decoded with

  std::vector<float> encFrame0;     // every dimension at encoder position 0
  std::vector<float> encDimMeans;   // per-dimension mean over every position

  struct LogitProbe {
    int id = 0;
    float value = 0.0f;
  };
  int argmax0 = -1;
  std::vector<LogitProbe> logitsTop;    // top-5 (id, logit) at the first generated position
  std::vector<LogitProbe> logitProbes;  // fixed ids, so a lucky top-5 agreement proves nothing

  std::vector<int> tokens;  // transformers' greedy stream (generated part only)
  std::string text;         // its decoded transcript, leading space and all

  bool load(const std::filesystem::path& path, std::string& error);
};

}  // namespace qorvix::audio
