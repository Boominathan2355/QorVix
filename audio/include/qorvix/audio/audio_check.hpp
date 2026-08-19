#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace qorvix::audio {

// Parsed reference fixture from scripts/capture_audio_reference.py.
//
// Tiered for the same reason VisionReference is, and the reason is not hypothetical: Phase 11b-1
// gated CLIP and found both of its bugs in preprocessing rather than the transformer. So the
// front end records its own tiers, each of which fails for a different cause:
//
//   * `filter` — the mel filter bank. A pure function of the config, with no audio involved at
//     all, so a mismatch here means the mel SCALE or the normalization is wrong and nothing about
//     the FFT, the window or the padding is implicated.
//   * `waveform` — a checksum of the decoded samples. Separates "the WAV was read wrong" from
//     "the spectrogram was computed wrong", which otherwise present identically.
//   * `mel` — the finished log-mel frames. Only reached when the two below it agree, so by the
//     time this fails the cause is the window, the padding mode, the power/magnitude choice or
//     the dynamic-range clamp — a short list rather than the whole pipeline.
struct AudioReference {
  std::string model;
  int sampleRate = 0;
  int nFft = 0;
  int hopLength = 0;
  int nMels = 0;
  int frames = 0;

  float waveformMean = 0.0f;
  float waveformAbsMean = 0.0f;
  int waveformSamples = 0;

  // A handful of (bin, mel) probes into the filter bank, plus its column sums. The sums catch a
  // wrong normalization even where every probe happens to land on a zero.
  struct FilterProbe {
    int bin = 0, mel = 0;
    float value = 0.0f;
  };
  std::vector<FilterProbe> filterProbes;
  std::vector<float> filterColumnSums;  // one per mel bin

  std::vector<float> frame0;    // every mel bin at t = 0, in full
  std::vector<float> melMeans;  // per-mel-bin mean over all frames

  bool load(const std::filesystem::path& path, std::string& error);
};

}  // namespace qorvix::audio
