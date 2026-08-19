#pragma once

#include <string>
#include <vector>

// Whisper's log-mel front end, reproducing HuggingFace's WhisperFeatureExtractor.
//
// This is the audio analogue of vision/preprocess.hpp, and it carries the same warning for the
// same reason. Phase 11b-1 gated the CLIP tower and found that BOTH of its bugs were in
// preprocessing, not the transformer: the weights are exact, but if the window, the padding mode
// or the mel scale differ slightly, every frame differs and nothing errors. So the front end is a
// separate module with a separate gate, and it is verified before a single Whisper weight loads.
//
// The pipeline, and the detail in each step that is easy to get wrong:
//
//   1. pad or truncate to exactly 30 s (480,000 samples at 16 kHz) — Whisper has no other length
//   2. reflect-pad by n_fft/2 at both ends (torch.stft's center=True default; ZERO padding here
//      is the common mistake and it darkens the first and last few frames)
//   3. frame at hop 160, multiply by a PERIODIC Hann window (torch's default; the symmetric
//      variant differs in one sample and shifts every bin slightly)
//   4. FFT, then the POWER spectrum |X|^2 — not the magnitude
//   5. project through the mel filter bank, using the "slaney" mel scale AND "slaney"
//      normalization (librosa's htk scale is the other convention and gives different filters)
//   6. log10 with a 1e-10 floor, clamp to within 8 decades of the maximum, then (x + 4) / 4
//
// Step 6's clamp is relative to the maximum of THIS clip, which makes the transform non-local:
// two clips concatenated do not produce the concatenation of their features. That is Whisper's
// own definition, not an accident, and it is why streaming transcription needs the chunking to be
// decided before the features are computed.
namespace qorvix::audio {

struct MelConfig {
  int sampleRate = 16000;
  int nFft = 400;
  int hopLength = 160;
  int nMels = 80;  // 128 for large-v3
  int chunkSeconds = 30;

  int samples() const { return chunkSeconds * sampleRate; }          // 480000
  int frames() const { return samples() / hopLength; }               // 3000
  int bins() const { return nFft / 2 + 1; }                          // 201
  bool valid() const {
    return sampleRate > 0 && nFft > 1 && hopLength > 0 && nMels > 0 && chunkSeconds > 0;
  }
};

// The filter bank, [bins x nMels] row-major by frequency bin (bank[b * nMels + m]).
//
// Exposed on its own because it depends on no audio at all — it is a pure function of the config —
// so it can be checked against the reference exactly, without a waveform in the picture. When the
// spectrogram is wrong, knowing whether the bank is wrong halves the search immediately.
std::vector<float> melFilterBank(const MelConfig& cfg);

// The log-mel spectrogram, [nMels x frames] row-major by mel bin (out[m * frames + t]) — the
// layout Whisper's convolutional stem consumes, with mel bins as channels.
//
// `samples` must already be at cfg.sampleRate; resampling is the caller's problem and is refused
// rather than guessed at (see loadAudio and `qorvix audio-mel`).
bool logMelSpectrogram(const std::vector<float>& samples, const MelConfig& cfg,
                       std::vector<float>& out, std::string& error);

}  // namespace qorvix::audio
