#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace qorvix::audio {

// Decoded audio: mono float samples in [-1, 1], plus the rate they were recorded at.
//
// Mono because every model this feeds is mono. Multi-channel files are downmixed at load time by
// averaging the channels, which is what librosa's `mono=True` does and what Whisper's own
// preprocessing assumes; carrying the channels further would only defer the same decision to a
// place with less information.
struct AudioBuffer {
  int sampleRate = 0;
  std::vector<float> samples;  // mono, [-1, 1]

  bool valid() const { return sampleRate > 0 && !samples.empty(); }
  double seconds() const {
    return sampleRate > 0 ? static_cast<double>(samples.size()) / sampleRate : 0.0;
  }
};

// Decodes RIFF/WAVE: PCM integer at 8 (unsigned), 16, 24 and 32 bits, and IEEE float at 32 and 64
// bits, including WAVE_FORMAT_EXTENSIBLE where the real format sits in the subformat GUID.
// Dispatches on the file's magic bytes rather than its extension, matching decodeImage() — a
// mislabelled file is common and the magic is authoritative.
//
// Compressed containers (MP3, FLAC, OGG, M4A) are REFUSED BY NAME rather than misread. Each is a
// whole decoder, and the honest failure is "this format is not implemented", not a wall of noise
// produced by reading compressed bytes as PCM.
bool decodeAudio(const std::uint8_t* data, std::size_t size, AudioBuffer& out, std::string& error);

bool loadAudio(const std::filesystem::path& path, AudioBuffer& out, std::string& error);

}  // namespace qorvix::audio
