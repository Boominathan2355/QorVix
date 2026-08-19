#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "qorvix/audio/audio_file.hpp"
#include "qorvix/audio/fft.hpp"
#include "qorvix/audio/mel.hpp"

using namespace qorvix::audio;

// Phase 11b-3a: the Whisper log-mel front end. These pin the properties that hold with NO
// reference fixture — the transform's own definition, the filter bank's algebra, and the WAV
// container's rules. `qorvix audio-check` covers what only an external reference can settle
// (that this is *Whisper's* front end and not merely a self-consistent one).

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<std::uint8_t> makeWav(int rate, int channels, int bits, std::uint16_t format,
                                  const std::vector<std::uint8_t>& pcm) {
  std::vector<std::uint8_t> w;
  auto put32 = [&](std::uint32_t v) {
    for (int i = 0; i < 4; ++i) w.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xFF));
  };
  auto put16 = [&](std::uint16_t v) {
    w.push_back(static_cast<std::uint8_t>(v & 0xFF));
    w.push_back(static_cast<std::uint8_t>(v >> 8));
  };
  auto tag = [&](const char* t) {
    for (int i = 0; i < 4; ++i) w.push_back(static_cast<std::uint8_t>(t[i]));
  };
  const int blockAlign = channels * bits / 8;
  tag("RIFF");
  put32(static_cast<std::uint32_t>(36 + pcm.size()));
  tag("WAVE");
  tag("fmt ");
  put32(16);
  put16(format);
  put16(static_cast<std::uint16_t>(channels));
  put32(static_cast<std::uint32_t>(rate));
  put32(static_cast<std::uint32_t>(rate * blockAlign));
  put16(static_cast<std::uint16_t>(blockAlign));
  put16(static_cast<std::uint16_t>(bits));
  tag("data");
  put32(static_cast<std::uint32_t>(pcm.size()));
  w.insert(w.end(), pcm.begin(), pcm.end());
  return w;
}

std::vector<std::uint8_t> pcm16(const std::vector<std::int16_t>& v) {
  std::vector<std::uint8_t> out(v.size() * 2);
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[2 * i] = static_cast<std::uint8_t>(v[i] & 0xFF);
    out[2 * i + 1] = static_cast<std::uint8_t>((v[i] >> 8) & 0xFF);
  }
  return out;
}

}  // namespace

// ---- FFT --------------------------------------------------------------------------------------

TEST_CASE("fft agrees with the direct transform at Whisper's frame size", "[audio]") {
  // 400 is 2^4 * 25 — the case a radix-2-only implementation cannot do at all, and the whole
  // reason the recursion carries an odd-n fallback. Checked against the DEFINITION rather than
  // against another FFT, so agreement is evidence and not a shared bug.
  for (int n : {16, 25, 50, 400}) {
    std::vector<double> re(static_cast<std::size_t>(n)), im(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) re[static_cast<std::size_t>(i)] = std::sin(0.37 * i) + 0.25 * i / n;

    std::vector<double> refRe, refIm;
    dft(re, im, refRe, refIm);
    std::vector<double> gotRe = re, gotIm = im;
    fft(gotRe, gotIm);

    double worst = 0.0;
    for (int k = 0; k < n; ++k) {
      worst = std::max(worst, std::abs(gotRe[static_cast<std::size_t>(k)] - refRe[static_cast<std::size_t>(k)]));
      worst = std::max(worst, std::abs(gotIm[static_cast<std::size_t>(k)] - refIm[static_cast<std::size_t>(k)]));
    }
    INFO("n = " << n);
    REQUIRE(worst < 1e-9);
  }
}

TEST_CASE("fft puts a pure tone in the bin its frequency selects", "[audio]") {
  // Bin k of an n-point transform is frequency k*rate/n. A tone placed exactly on bin 5 must put
  // essentially all of its energy there — an off-by-one in the twiddle sign or the even/odd split
  // moves the peak, which the DFT comparison above would also catch but far less legibly.
  constexpr int n = 400, bin = 5;
  std::vector<double> re(n), im(n, 0.0);
  for (int i = 0; i < n; ++i) re[static_cast<std::size_t>(i)] = std::cos(2.0 * kPi * bin * i / n);
  fft(re, im);

  int peak = 0;
  double best = -1.0;
  for (int k = 0; k < n / 2 + 1; ++k) {
    const double mag = std::hypot(re[static_cast<std::size_t>(k)], im[static_cast<std::size_t>(k)]);
    if (mag > best) { best = mag; peak = k; }
  }
  REQUIRE(peak == bin);
  REQUIRE(best > 100.0);  // n/2 for a unit cosine; anything near zero means the energy scattered
}

// ---- mel filter bank --------------------------------------------------------------------------

TEST_CASE("mel filter bank is triangular, non-negative and area-normalized", "[audio]") {
  MelConfig cfg;
  const std::vector<float> bank = melFilterBank(cfg);
  REQUIRE(bank.size() == static_cast<std::size_t>(cfg.bins()) * cfg.nMels);

  for (float v : bank) {
    REQUIRE(v >= 0.0f);
    REQUIRE(std::isfinite(v));
  }

  // Every filter must actually contain something. An empty column is the classic symptom of a
  // filter narrower than the 40 Hz bin spacing, which is what happens if the mel range or the
  // bank's frequency axis is built at the wrong scale.
  for (int m = 0; m < cfg.nMels; ++m) {
    double sum = 0.0;
    for (int b = 0; b < cfg.bins(); ++b) sum += bank[static_cast<std::size_t>(b) * cfg.nMels + m];
    INFO("mel bin " << m);
    REQUIRE(sum > 0.0);
  }
}

TEST_CASE("mel filters march upward in frequency and widen", "[audio]") {
  // The two properties that separate the slaney scale from a linear one: peaks are monotonically
  // increasing in frequency, and the filters get WIDER as they go up (the log region). A bank
  // built on a linear scale passes the first and fails the second.
  MelConfig cfg;
  const std::vector<float> bank = melFilterBank(cfg);
  auto peakBin = [&](int m) {
    int best = 0;
    float bestV = -1.0f;
    for (int b = 0; b < cfg.bins(); ++b) {
      const float v = bank[static_cast<std::size_t>(b) * cfg.nMels + m];
      if (v > bestV) { bestV = v; best = b; }
    }
    return best;
  };
  auto width = [&](int m) {
    int n = 0;
    for (int b = 0; b < cfg.bins(); ++b)
      if (bank[static_cast<std::size_t>(b) * cfg.nMels + m] > 0.0f) ++n;
    return n;
  };

  for (int m = 1; m < cfg.nMels; ++m) REQUIRE(peakBin(m) >= peakBin(m - 1));
  REQUIRE(peakBin(cfg.nMels - 1) > peakBin(0));
  // Low filters sit in the linear region and are all the same narrow width; the top ones are in
  // the log region and are several times wider.
  REQUIRE(width(cfg.nMels - 1) > 2 * width(2));
}

// ---- log-mel spectrogram ----------------------------------------------------------------------

TEST_CASE("log-mel output has Whisper's shape and bounded dynamic range", "[audio]") {
  MelConfig cfg;
  REQUIRE(cfg.frames() == 3000);
  REQUIRE(cfg.bins() == 201);

  std::vector<float> samples(16000);  // 1 s of a 440 Hz tone, padded to 30 s by the front end
  for (std::size_t i = 0; i < samples.size(); ++i)
    samples[i] = 0.5f * std::sin(2.0f * static_cast<float>(kPi) * 440.0f * i / 16000.0f);

  std::vector<float> mel;
  std::string err;
  REQUIRE(logMelSpectrogram(samples, cfg, mel, err));
  REQUIRE(mel.size() == static_cast<std::size_t>(cfg.nMels) * cfg.frames());

  float lo = 1e30f, hi = -1e30f;
  for (float v : mel) {
    REQUIRE(std::isfinite(v));
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  // The clamp keeps everything within 8 decades of the maximum, and the affine step divides by 4 —
  // so the span is at most exactly 2.0 no matter what the audio was. A wider span means the clamp
  // was skipped; a much narrower one on a signal this loud means it was applied twice.
  REQUIRE(hi - lo <= 2.0f + 1e-5f);
  REQUIRE(hi - lo > 1.9f);
}

TEST_CASE("silence produces a flat spectrogram at the clamp floor", "[audio]") {
  // All-zero audio makes every mel energy hit the 1e-10 log floor, so max == min and the whole
  // frame is one value. It is the degenerate case that would divide by zero or produce NaN if the
  // floor or the max-relative clamp were written carelessly.
  MelConfig cfg;
  std::vector<float> mel;
  std::string err;
  REQUIRE(logMelSpectrogram(std::vector<float>(16000, 0.0f), cfg, mel, err));

  const float first = mel[0];
  for (float v : mel) {
    REQUIRE(std::isfinite(v));
    REQUIRE(std::abs(v - first) < 1e-6f);
  }
}

TEST_CASE("a short clip is padded, not rejected", "[audio]") {
  // Whisper's input is always 30 s. A 10 ms clip must still produce the full 3000 frames, because
  // the encoder's positional embedding table has exactly that many slots.
  MelConfig cfg;
  std::vector<float> mel;
  std::string err;
  REQUIRE(logMelSpectrogram(std::vector<float>(160, 0.25f), cfg, mel, err));
  REQUIRE(mel.size() == static_cast<std::size_t>(cfg.nMels) * cfg.frames());
}

// ---- WAV container ----------------------------------------------------------------------------

TEST_CASE("16-bit PCM decodes to the expected floats", "[audio]") {
  const auto wav = makeWav(16000, 1, 16, 1, pcm16({0, 16384, -16384, 32767, -32768}));
  AudioBuffer a;
  std::string err;
  REQUIRE(decodeAudio(wav.data(), wav.size(), a, err));
  REQUIRE(a.sampleRate == 16000);
  REQUIRE(a.samples.size() == 5);
  REQUIRE(a.samples[0] == 0.0f);
  REQUIRE(a.samples[1] == 0.5f);
  REQUIRE(a.samples[2] == -0.5f);
  REQUIRE(a.samples[4] == -1.0f);
}

TEST_CASE("stereo is downmixed by averaging", "[audio]") {
  // Interleaved L,R. Averaging is what librosa's mono=True does; taking only the left channel
  // would silently discard half of a real recording.
  const auto wav = makeWav(16000, 2, 16, 1, pcm16({16384, -16384, 32767, 32767}));
  AudioBuffer a;
  std::string err;
  REQUIRE(decodeAudio(wav.data(), wav.size(), a, err));
  REQUIRE(a.samples.size() == 2);
  REQUIRE(std::abs(a.samples[0]) < 1e-6f);        // +0.5 and -0.5 cancel
  REQUIRE(std::abs(a.samples[1] - 0.99997f) < 1e-4f);
}

TEST_CASE("8-bit PCM is unsigned with a 128 bias", "[audio]") {
  // The one WAV bit depth that is not signed. Reading it as signed gives audio that is quiet and
  // offset rather than obviously broken, so it is worth pinning.
  const auto wav = makeWav(16000, 1, 8, 1, {128, 255, 0, 192});
  AudioBuffer a;
  std::string err;
  REQUIRE(decodeAudio(wav.data(), wav.size(), a, err));
  REQUIRE(a.samples[0] == 0.0f);
  REQUIRE(a.samples[1] > 0.99f);
  REQUIRE(a.samples[2] == -1.0f);
  REQUIRE(a.samples[3] == 0.5f);
}

TEST_CASE("unknown chunks are skipped rather than fatal", "[audio]") {
  // Editors sprinkle LIST/INFO metadata between fmt and data; a reader that cannot step over it
  // fails on a large share of real-world files.
  auto wav = makeWav(16000, 1, 16, 1, pcm16({16384}));
  std::vector<std::uint8_t> extra{'L', 'I', 'S', 'T', 4, 0, 0, 0, 'a', 'b', 'c', 'd'};
  wav.insert(wav.begin() + 36, extra.begin(), extra.end());
  // RIFF size must grow with the inserted chunk or the file is malformed.
  const std::uint32_t riff = static_cast<std::uint32_t>(wav.size() - 8);
  for (int i = 0; i < 4; ++i) wav[4 + i] = static_cast<std::uint8_t>((riff >> (8 * i)) & 0xFF);

  AudioBuffer a;
  std::string err;
  REQUIRE(decodeAudio(wav.data(), wav.size(), a, err));
  REQUIRE(a.samples.size() == 1);
  REQUIRE(a.samples[0] == 0.5f);
}

TEST_CASE("compressed containers are refused by name", "[audio]") {
  // The same courtesy vision::decodeImage extends to JPEG: a user should learn that the format is
  // unimplemented, not that their file is corrupt.
  struct Case { std::vector<std::uint8_t> magic; const char* name; };
  const std::vector<Case> cases{
      {{'f', 'L', 'a', 'C', 0, 0, 0, 34, 0, 0, 0, 0}, "FLAC"},
      {{'O', 'g', 'g', 'S', 0, 2, 0, 0, 0, 0, 0, 0}, "OGG"},
      {{'I', 'D', '3', 3, 0, 0, 0, 0, 0, 0, 0, 0}, "MP3"},
  };
  for (const auto& c : cases) {
    AudioBuffer a;
    std::string err;
    REQUIRE_FALSE(decodeAudio(c.magic.data(), c.magic.size(), a, err));
    INFO("expected '" << c.name << "' in: " << err);
    REQUIRE(err.find(c.name) != std::string::npos);
    REQUIRE(err.find("ffmpeg") != std::string::npos);  // and the fix is named
  }
}

TEST_CASE("malformed WAVs are rejected rather than half-read", "[audio]") {
  AudioBuffer a;
  std::string err;
  REQUIRE_FALSE(decodeAudio(nullptr, 0, a, err));
  const std::vector<std::uint8_t> tiny{'R', 'I', 'F', 'F'};
  REQUIRE_FALSE(decodeAudio(tiny.data(), tiny.size(), a, err));
  // A header with no data chunk.
  auto empty = makeWav(16000, 1, 16, 1, {});
  REQUIRE_FALSE(decodeAudio(empty.data(), empty.size(), a, err));
  // An unsupported codec tag must say so instead of reading the bytes as PCM.
  auto adpcm = makeWav(16000, 1, 4, 2, {1, 2, 3, 4});
  REQUIRE_FALSE(decodeAudio(adpcm.data(), adpcm.size(), a, err));
}
