#include "qorvix/audio/mel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "qorvix/audio/fft.hpp"

namespace qorvix::audio {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// --- the mel scale ---------------------------------------------------------------------------
//
// The "slaney" scale (Auditory Toolbox), which is what Whisper uses. It is LINEAR below 1 kHz at
// 3 mels per 200 Hz and logarithmic above, with the two halves meeting at exactly 15 mel. The
// other convention in common use, HTK's 2595*log10(1 + f/700), is a different curve entirely —
// picking it produces a filter bank that looks right on a plot and puts every filter in the wrong
// place. librosa defaults to slaney, torchaudio defaults to htk, which is exactly why this is
// spelled out rather than pulled from memory.
constexpr double kMinLogHertz = 1000.0;
constexpr double kMinLogMel = 15.0;

double hertzToMel(double f) {
  if (f >= kMinLogHertz) {
    const double logstep = 27.0 / std::log(6.4);
    return kMinLogMel + std::log(f / kMinLogHertz) * logstep;
  }
  return 3.0 * f / 200.0;
}

double melToHertz(double m) {
  if (m >= kMinLogMel) {
    const double logstep = std::log(6.4) / 27.0;
    return kMinLogHertz * std::exp(logstep * (m - kMinLogMel));
  }
  return 200.0 * m / 3.0;
}

}  // namespace

std::vector<float> melFilterBank(const MelConfig& cfg) {
  const int bins = cfg.bins(), nMels = cfg.nMels;
  std::vector<float> bank(static_cast<std::size_t>(bins) * nMels, 0.0f);
  if (!cfg.valid()) return bank;

  // Whisper's extractor pins max_frequency at the Nyquist rate and min at 0.
  const double maxHertz = cfg.sampleRate / 2.0;

  // nMels + 2 mel-spaced edges: filter m spans edges [m, m+2] and peaks at m+1.
  std::vector<double> edgeHertz(static_cast<std::size_t>(nMels) + 2);
  const double melMin = hertzToMel(0.0), melMax = hertzToMel(maxHertz);
  for (int i = 0; i < nMels + 2; ++i) {
    const double m = melMin + (melMax - melMin) * i / (nMels + 1);
    edgeHertz[static_cast<std::size_t>(i)] = melToHertz(m);
  }

  // Bin centre frequencies: linspace(0, Nyquist, bins), which is the same 40 Hz spacing as
  // sampleRate / nFft. Written as a linspace to mirror the reference exactly.
  std::vector<double> binHertz(static_cast<std::size_t>(bins));
  for (int b = 0; b < bins; ++b) binHertz[static_cast<std::size_t>(b)] = maxHertz * b / (bins - 1);

  for (int m = 0; m < nMels; ++m) {
    const double lo = edgeHertz[static_cast<std::size_t>(m)];
    const double mid = edgeHertz[static_cast<std::size_t>(m) + 1];
    const double hi = edgeHertz[static_cast<std::size_t>(m) + 2];
    // "slaney" normalization: each triangle is scaled to unit AREA rather than unit peak, so wide
    // high-frequency filters do not swamp narrow low-frequency ones. Omitting it leaves the bank
    // shaped correctly and scaled wrongly, which after log10 is a per-bin offset — subtle enough
    // to survive a plot and large enough to change every token.
    const double enorm = 2.0 / (hi - lo);
    for (int b = 0; b < bins; ++b) {
      const double f = binHertz[static_cast<std::size_t>(b)];
      const double rising = (f - lo) / (mid - lo);
      const double falling = (hi - f) / (hi - mid);
      const double w = std::max(0.0, std::min(rising, falling));
      bank[static_cast<std::size_t>(b) * nMels + m] = static_cast<float>(w * enorm);
    }
  }
  return bank;
}

bool logMelSpectrogram(const std::vector<float>& samples, const MelConfig& cfg,
                       std::vector<float>& out, std::string& error) {
  if (!cfg.valid()) {
    error = "invalid mel config";
    return false;
  }
  const int nFft = cfg.nFft, hop = cfg.hopLength, bins = cfg.bins();
  const int nMels = cfg.nMels, frames = cfg.frames();
  const int total = cfg.samples();
  const int half = nFft / 2;

  // 1. Whisper's input is exactly 30 s. Shorter clips are zero-padded at the END (not centred),
  //    longer ones truncated; both are what the reference does.
  std::vector<double> padded(static_cast<std::size_t>(total) + nFft, 0.0);
  // 2. Reflect padding of `half` samples at each end, around the edge sample but not repeating it
  //    (numpy/torch "reflect", not "symmetric"). With the clip zero-padded to 30 s the tail
  //    reflection is usually zeros anyway, but the HEAD reflection is real signal and getting it
  //    wrong shows up as a wrong first few frames — the part of the clip a transcript starts from.
  auto src = [&](int i) -> double {
    if (i < 0 || i >= total) return 0.0;
    return i < static_cast<int>(samples.size()) ? static_cast<double>(samples[static_cast<std::size_t>(i)]) : 0.0;
  };
  for (int i = 0; i < total; ++i) padded[static_cast<std::size_t>(half) + i] = src(i);
  for (int i = 0; i < half; ++i) {
    padded[static_cast<std::size_t>(half - 1 - i)] = src(i + 1);              // head reflection
    padded[static_cast<std::size_t>(half + total + i)] = src(total - 2 - i);  // tail reflection
  }

  // 3. Periodic Hann: 0.5 - 0.5*cos(2*pi*n/N), n in [0, N). torch.hann_window is periodic by
  //    default; the symmetric form divides by N-1 instead and is a different window.
  std::vector<double> window(static_cast<std::size_t>(nFft));
  for (int n = 0; n < nFft; ++n)
    window[static_cast<std::size_t>(n)] = 0.5 - 0.5 * std::cos(kTwoPi * n / nFft);

  const std::vector<float> bank = melFilterBank(cfg);

  // torch.stft with center=True yields frames+1 frames over this padded length; Whisper drops the
  // last one, which is why `frames` and not `frames + 1` is what the model sees.
  out.assign(static_cast<std::size_t>(nMels) * frames, 0.0f);
  std::vector<double> re(static_cast<std::size_t>(nFft)), im(static_cast<std::size_t>(nFft));
  std::vector<double> power(static_cast<std::size_t>(bins));

  double maxLog = -1e30;
  for (int t = 0; t < frames; ++t) {
    const std::size_t base = static_cast<std::size_t>(t) * hop;
    for (int n = 0; n < nFft; ++n) {
      re[static_cast<std::size_t>(n)] = padded[base + n] * window[static_cast<std::size_t>(n)];
      im[static_cast<std::size_t>(n)] = 0.0;
    }
    fft(re, im);
    // 4. POWER, not magnitude. Whisper squares before the mel projection; taking |X| instead
    //    halves every value after log10 and is invisible without a reference.
    for (int b = 0; b < bins; ++b) {
      const double r = re[static_cast<std::size_t>(b)], i = im[static_cast<std::size_t>(b)];
      power[static_cast<std::size_t>(b)] = r * r + i * i;
    }
    // 5. mel projection.
    for (int m = 0; m < nMels; ++m) {
      double acc = 0.0;
      for (int b = 0; b < bins; ++b)
        acc += power[static_cast<std::size_t>(b)] * bank[static_cast<std::size_t>(b) * nMels + m];
      const double lg = std::log10(std::max(acc, 1e-10));
      out[static_cast<std::size_t>(m) * frames + t] = static_cast<float>(lg);
      maxLog = std::max(maxLog, lg);
    }
  }

  // 6. Dynamic-range clamp and affine scale. The clamp is relative to this clip's own maximum,
  //    which is what makes the transform non-local (see the header).
  const float floorValue = static_cast<float>(maxLog - 8.0);
  for (float& v : out) v = (std::max(v, floorValue) + 4.0f) / 4.0f;
  return true;
}

}  // namespace qorvix::audio
