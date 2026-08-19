#include "qorvix/audio/fft.hpp"

#include <cmath>
#include <cstddef>

namespace qorvix::audio {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// Everything is computed in double even though the samples arrive as float and the mel bins leave
// as float. The reference implementation (torch.stft) is float32 throughout, so this is strictly
// more accurate, not less — and a 400-point transform of a windowed frame is cheap enough that the
// precision costs nothing worth measuring.
void fftRec(std::vector<double>& re, std::vector<double>& im) {
  const std::size_t n = re.size();
  if (n <= 1) return;

  if (n % 2 != 0) {
    std::vector<double> outRe, outIm;
    dft(re, im, outRe, outIm);
    re.swap(outRe);
    im.swap(outIm);
    return;
  }

  const std::size_t h = n / 2;
  std::vector<double> evenRe(h), evenIm(h), oddRe(h), oddIm(h);
  for (std::size_t i = 0; i < h; ++i) {
    evenRe[i] = re[2 * i];
    evenIm[i] = im[2 * i];
    oddRe[i] = re[2 * i + 1];
    oddIm[i] = im[2 * i + 1];
  }
  fftRec(evenRe, evenIm);
  fftRec(oddRe, oddIm);

  for (std::size_t k = 0; k < h; ++k) {
    const double angle = -kTwoPi * static_cast<double>(k) / static_cast<double>(n);
    const double c = std::cos(angle), s = std::sin(angle);
    const double tr = oddRe[k] * c - oddIm[k] * s;
    const double ti = oddRe[k] * s + oddIm[k] * c;
    re[k] = evenRe[k] + tr;
    im[k] = evenIm[k] + ti;
    re[k + h] = evenRe[k] - tr;
    im[k + h] = evenIm[k] - ti;
  }
}

}  // namespace

void dft(const std::vector<double>& re, const std::vector<double>& im, std::vector<double>& outRe,
         std::vector<double>& outIm) {
  const std::size_t n = re.size();
  outRe.assign(n, 0.0);
  outIm.assign(n, 0.0);
  if (n == 0 || im.size() != n) return;

  for (std::size_t k = 0; k < n; ++k) {
    double sumRe = 0.0, sumIm = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
      // The angle is formed from the k*t product reduced modulo n rather than from k*t directly:
      // for n = 25 the difference is nil, but it keeps the argument small if this base case is
      // ever handed a large odd n, where cos/sin of a huge argument loses precision.
      const double angle =
          -kTwoPi * static_cast<double>((k * t) % n) / static_cast<double>(n);
      const double c = std::cos(angle), s = std::sin(angle);
      sumRe += re[t] * c - im[t] * s;
      sumIm += re[t] * s + im[t] * c;
    }
    outRe[k] = sumRe;
    outIm[k] = sumIm;
  }
}

void fft(std::vector<double>& re, std::vector<double>& im) {
  if (re.size() != im.size()) return;
  fftRec(re, im);
}

}  // namespace qorvix::audio
