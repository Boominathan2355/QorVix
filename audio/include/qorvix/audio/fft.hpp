#pragma once

#include <vector>

// Discrete Fourier transform, from scratch — the same standard the DEFLATE decoder in `vision`
// holds to. Nothing here is audio-specific; it is separated from mel.cpp so the transform can be
// checked against its own mathematical definition rather than against another FFT.
namespace qorvix::audio {

// In-place complex DFT over `re` / `im` (both length n, n >= 1).
//
// n is deliberately NOT required to be a power of two. Whisper's analysis frame is 400 samples and
// 400 = 2^4 * 25, so a radix-2-only implementation could not transform it at all. This recurses on
// the even/odd split while n is even and falls back to the direct O(n^2) transform once it is not:
// 400 costs four butterfly passes plus sixteen 25-point direct transforms, a few thousand
// operations per frame instead of the 160,000 a direct 400-point transform would take.
//
// The tempting shortcut — zero-pad the frame to 512 and use a clean radix-2 — is not an
// optimization, it is a different transform. The bin spacing would become 16000/512 = 31.25 Hz
// instead of 40 Hz, so every mel filter would integrate over the wrong frequencies and produce a
// spectrogram that looks entirely plausible and is wrong everywhere.
void fft(std::vector<double>& re, std::vector<double>& im);

// The direct transform, straight from the definition. Used as the odd-n base case above, and
// exposed so tests can check `fft` against the definition instead of against itself.
void dft(const std::vector<double>& re, const std::vector<double>& im, std::vector<double>& outRe,
         std::vector<double>& outIm);

}  // namespace qorvix::audio
