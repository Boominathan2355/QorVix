#pragma once

#include <cmath>
#include <cstdint>

namespace qorvix::image {

// The noise source for the initial latent and for the ancestral samplers.
//
// THESE SEEDS ARE OURS. Seed 42 here does not produce the picture seed 42 produces in a PyTorch
// pipeline, and it cannot without reimplementing MT19937 plus the exact order `torch.randn` fills
// a tensor — which would still only agree on CPU, and would tie this runtime's output to another
// project's internals forever. What is guaranteed instead, and is what a seed is actually for, is
// that the same seed and settings give the same image here, on any machine, in any build.
//
// The gate sidesteps the question entirely: `scripts/capture_sd_reference.py` exports the latents
// it started from, and `qorvix sd-check` denoises those, so the model is compared without a
// random number generator anywhere in the comparison.
//
// splitmix64 for the stream (small, well-distributed, no warm-up), Box-Muller for the normal.
class GaussianRng {
 public:
  explicit GaussianRng(std::uint64_t seed) : state_(seed) {}

  double uniform() {
    // (0, 1): Box-Muller takes a logarithm of this, and a hard zero would produce an infinity.
    return (static_cast<double>(next() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }

  float normal() {
    if (hasSpare_) {
      hasSpare_ = false;
      return spare_;
    }
    const double u1 = uniform();
    const double u2 = uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 6.283185307179586476925286766559 * u2;
    spare_ = static_cast<float>(r * std::sin(theta));
    hasSpare_ = true;
    return static_cast<float>(r * std::cos(theta));
  }

 private:
  std::uint64_t next() {
    state_ += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  std::uint64_t state_;
  float spare_ = 0.0f;
  bool hasSpare_ = false;
};

}  // namespace qorvix::image
