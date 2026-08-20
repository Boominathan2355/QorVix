#pragma once

#include <optional>
#include <string>
#include <vector>

#include "qorvix/image/nn.hpp"
#include "qorvix/image/rng.hpp"
#include "qorvix/image/sd_weights.hpp"

namespace qorvix::image {

// The sampler: the arithmetic that walks a latent from noise to a picture, given a model that
// predicts what the noise was.
//
// EVERYTHING HERE IS A PURE FUNCTION OF THE SCHEDULER CONFIG. No weights, no network — which
// makes it the one part of image generation that can be pinned exactly against a reference
// without a model, and the reason `sd-check`'s first tier is the schedule rather than the UNet.
// It is also where a wrong answer is least visible: a mis-derived beta ladder produces a picture,
// just a washed-out or over-saturated one, and nothing anywhere reports a problem.
//
// TWO CONVENTIONS THAT ARE NOT INTERCHANGEABLE and are read from the file, not assumed:
//
//   * `beta_schedule`. "scaled_linear" interpolates the SQUARE ROOTS of the endpoints and squares
//     the result; "linear" interpolates the endpoints. Every Stable Diffusion release uses the
//     first, and they differ by enough to change the image everywhere.
//   * `prediction_type`. "epsilon" means the model predicts the noise; "v_prediction" means it
//     predicts a velocity. Reading one as the other yields a smooth grey field from the first
//     step onward, which is at least loud.
enum class SamplerKind {
  Ddim,             // deterministic, the reference implementation's simplest sampler
  Euler,            // deterministic, in sigma space
  EulerAncestral,   // Euler plus injected noise at every step
};

// Returns nullopt for an unknown name. Names match the diffusers scheduler they reproduce.
std::optional<SamplerKind> samplerFromName(const std::string& name);
const char* samplerName(SamplerKind kind);

class Scheduler {
 public:
  static std::optional<Scheduler> make(const SdSchedulerConfig& cfg, SamplerKind kind, int steps,
                                       std::string& error);

  // The timesteps the UNet is evaluated at, in the order they are visited (descending).
  const std::vector<int>& timesteps() const { return timesteps_; }
  int steps() const { return static_cast<int>(timesteps_.size()); }

  // The standard deviation the initial latent is drawn at. 1.0 for DDIM; the largest sigma for
  // the Euler samplers, which work in a space where the latent is not unit-variance.
  float initNoiseSigma() const { return initNoiseSigma_; }

  // Applied to the latent before every UNet call. Identity for DDIM.
  void scaleModelInput(FeatureMap& x, int stepIndex) const;

  // One denoising step, in place. `modelOutput` is whatever the UNet predicted; interpreting it
  // is this class's job, which is why `prediction_type` lives here and not in the UNet.
  bool step(FeatureMap& sample, const FeatureMap& modelOutput, int stepIndex, GaussianRng& rng,
            std::string& error);

  // Exposed so the gate can compare the schedule itself, before any weights are involved.
  const std::vector<float>& alphasCumprod() const { return alphasCumprod_; }
  const std::vector<float>& sigmas() const { return sigmas_; }
  SamplerKind kind() const { return kind_; }

 private:
  SamplerKind kind_ = SamplerKind::Ddim;
  SdSchedulerConfig cfg_;
  std::vector<float> alphasCumprod_;  // [trainTimesteps]
  std::vector<float> sigmas_;         // [steps + 1] for the Euler samplers, empty for DDIM
  std::vector<int> timesteps_;
  float finalAlphaCumprod_ = 1.0f;
  float initNoiseSigma_ = 1.0f;
  int stepRatio_ = 1;
};

}  // namespace qorvix::image
