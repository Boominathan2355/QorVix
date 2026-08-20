#include "qorvix/image/scheduler.hpp"

#include <algorithm>
#include <cmath>

namespace qorvix::image {

namespace {

// Linear interpolation of the sigma ladder at a fractional training step, which is how the Euler
// samplers reach timesteps that are not integers on the training grid.
float interpolateSigma(const std::vector<float>& table, double t) {
  if (table.empty()) return 0.0f;
  if (t <= 0.0) return table.front();
  const double last = static_cast<double>(table.size() - 1);
  if (t >= last) return table.back();
  const std::size_t lo = static_cast<std::size_t>(t);
  const double frac = t - static_cast<double>(lo);
  return static_cast<float>(table[lo] * (1.0 - frac) + table[lo + 1] * frac);
}

}  // namespace

std::optional<SamplerKind> samplerFromName(const std::string& name) {
  if (name == "ddim") return SamplerKind::Ddim;
  if (name == "euler") return SamplerKind::Euler;
  if (name == "euler-a" || name == "euler_a" || name == "euler-ancestral") {
    return SamplerKind::EulerAncestral;
  }
  return std::nullopt;
}

const char* samplerName(SamplerKind kind) {
  switch (kind) {
    case SamplerKind::Ddim: return "ddim";
    case SamplerKind::Euler: return "euler";
    case SamplerKind::EulerAncestral: return "euler-a";
  }
  return "?";
}

std::optional<Scheduler> Scheduler::make(const SdSchedulerConfig& cfg, SamplerKind kind, int steps,
                                         std::string& error) {
  error.clear();
  const int T = cfg.trainTimesteps;
  if (steps <= 0 || steps > T) {
    error = "steps must be between 1 and " + std::to_string(T);
    return std::nullopt;
  }
  if (cfg.predictionType != "epsilon" && cfg.predictionType != "v_prediction") {
    error = "unsupported prediction type '" + cfg.predictionType + "'";
    return std::nullopt;
  }

  Scheduler s;
  s.kind_ = kind;
  s.cfg_ = cfg;
  s.stepRatio_ = T / steps;

  // The beta ladder, then its cumulative product. `scaled_linear` interpolates the square roots
  // of the endpoints — the schedule Stable Diffusion was trained with, and not the same curve as
  // interpolating the endpoints themselves.
  s.alphasCumprod_.resize(static_cast<std::size_t>(T));
  double cumulative = 1.0;
  for (int i = 0; i < T; ++i) {
    const double frac = T > 1 ? static_cast<double>(i) / (T - 1) : 0.0;
    double beta;
    if (cfg.betaSchedule == "scaled_linear") {
      const double root = std::sqrt(cfg.betaStart) + frac * (std::sqrt(cfg.betaEnd) - std::sqrt(cfg.betaStart));
      beta = root * root;
    } else if (cfg.betaSchedule == "linear") {
      beta = cfg.betaStart + frac * (cfg.betaEnd - cfg.betaStart);
    } else {
      error = "unsupported beta schedule '" + cfg.betaSchedule + "'";
      return std::nullopt;
    }
    cumulative *= (1.0 - beta);
    s.alphasCumprod_[static_cast<std::size_t>(i)] = static_cast<float>(cumulative);
  }
  // What the step BEFORE zero is worth. `set_alpha_to_one` says "treat it as pure signal";
  // otherwise the first entry of the ladder stands in.
  s.finalAlphaCumprod_ = cfg.setAlphaToOne ? 1.0f : s.alphasCumprod_.front();

  // The visited timesteps. Three spacings exist and they are NOT cosmetic: `leading` (SD 1.x's
  // default) starts at 981 for 20 steps, `trailing` starts at 999, and the difference shows up as
  // a systematic over- or under-denoising at the first step.
  s.timesteps_.reserve(static_cast<std::size_t>(steps));
  if (cfg.timestepSpacing == "leading") {
    for (int i = steps - 1; i >= 0; --i) {
      s.timesteps_.push_back(i * s.stepRatio_ + cfg.stepsOffset);
    }
  } else if (cfg.timestepSpacing == "trailing") {
    const double ratio = static_cast<double>(T) / steps;
    for (int i = 0; i < steps; ++i) {
      s.timesteps_.push_back(static_cast<int>(std::round(T - i * ratio)) - 1);
    }
  } else if (cfg.timestepSpacing == "linspace") {
    for (int i = steps - 1; i >= 0; --i) {
      s.timesteps_.push_back(
          static_cast<int>(std::round(static_cast<double>(i) * (T - 1) / (steps - 1 ? steps - 1 : 1))));
    }
  } else {
    error = "unsupported timestep spacing '" + cfg.timestepSpacing + "'";
    return std::nullopt;
  }

  if (kind == SamplerKind::Ddim) {
    s.initNoiseSigma_ = 1.0f;
    return s;
  }

  // The Euler samplers do not work in the model's own latent units; they work in sigma space,
  // where sigma = sqrt((1 - alpha_bar) / alpha_bar). The ladder is sampled at the (generally
  // fractional) training steps this run visits, and closed with an explicit zero so the last step
  // lands exactly on the clean sample.
  std::vector<float> full(static_cast<std::size_t>(T));
  for (int i = 0; i < T; ++i) {
    const double a = s.alphasCumprod_[static_cast<std::size_t>(i)];
    full[static_cast<std::size_t>(i)] = static_cast<float>(std::sqrt((1.0 - a) / a));
  }
  s.sigmas_.reserve(static_cast<std::size_t>(steps) + 1);
  for (int t : s.timesteps_) s.sigmas_.push_back(interpolateSigma(full, t));
  s.sigmas_.push_back(0.0f);

  const float maxSigma = *std::max_element(s.sigmas_.begin(), s.sigmas_.end());
  // Only `leading` needs the sqrt(sigma^2 + 1) form — the other two spacings start the latent at
  // the largest sigma directly. Copied from the reference implementation rather than derived,
  // because it is a compatibility fact about existing checkpoints, not a property of the maths.
  s.initNoiseSigma_ = cfg.timestepSpacing == "leading"
                          ? static_cast<float>(std::sqrt(maxSigma * maxSigma + 1.0))
                          : maxSigma;
  return s;
}

void Scheduler::scaleModelInput(FeatureMap& x, int stepIndex) const {
  if (kind_ == SamplerKind::Ddim) return;
  if (stepIndex < 0 || stepIndex >= static_cast<int>(sigmas_.size())) return;
  const float sigma = sigmas_[static_cast<std::size_t>(stepIndex)];
  const float scale = 1.0f / static_cast<float>(std::sqrt(sigma * sigma + 1.0));
  for (float& v : x.data) v *= scale;
}

bool Scheduler::step(FeatureMap& sample, const FeatureMap& modelOutput, int stepIndex,
                     GaussianRng& rng, std::string& error) {
  if (stepIndex < 0 || stepIndex >= steps()) {
    error = "step index " + std::to_string(stepIndex) + " is outside the schedule";
    return false;
  }
  if (!sample.matches(modelOutput)) {
    error = "the model's prediction does not match the latent's shape";
    return false;
  }
  const std::size_t n = sample.size();
  const bool vPred = cfg_.predictionType == "v_prediction";

  if (kind_ == SamplerKind::Ddim) {
    const int t = timesteps_[static_cast<std::size_t>(stepIndex)];
    const int prev = t - stepRatio_;
    const double alphaT = alphasCumprod_[static_cast<std::size_t>(std::clamp(t, 0, static_cast<int>(alphasCumprod_.size()) - 1))];
    const double alphaPrev = prev >= 0 ? alphasCumprod_[static_cast<std::size_t>(prev)] : finalAlphaCumprod_;
    const double betaT = 1.0 - alphaT;
    const double sqrtAlphaT = std::sqrt(alphaT);
    const double sqrtBetaT = std::sqrt(betaT);
    const double sqrtAlphaPrev = std::sqrt(alphaPrev);
    const double dirScale = std::sqrt(1.0 - alphaPrev);
    for (std::size_t i = 0; i < n; ++i) {
      const double x = sample.data[i];
      const double m = modelOutput.data[i];
      // pred_original is the model's guess at the CLEAN latent; pred_epsilon is its guess at the
      // noise. Under v-prediction both are mixtures, which is the whole reason the two are
      // tracked separately rather than one derived from the other at the end.
      double predOriginal, predEpsilon;
      if (vPred) {
        predOriginal = sqrtAlphaT * x - sqrtBetaT * m;
        predEpsilon = sqrtAlphaT * m + sqrtBetaT * x;
      } else {
        predOriginal = (x - sqrtBetaT * m) / sqrtAlphaT;
        predEpsilon = m;
      }
      sample.data[i] = static_cast<float>(sqrtAlphaPrev * predOriginal + dirScale * predEpsilon);
    }
    return true;
  }

  const double sigma = sigmas_[static_cast<std::size_t>(stepIndex)];
  const double sigmaNext = sigmas_[static_cast<std::size_t>(stepIndex) + 1];
  double dt = sigmaNext - sigma;
  double noiseScale = 0.0;
  if (kind_ == SamplerKind::EulerAncestral) {
    // Split the move to sigmaNext into a deterministic part and a fresh-noise part. sigma_up is
    // how much noise is put back; sigma_down is how far the deterministic move goes.
    const double up = sigma > 0.0
                          ? std::sqrt(sigmaNext * sigmaNext * (sigma * sigma - sigmaNext * sigmaNext) /
                                      (sigma * sigma))
                          : 0.0;
    const double down = std::sqrt(std::max(0.0, sigmaNext * sigmaNext - up * up));
    dt = down - sigma;
    noiseScale = up;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double x = sample.data[i];
    const double m = modelOutput.data[i];
    const double predOriginal = vPred ? m * (-sigma / std::sqrt(sigma * sigma + 1.0)) +
                                            x / (sigma * sigma + 1.0)
                                      : x - sigma * m;
    // The Euler derivative: the direction from the current sample toward the model's clean guess,
    // normalized by how far away it is.
    const double derivative = sigma > 0.0 ? (x - predOriginal) / sigma : 0.0;
    double next = x + derivative * dt;
    if (noiseScale > 0.0) next += noiseScale * rng.normal();
    sample.data[i] = static_cast<float>(next);
  }
  return true;
}

}  // namespace qorvix::image
