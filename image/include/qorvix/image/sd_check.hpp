#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace qorvix::image {

// Parsed reference trace from scripts/capture_sd_reference.py (diffusers, fp32 CPU torch).
//
// Tiered for the reason every *-check in this repo is, and with the sharpest version of the
// problem yet: a diffusion model has five things that can be wrong and ONE observable, a picture.
// A wrong beta schedule, a wrong BPE split, a bidirectional text encoder, a skip stack off by one
// and a swapped GEGLU half all present identically — an image that is plausible and not the one
// every other runtime produces. Ordered so each tier can only fail for causes the ones above it
// have cleared:
//
//   schedule -> tokenizer -> conditioning -> one UNet step -> the whole loop -> the VAE decode
//
// The starting latent travels IN the fixture, so the comparison contains no random number
// generator at all — see image/rng.hpp for why matching torch's would be the wrong goal.
struct SdReference {
  std::string model;
  std::string prompt, negative, sampler;
  int width = 0, height = 0;
  int steps = 0;
  int clipSkip = 1;
  float guidance = 7.5f;

  std::vector<int> tokens, negTokens;
  std::vector<int> timesteps;

  struct AlphaProbe {
    int index = 0;
    float value = 0.0f;
  };
  std::vector<AlphaProbe> alphaProbes;

  std::vector<float> condRow0, condDimMeans, uncondRow0;

  int latentC = 0, latentH = 0, latentW = 0;
  std::vector<float> latent;       // position-major, matching FeatureMap
  int unetTimestep = 0;
  std::vector<float> unetRow0, unetChannelMeans;
  std::vector<float> finalLatent;  // position-major

  int imageH = 0, imageW = 0;
  std::vector<float> imageRow0, imageChannelMeans;

  bool load(const std::filesystem::path& path, std::string& error);
};

}  // namespace qorvix::image
