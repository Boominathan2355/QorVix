#include "qorvix/image/sd_check.hpp"

#include <fstream>
#include <sstream>

namespace qorvix::image {

namespace {

// Fixtures are committed as text and checked out with the platform's line endings, so a stray
// carriage return would otherwise end up inside a prompt this file compares character for
// character.
void stripCr(std::string& s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
}

void readFloats(std::istringstream& ls, std::vector<float>& dst) {
  float v = 0.0f;
  while (ls >> v) dst.push_back(v);
}

void readInts(std::istringstream& ls, std::vector<int>& dst) {
  int v = 0;
  while (ls >> v) dst.push_back(v);
}

}  // namespace

bool SdReference::load(const std::filesystem::path& path, std::string& error) {
  error.clear();
  std::ifstream in(path);
  if (!in) {
    error = "cannot open '" + path.string() + "'";
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    stripCr(line);
    if (line.empty() || line[0] == '#') continue;

    // The two prompts are the raw remainder of their line: they contain spaces, and a
    // stream-extracted first word would silently compare only "a".
    if (line.rfind("prompt ", 0) == 0) {
      prompt = line.substr(7);
      continue;
    }
    if (line.rfind("negative ", 0) == 0) {
      negative = line.substr(9);
      continue;
    }

    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "model") {
      ls >> model;
    } else if (key == "sampler") {
      ls >> sampler;
    } else if (key == "size") {
      ls >> width >> height;
    } else if (key == "steps") {
      ls >> steps;
    } else if (key == "clip_skip") {
      ls >> clipSkip;
    } else if (key == "guidance") {
      ls >> guidance;
    } else if (key == "tokens") {
      readInts(ls, tokens);
    } else if (key == "neg_tokens") {
      readInts(ls, negTokens);
    } else if (key == "timesteps") {
      readInts(ls, timesteps);
    } else if (key == "alpha_probe") {
      AlphaProbe p;
      ls >> p.index >> p.value;
      alphaProbes.push_back(p);
    } else if (key == "cond_row0") {
      readFloats(ls, condRow0);
    } else if (key == "cond_dim_means") {
      readFloats(ls, condDimMeans);
    } else if (key == "uncond_row0") {
      readFloats(ls, uncondRow0);
    } else if (key == "latent_shape") {
      ls >> latentC >> latentH >> latentW;
    } else if (key == "latent") {
      readFloats(ls, latent);
    } else if (key == "unet_t") {
      ls >> unetTimestep;
    } else if (key == "unet_row0") {
      readFloats(ls, unetRow0);
    } else if (key == "unet_channel_means") {
      readFloats(ls, unetChannelMeans);
    } else if (key == "final_latent") {
      readFloats(ls, finalLatent);
    } else if (key == "image_shape") {
      ls >> imageH >> imageW;
    } else if (key == "image_row0") {
      readFloats(ls, imageRow0);
    } else if (key == "image_channel_means") {
      readFloats(ls, imageChannelMeans);
    }
  }

  if (steps <= 0 || tokens.empty() || timesteps.empty() || condRow0.empty() || latent.empty()) {
    error = "malformed sd fixture (steps " + std::to_string(steps) + " tokens " +
            std::to_string(tokens.size()) + " timesteps " + std::to_string(timesteps.size()) +
            " cond_row0 " + std::to_string(condRow0.size()) + " latent " +
            std::to_string(latent.size()) + ")";
    return false;
  }
  // The header and the payload have to agree. Without this a fixture captured on one model could
  // be compared against another's forward pass and fail as a numerical mismatch rather than as
  // the wrong-pairing mistake it is.
  const std::size_t expectLatent = static_cast<std::size_t>(latentC) * latentH * latentW;
  if (latent.size() != expectLatent || (!finalLatent.empty() && finalLatent.size() != expectLatent)) {
    error = "sd fixture latent has " + std::to_string(latent.size()) + " values but its shape says " +
            std::to_string(expectLatent);
    return false;
  }
  if (static_cast<int>(timesteps.size()) != steps) {
    error = "sd fixture lists " + std::to_string(timesteps.size()) + " timesteps for " +
            std::to_string(steps) + " steps";
    return false;
  }
  if (!condDimMeans.empty() && condDimMeans.size() != condRow0.size()) {
    error = "sd fixture conditioning rows disagree with each other";
    return false;
  }
  return true;
}

}  // namespace qorvix::image
