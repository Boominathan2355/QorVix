#include "qorvix/audio/audio_check.hpp"

#include <fstream>
#include <sstream>

namespace qorvix::audio {

bool AudioReference::load(const std::filesystem::path& path, std::string& error) {
  error.clear();
  std::ifstream in(path);
  if (!in) {
    error = "cannot open '" + path.string() + "'";
    return false;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    if (key == "model") {
      ls >> model;
    } else if (key == "sample_rate") {
      ls >> sampleRate;
    } else if (key == "n_fft") {
      ls >> nFft;
    } else if (key == "hop_length") {
      ls >> hopLength;
    } else if (key == "n_mels") {
      ls >> nMels;
    } else if (key == "frames") {
      ls >> frames;
    } else if (key == "waveform_mean") {
      ls >> waveformMean;
    } else if (key == "waveform_absmean") {
      ls >> waveformAbsMean;
    } else if (key == "waveform_samples") {
      ls >> waveformSamples;
    } else if (key == "filter") {
      FilterProbe p;
      ls >> p.bin >> p.mel >> p.value;
      filterProbes.push_back(p);
    } else if (key == "filtersums") {
      float v = 0.0f;
      while (ls >> v) filterColumnSums.push_back(v);
    } else if (key == "frame0") {
      float v = 0.0f;
      while (ls >> v) frame0.push_back(v);
    } else if (key == "melmeans") {
      float v = 0.0f;
      while (ls >> v) melMeans.push_back(v);
    }
  }
  if (nMels <= 0 || frames <= 0 || frame0.empty() || melMeans.empty()) {
    error = "malformed audio fixture (n_mels=" + std::to_string(nMels) +
            " frames=" + std::to_string(frames) + " frame0=" + std::to_string(frame0.size()) +
            " melmeans=" + std::to_string(melMeans.size()) + ")";
    return false;
  }
  // The header and the payload have to agree, or a fixture captured at one config could be
  // silently compared against features computed at another — which would read as a numerical
  // failure rather than the mismatched-inputs mistake it is.
  if (static_cast<int>(frame0.size()) != nMels || static_cast<int>(melMeans.size()) != nMels) {
    error = "audio fixture shapes disagree with its own header";
    return false;
  }
  if (!filterColumnSums.empty() && static_cast<int>(filterColumnSums.size()) != nMels) {
    error = "audio fixture filter sums disagree with n_mels";
    return false;
  }
  return true;
}

}  // namespace qorvix::audio
