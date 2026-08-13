#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace qorvix::vision {

// Parsed reference fixture from scripts/capture_vision_reference.py.
//
// Deliberately split into a preprocessing tier and a transformer tier. Preprocessing is where the
// vision path is most likely to be subtly wrong — a different resize filter or crop origin moves
// every output value and errors on nothing — so a single "features differ" verdict would leave the
// cause ambiguous. With both recorded, a failure says which half.
struct VisionReference {
  std::string model;
  int imageSize = 0;
  int patches = 0;
  int dim = 0;

  float pixelMean = 0.0f;
  float pixelAbsMean = 0.0f;
  struct PixelProbe {
    int c = 0, y = 0, x = 0;
    float value = 0.0f;
  };
  std::vector<PixelProbe> pixelProbes;

  std::vector<float> row0;       // first patch token, in full
  std::vector<float> rowMeans;   // per-patch means, one per patch token

  bool load(const std::filesystem::path& path, std::string& error);
};

}  // namespace qorvix::vision
