#include "qorvix/vision/vision_check.hpp"

#include <fstream>
#include <sstream>

namespace qorvix::vision {

bool VisionReference::load(const std::filesystem::path& path, std::string& error) {
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
    } else if (key == "image_size") {
      ls >> imageSize;
    } else if (key == "patches") {
      ls >> patches;
    } else if (key == "dim") {
      ls >> dim;
    } else if (key == "pixel_mean") {
      ls >> pixelMean;
    } else if (key == "pixel_absmean") {
      ls >> pixelAbsMean;
    } else if (key == "pixel") {
      PixelProbe p;
      ls >> p.c >> p.y >> p.x >> p.value;
      pixelProbes.push_back(p);
    } else if (key == "row0") {
      float v = 0.0f;
      while (ls >> v) row0.push_back(v);
    } else if (key == "rowmeans") {
      float v = 0.0f;
      while (ls >> v) rowMeans.push_back(v);
    }
  }
  if (patches <= 0 || dim <= 0 || row0.empty() || rowMeans.empty()) {
    error = "malformed vision fixture (patches=" + std::to_string(patches) + " dim=" +
            std::to_string(dim) + " row0=" + std::to_string(row0.size()) + " rowmeans=" +
            std::to_string(rowMeans.size()) + ")";
    return false;
  }
  if (static_cast<int>(row0.size()) != dim || static_cast<int>(rowMeans.size()) != patches) {
    error = "vision fixture shapes disagree with its own header";
    return false;
  }
  return true;
}

}  // namespace qorvix::vision
