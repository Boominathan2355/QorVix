#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace qorvix::vision {

// An 8-bit RGB image, row-major, 3 bytes per pixel. Alpha is composited away at load time: a
// vision encoder has no alpha channel, and carrying one only defers the question of what to
// composite against (white, matching PIL's default when converting RGBA to RGB).
struct Image {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;  // [height * width * 3]

  bool valid() const {
    return width > 0 && height > 0 &&
           rgb.size() == static_cast<std::size_t>(width) * height * 3;
  }
  const std::uint8_t* pixel(int y, int x) const {
    return rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
  }
};

// Decodes PNG (all non-interlaced colour types, 8- and 16-bit depths), BMP (24/32-bit
// uncompressed), and binary PPM. Dispatches on content — the file's magic bytes — not on the
// extension, since a mislabelled file is common and the magic is authoritative.
bool decodeImage(const std::uint8_t* data, std::size_t size, Image& out, std::string& error);

bool loadImage(const std::filesystem::path& path, Image& out, std::string& error);

// Exposed for tests and for callers that already know the format.
bool decodePng(const std::uint8_t* data, std::size_t size, Image& out, std::string& error);

}  // namespace qorvix::vision
