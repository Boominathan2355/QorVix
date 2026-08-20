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

// ---- writing ---------------------------------------------------------------------------------
// Phase 11b-3c needs an image OUT of the runtime, not just in. It lives here rather than in
// `image/` because it is the same format the decoder above reads: one module owns PNG, and the
// round-trip (encode then decode, pixel-for-pixel) is the test that keeps both honest.
//
// The output is a correct PNG that any decoder reads. It is not a SMALL one: the DEFLATE stream
// is stored (uncompressed) blocks, so a 512x512 image is about 790 KiB where libpng would write
// ~400 KiB. Compression is a size property, not a correctness one, and an LZ77 matcher belongs
// with a round-trip test against this repo's own inflate rather than bolted on here — the filter
// bytes are written as 0 (None) for the same reason, since row filters only exist to make a
// compressor's job easier and there is no compressor yet to help.
bool encodePng(const Image& img, std::vector<std::uint8_t>& out, std::string& error);

bool savePng(const Image& img, const std::filesystem::path& path, std::string& error);

// CRC-32 as specified by PNG (RFC 1952 polynomial), exposed for tests.
std::uint32_t crc32Png(const std::uint8_t* data, std::size_t size);

}  // namespace qorvix::vision
