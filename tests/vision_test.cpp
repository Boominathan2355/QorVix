#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "qorvix/vision/image.hpp"
#include "qorvix/vision/inflate.hpp"
#include "qorvix/vision/preprocess.hpp"

using namespace qorvix::vision;
using Catch::Matchers::WithinAbs;

namespace {

std::vector<std::uint8_t> bytes(std::initializer_list<int> v) {
  std::vector<std::uint8_t> out;
  out.reserve(v.size());
  for (int b : v) out.push_back(static_cast<std::uint8_t>(b));
  return out;
}

}  // namespace

// ---- DEFLATE --------------------------------------------------------------------------------

TEST_CASE("inflate decodes a stored (uncompressed) block", "[vision]") {
  // BFINAL=1, BTYPE=00, pad to byte, LEN=5, NLEN=~5, then "hello".
  const auto in = bytes({0x01, 0x05, 0x00, 0xFA, 0xFF, 'h', 'e', 'l', 'l', 'o'});
  std::vector<std::uint8_t> out;
  std::string err;
  REQUIRE(inflateRaw(in.data(), in.size(), out, err));
  REQUIRE(std::string(out.begin(), out.end()) == "hello");
}

TEST_CASE("inflate rejects a stored block whose length check fails", "[vision]") {
  // NLEN must be the one's complement of LEN; a mismatch means the stream is corrupt, and
  // trusting LEN anyway would read arbitrary following bytes as pixel data.
  const auto in = bytes({0x01, 0x05, 0x00, 0x00, 0x00, 'h', 'e', 'l', 'l', 'o'});
  std::vector<std::uint8_t> out;
  std::string err;
  REQUIRE_FALSE(inflateRaw(in.data(), in.size(), out, err));
  REQUIRE(err.find("length check") != std::string::npos);
}

TEST_CASE("inflate decodes fixed-huffman output from a real compressor", "[vision]") {
  // Produced by zlib itself, not written by hand — a hand-assembled bit stream tests whether I
  // can pack Huffman codes, which is not the thing under test. Exercises the fixed-Huffman path
  // AND a back-reference whose length exceeds its distance (the overlapping copy a memcpy would
  // get wrong).
  const auto in = bytes({0x78, 0x01, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01, 0x68, 0x03, 0x08, 0xb1});
  std::vector<std::uint8_t> out;
  std::string err;
  REQUIRE(inflateZlib(in.data(), in.size(), out, err));
  REQUIRE(std::string(out.begin(), out.end()) == "hello hello hello hello");
}

TEST_CASE("inflate decodes dynamic-huffman output from a real compressor", "[vision]") {
  // The path PNG actually uses: a per-block code table, itself Huffman-coded through the
  // permuted code-length alphabet.
  const auto in = bytes({0x78, 0xda, 0x2b, 0xc9, 0x48, 0x55, 0x28, 0x2c, 0xcd, 0x4c, 0xce, 0x56, 0x48, 0x2a, 0xca, 0x2f, 0xcf, 0x53, 0x48, 0xcb, 0xaf, 0x50, 0xc8, 0x2a, 0xcd, 0x2d, 0x28, 0x56, 0xc8, 0x2f, 0x4b, 0x2d, 0x52, 0x28, 0x01, 0x4a, 0xe7, 0x24, 0x56, 0x55, 0x2a, 0xa4, 0xe4, 0xa7, 0xeb, 0x81, 0x79, 0xc3, 0x5a, 0x31, 0x00, 0xf7, 0xe0, 0x61, 0xab});
  std::vector<std::uint8_t> out;
  std::string err;
  REQUIRE(inflateZlib(in.data(), in.size(), out, err));
  std::string expect;
  for (int i = 0; i < 6; ++i) expect += "the quick brown fox jumps over the lazy dog. ";
  REQUIRE(std::string(out.begin(), out.end()) == expect);
}

TEST_CASE("inflate verifies the zlib adler-32 trailer", "[vision]") {
  // A silently corrupt image is worse than a rejected one: it becomes a plausible wrong
  // embedding rather than an error anyone can see.
  auto in = bytes({0x78, 0x01, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01, 0x68, 0x03, 0x08, 0xb1});
  in.back() ^= 0xFF;
  std::vector<std::uint8_t> out;
  std::string err;
  REQUIRE_FALSE(inflateZlib(in.data(), in.size(), out, err));
  REQUIRE(err.find("checksum") != std::string::npos);
}

TEST_CASE("adler32 matches the RFC 1950 definition", "[vision]") {
  REQUIRE(adler32(reinterpret_cast<const std::uint8_t*>(""), 0) == 1u);
  // "Wikipedia" is the worked example in the specification's own documentation.
  const std::string s = "Wikipedia";
  REQUIRE(adler32(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()) == 0x11E60398u);
}

TEST_CASE("inflate refuses a reserved block type and a truncated stream", "[vision]") {
  std::vector<std::uint8_t> out;
  std::string err;
  const auto reserved = bytes({0x07});  // BFINAL=1, BTYPE=11
  REQUIRE_FALSE(inflateRaw(reserved.data(), reserved.size(), out, err));

  const auto truncated = bytes({0x01, 0x05, 0x00, 0xFA, 0xFF, 'h'});
  REQUIRE_FALSE(inflateRaw(truncated.data(), truncated.size(), out, err));
}

// ---- PNG ------------------------------------------------------------------------------------

namespace {

// Builds a minimal 8-bit truecolour PNG with a single all-zero (None) filter per row, so the test
// does not depend on a compressor being available.
std::vector<std::uint8_t> makePng(int w, int h, const std::vector<std::uint8_t>& rgb) {
  auto beU32 = [](std::vector<std::uint8_t>& v, std::uint32_t x) {
    v.push_back(static_cast<std::uint8_t>(x >> 24));
    v.push_back(static_cast<std::uint8_t>(x >> 16));
    v.push_back(static_cast<std::uint8_t>(x >> 8));
    v.push_back(static_cast<std::uint8_t>(x));
  };
  // Raw scanlines with a filter byte, wrapped in a zlib stream of STORED deflate blocks.
  std::vector<std::uint8_t> raw;
  for (int y = 0; y < h; ++y) {
    raw.push_back(0);  // filter: None
    for (int x = 0; x < w * 3; ++x) {
      raw.push_back(rgb[static_cast<std::size_t>(y) * w * 3 + x]);
    }
  }
  std::vector<std::uint8_t> z{0x78, 0x01};
  z.push_back(0x01);  // BFINAL=1, stored
  z.push_back(static_cast<std::uint8_t>(raw.size() & 0xFF));
  z.push_back(static_cast<std::uint8_t>((raw.size() >> 8) & 0xFF));
  z.push_back(static_cast<std::uint8_t>(~raw.size() & 0xFF));
  z.push_back(static_cast<std::uint8_t>((~raw.size() >> 8) & 0xFF));
  z.insert(z.end(), raw.begin(), raw.end());
  const std::uint32_t a = adler32(raw.data(), raw.size());
  z.push_back(static_cast<std::uint8_t>(a >> 24));
  z.push_back(static_cast<std::uint8_t>(a >> 16));
  z.push_back(static_cast<std::uint8_t>(a >> 8));
  z.push_back(static_cast<std::uint8_t>(a));

  std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  auto chunk = [&](const char* type, const std::vector<std::uint8_t>& body) {
    beU32(png, static_cast<std::uint32_t>(body.size()));
    png.insert(png.end(), type, type + 4);
    png.insert(png.end(), body.begin(), body.end());
    beU32(png, 0);  // CRC is not verified by the decoder; zlib's adler already covers the pixels
  };
  std::vector<std::uint8_t> ihdr;
  beU32(ihdr, static_cast<std::uint32_t>(w));
  beU32(ihdr, static_cast<std::uint32_t>(h));
  ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // depth 8, colour type 2, no interlace
  chunk("IHDR", ihdr);
  chunk("IDAT", z);
  chunk("IEND", {});
  return png;
}

}  // namespace

TEST_CASE("png decodes an uncompressed truecolour image", "[vision]") {
  const std::vector<std::uint8_t> rgb{255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
  const auto png = makePng(2, 2, rgb);

  Image img;
  std::string err;
  REQUIRE(decodePng(png.data(), png.size(), img, err));
  REQUIRE(img.width == 2);
  REQUIRE(img.height == 2);
  REQUIRE(img.rgb == rgb);
}

TEST_CASE("png rejects a file that is not a png, and an interlaced one", "[vision]") {
  Image img;
  std::string err;
  const auto junk = bytes({1, 2, 3, 4, 5, 6, 7, 8, 9});
  REQUIRE_FALSE(decodePng(junk.data(), junk.size(), img, err));

  auto png = makePng(2, 2, std::vector<std::uint8_t>(12, 0));
  png[8 + 8 + 12] = 1;  // IHDR interlace byte
  REQUIRE_FALSE(decodePng(png.data(), png.size(), img, err));
  REQUIRE(err.find("interlaced") != std::string::npos);
}

TEST_CASE("image dispatch names JPEG rather than reporting an unknown format", "[vision]") {
  // An error a user can act on. "unrecognized format" for the most common photo format on earth
  // sends them looking for a corrupt file instead of a missing feature.
  const auto jpeg = bytes({0xFF, 0xD8, 0xFF, 0xE0, 0, 16, 'J', 'F', 'I', 'F'});
  Image img;
  std::string err;
  REQUIRE_FALSE(decodeImage(jpeg.data(), jpeg.size(), img, err));
  REQUIRE(err.find("JPEG") != std::string::npos);
}

// ---- preprocessing --------------------------------------------------------------------------

namespace {

Image solid(int w, int h, std::uint8_t r, std::uint8_t g, std::uint8_t b) {
  Image img;
  img.width = w;
  img.height = h;
  img.rgb.resize(static_cast<std::size_t>(w) * h * 3);
  for (int i = 0; i < w * h; ++i) {
    img.rgb[static_cast<std::size_t>(i) * 3] = r;
    img.rgb[static_cast<std::size_t>(i) * 3 + 1] = g;
    img.rgb[static_cast<std::size_t>(i) * 3 + 2] = b;
  }
  return img;
}

}  // namespace

TEST_CASE("preprocessing emits CHW planes at the configured size", "[vision]") {
  PreprocessConfig cfg;
  cfg.size = 28;
  std::vector<float> out;
  std::string err;
  REQUIRE(preprocessClip(solid(50, 40, 10, 20, 30), cfg, out, err));
  REQUIRE(out.size() == static_cast<std::size_t>(3) * 28 * 28);
}

TEST_CASE("preprocessing normalizes each channel with its own mean and std", "[vision]") {
  // A solid colour makes the expected value exact: every pixel of channel c must be
  // (value/255 - mean[c]) / std[c], so a swapped or shared constant shows up immediately.
  PreprocessConfig cfg;
  cfg.size = 16;
  std::vector<float> out;
  std::string err;
  REQUIRE(preprocessClip(solid(64, 64, 255, 128, 0), cfg, out, err));

  const std::size_t plane = 16 * 16;
  const float expect0 = (1.0f - cfg.mean[0]) / cfg.std[0];
  const float expect1 = (128.0f / 255.0f - cfg.mean[1]) / cfg.std[1];
  const float expect2 = (0.0f - cfg.mean[2]) / cfg.std[2];
  REQUIRE_THAT(out[0], WithinAbs(expect0, 1e-4f));
  REQUIRE_THAT(out[plane], WithinAbs(expect1, 1e-4f));
  REQUIRE_THAT(out[2 * plane], WithinAbs(expect2, 1e-4f));
  // Uniform input stays uniform: the resize must not introduce edge artefacts.
  REQUIRE_THAT(out[plane - 1], WithinAbs(expect0, 1e-4f));
}

TEST_CASE("preprocessing resizes the shortest side and centre-crops the longer one", "[vision]") {
  // A wide image: after scaling the short side to `size`, the crop must take the MIDDLE of the
  // width. Painting the centre column differently proves the crop origin rather than assuming it.
  PreprocessConfig cfg;
  cfg.size = 8;
  cfg.mean = {0.0f, 0.0f, 0.0f};
  cfg.std = {1.0f, 1.0f, 1.0f};

  Image img = solid(80, 20, 0, 0, 0);
  for (int y = 0; y < 20; ++y) {
    for (int x = 30; x < 50; ++x) {  // the middle quarter, which is what a centre crop keeps
      img.rgb[(static_cast<std::size_t>(y) * 80 + x) * 3] = 255;
    }
  }
  std::vector<float> out;
  std::string err;
  REQUIRE(preprocessClip(img, cfg, out, err));
  // Centre of the crop comes from the painted band -> near 1.0 after rescale.
  REQUIRE(out[4 * 8 + 4] > 0.9f);
}

TEST_CASE("a bicubic resize that changes nothing passes the image through unchanged", "[vision]") {
  // Pillow skips a pass whose axis is unchanged; running a unit-scale filter anyway would blur an
  // image that should have been copied, which is invisible until it moves an embedding.
  const Image src = solid(9, 7, 200, 100, 50);
  Image out;
  std::string err;
  REQUIRE(resizeBicubic(src, 9, 7, out, err));
  REQUIRE(out.rgb == src.rgb);
}
