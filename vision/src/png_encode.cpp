#include <cstdio>
#include <fstream>

#include "qorvix/vision/image.hpp"
#include "qorvix/vision/inflate.hpp"

// PNG writing: the other half of image.cpp's reader, kept in its own translation unit because the
// two share nothing but the format spec — the decoder is a state machine over chunks, this is a
// serializer.
namespace qorvix::vision {

namespace {

// PNG's CRC-32 (RFC 1952 polynomial 0xEDB88320, reflected). Built once on first use rather than
// baked in as a 1 KiB literal table.
struct Crc32Table {
  std::uint32_t v[256];
  Crc32Table() {
    for (std::uint32_t n = 0; n < 256; ++n) {
      std::uint32_t c = n;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      v[n] = c;
    }
  }
};

void put32(std::vector<std::uint8_t>& out, std::uint32_t v) {
  out.push_back(static_cast<std::uint8_t>(v >> 24));
  out.push_back(static_cast<std::uint8_t>(v >> 16));
  out.push_back(static_cast<std::uint8_t>(v >> 8));
  out.push_back(static_cast<std::uint8_t>(v));
}

// One PNG chunk: length, type, payload, CRC over (type + payload). The CRC deliberately covers
// the type as well as the data — a chunk whose type byte flipped is a different chunk, and a
// checksum that did not see the type would call it intact.
void chunk(std::vector<std::uint8_t>& out, const char (&type)[5],
           const std::vector<std::uint8_t>& payload) {
  put32(out, static_cast<std::uint32_t>(payload.size()));
  const std::size_t crcStart = out.size();
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(type[i]));
  out.insert(out.end(), payload.begin(), payload.end());
  put32(out, crc32Png(out.data() + crcStart, out.size() - crcStart));
}

// A zlib stream (RFC 1950) whose DEFLATE payload is stored blocks (RFC 1951 §3.2.4). Every block
// carries at most 65535 bytes, so the raster is emitted in chunks of that size with BFINAL set on
// the last one. Stored blocks are byte-aligned by construction, which is the whole reason this is
// three lines of bit twiddling instead of a Huffman coder.
void deflateStored(const std::vector<std::uint8_t>& raw, std::vector<std::uint8_t>& out) {
  out.push_back(0x78);  // CM = 8 (deflate), CINFO = 7 (32 KiB window)
  out.push_back(0x01);  // FLEVEL = 0, FDICT = 0, FCHECK chosen so 0x7801 % 31 == 0
  std::size_t pos = 0;
  do {
    const std::size_t n = std::min<std::size_t>(65535, raw.size() - pos);
    const bool last = pos + n >= raw.size();
    out.push_back(last ? 1 : 0);  // BFINAL in bit 0, BTYPE = 00 in bits 1-2
    out.push_back(static_cast<std::uint8_t>(n));
    out.push_back(static_cast<std::uint8_t>(n >> 8));
    out.push_back(static_cast<std::uint8_t>(~n));
    out.push_back(static_cast<std::uint8_t>(~n >> 8));
    out.insert(out.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos),
               raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
    pos += n;
  } while (pos < raw.size());  // do/while: a zero-byte payload still needs one final block
  put32(out, adler32(raw.data(), raw.size()));
}

}  // namespace

std::uint32_t crc32Png(const std::uint8_t* data, std::size_t size) {
  static const Crc32Table table;
  std::uint32_t c = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < size; ++i) c = table.v[(c ^ data[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

bool encodePng(const Image& img, std::vector<std::uint8_t>& out, std::string& error) {
  error.clear();
  out.clear();
  if (!img.valid()) {
    error = "image is empty or its pixel buffer does not match its dimensions";
    return false;
  }

  static const std::uint8_t kSignature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  out.insert(out.end(), kSignature, kSignature + 8);

  std::vector<std::uint8_t> ihdr;
  put32(ihdr, static_cast<std::uint32_t>(img.width));
  put32(ihdr, static_cast<std::uint32_t>(img.height));
  ihdr.push_back(8);  // bit depth
  ihdr.push_back(2);  // colour type 2 = truecolour RGB, matching Image's 3 bytes per pixel
  ihdr.push_back(0);  // compression method: deflate, the only one PNG defines
  ihdr.push_back(0);  // filter method 0
  ihdr.push_back(0);  // no interlacing
  chunk(out, "IHDR", ihdr);

  // The raster: one filter byte per scanline, then the row. Filter 0 (None) throughout — see the
  // header comment for why filtering without a compressor behind it buys nothing.
  const std::size_t stride = static_cast<std::size_t>(img.width) * 3;
  std::vector<std::uint8_t> raw;
  raw.reserve((stride + 1) * static_cast<std::size_t>(img.height));
  for (int y = 0; y < img.height; ++y) {
    raw.push_back(0);
    const std::uint8_t* row = img.rgb.data() + static_cast<std::size_t>(y) * stride;
    raw.insert(raw.end(), row, row + stride);
  }

  std::vector<std::uint8_t> idat;
  deflateStored(raw, idat);
  chunk(out, "IDAT", idat);
  chunk(out, "IEND", {});
  return true;
}

bool savePng(const Image& img, const std::filesystem::path& path, std::string& error) {
  std::vector<std::uint8_t> bytes;
  if (!encodePng(img, bytes, error)) return false;
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    error = "cannot open '" + path.string() + "' for writing";
    return false;
  }
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!f) {
    error = "failed while writing '" + path.string() + "'";
    return false;
  }
  return true;
}

}  // namespace qorvix::vision
