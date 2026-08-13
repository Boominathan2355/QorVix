#include "qorvix/vision/image.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "qorvix/vision/inflate.hpp"

namespace fs = std::filesystem;

namespace qorvix::vision {

namespace {

std::uint32_t beU32(const std::uint8_t* p) {
  return (static_cast<std::uint32_t>(p[0]) << 24) | (static_cast<std::uint32_t>(p[1]) << 16) |
         (static_cast<std::uint32_t>(p[2]) << 8) | static_cast<std::uint32_t>(p[3]);
}
std::uint32_t leU32(const std::uint8_t* p) {
  return (static_cast<std::uint32_t>(p[3]) << 24) | (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[1]) << 8) | static_cast<std::uint32_t>(p[0]);
}

int channelsForColorType(int colorType) {
  switch (colorType) {
    case 0: return 1;  // grayscale
    case 2: return 3;  // truecolour
    case 3: return 1;  // palette index
    case 4: return 2;  // grayscale + alpha
    case 6: return 4;  // truecolour + alpha
    default: return 0;
  }
}

// PNG's Paeth predictor (RFC 2083 §6.6). Reproduced exactly rather than "simplified": the
// tie-breaking order (a, then b, then c) is normative, and getting it wrong corrupts only some
// images, which is the worst kind of bug to chase.
std::uint8_t paeth(int a, int b, int c) {
  const int p = a + b - c;
  const int pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
  if (pa <= pb && pa <= pc) return static_cast<std::uint8_t>(a);
  if (pb <= pc) return static_cast<std::uint8_t>(b);
  return static_cast<std::uint8_t>(c);
}

}  // namespace

bool decodePng(const std::uint8_t* data, std::size_t size, Image& out, std::string& error) {
  error.clear();
  static const std::uint8_t kSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  if (size < 8 || std::memcmp(data, kSig, 8) != 0) {
    error = "not a PNG file";
    return false;
  }

  int width = 0, height = 0, bitDepth = 0, colorType = 0, interlace = 0;
  bool haveHeader = false;
  std::vector<std::uint8_t> idat;
  std::vector<std::uint8_t> palette;      // RGB triples
  std::vector<std::uint8_t> paletteAlpha; // tRNS for palette images

  std::size_t pos = 8;
  while (pos + 8 <= size) {
    const std::uint32_t len = beU32(data + pos);
    if (len > size || pos + 12 + len > size) {
      error = "PNG chunk overruns the file";
      return false;
    }
    const char* type = reinterpret_cast<const char*>(data + pos + 4);
    const std::uint8_t* body = data + pos + 8;

    if (std::memcmp(type, "IHDR", 4) == 0) {
      if (len != 13) {
        error = "malformed IHDR";
        return false;
      }
      width = static_cast<int>(beU32(body));
      height = static_cast<int>(beU32(body + 4));
      bitDepth = body[8];
      colorType = body[9];
      interlace = body[12];
      haveHeader = true;
      if (width <= 0 || height <= 0) {
        error = "PNG has non-positive dimensions";
        return false;
      }
      // A guard, not a policy: 64 megapixels is far past anything a vision encoder will see, and
      // without it a crafted IHDR turns into a multi-gigabyte allocation.
      if (static_cast<std::int64_t>(width) * height > 64ll * 1024 * 1024) {
        error = "PNG is implausibly large";
        return false;
      }
      if (interlace != 0) {
        // Adam7 needs seven separate passes with their own filtering. Refused rather than
        // half-implemented, since a partial pass would produce a scrambled but decodable image.
        error = "interlaced (Adam7) PNGs are not supported";
        return false;
      }
      if (channelsForColorType(colorType) == 0) {
        error = "unknown PNG colour type " + std::to_string(colorType);
        return false;
      }
      if (bitDepth != 8 && bitDepth != 16 && !(colorType == 3 && (bitDepth == 1 || bitDepth == 2 ||
                                                                  bitDepth == 4))) {
        error = "unsupported PNG bit depth " + std::to_string(bitDepth);
        return false;
      }
    } else if (std::memcmp(type, "PLTE", 4) == 0) {
      palette.assign(body, body + len);
    } else if (std::memcmp(type, "tRNS", 4) == 0 && colorType == 3) {
      paletteAlpha.assign(body, body + len);
    } else if (std::memcmp(type, "IDAT", 4) == 0) {
      idat.insert(idat.end(), body, body + len);
    } else if (std::memcmp(type, "IEND", 4) == 0) {
      break;
    }
    pos += 12 + len;  // length + type + body + CRC
  }

  if (!haveHeader) {
    error = "PNG has no IHDR";
    return false;
  }
  if (idat.empty()) {
    error = "PNG has no image data";
    return false;
  }
  if (colorType == 3 && palette.empty()) {
    error = "palette PNG has no PLTE chunk";
    return false;
  }

  const int channels = channelsForColorType(colorType);
  const int bitsPerPixel = channels * bitDepth;
  const std::size_t rowBytes = (static_cast<std::size_t>(width) * bitsPerPixel + 7) / 8;
  const std::size_t expect = (rowBytes + 1) * static_cast<std::size_t>(height);  // +1 filter byte

  std::vector<std::uint8_t> raw;
  if (!inflateZlib(idat.data(), idat.size(), raw, error, expect)) {
    error = "PNG image data: " + error;
    return false;
  }
  if (raw.size() < expect) {
    error = "PNG image data is truncated (" + std::to_string(raw.size()) + " of " +
            std::to_string(expect) + " bytes)";
    return false;
  }

  // Un-filter in place. Filters operate on BYTES at a fixed stride (the pixel size, minimum 1),
  // not on colour channels — which is why bpp is computed from bits and rounded up.
  const std::size_t bpp = std::max<std::size_t>(1, static_cast<std::size_t>(bitsPerPixel) / 8);
  std::vector<std::uint8_t> lines(rowBytes * static_cast<std::size_t>(height));
  const std::uint8_t* src = raw.data();
  for (int y = 0; y < height; ++y) {
    const std::uint8_t filter = *src++;
    std::uint8_t* cur = lines.data() + static_cast<std::size_t>(y) * rowBytes;
    const std::uint8_t* prev = y > 0 ? lines.data() + static_cast<std::size_t>(y - 1) * rowBytes
                                     : nullptr;
    for (std::size_t i = 0; i < rowBytes; ++i) {
      const int x = src[i];
      const int a = i >= bpp ? cur[i - bpp] : 0;
      const int b = prev ? prev[i] : 0;
      const int c = (prev && i >= bpp) ? prev[i - bpp] : 0;
      switch (filter) {
        case 0: cur[i] = static_cast<std::uint8_t>(x); break;
        case 1: cur[i] = static_cast<std::uint8_t>(x + a); break;
        case 2: cur[i] = static_cast<std::uint8_t>(x + b); break;
        case 3: cur[i] = static_cast<std::uint8_t>(x + ((a + b) >> 1)); break;
        case 4: cur[i] = static_cast<std::uint8_t>(x + paeth(a, b, c)); break;
        default:
          error = "unknown PNG filter type " + std::to_string(filter);
          return false;
      }
    }
    src += rowBytes;
  }

  // Expand to 8-bit RGB, compositing any alpha over white (PIL's RGBA->RGB default).
  out.width = width;
  out.height = height;
  out.rgb.assign(static_cast<std::size_t>(width) * height * 3, 0);

  auto sample16 = [&](const std::uint8_t* p) { return p[0]; };  // 16-bit: keep the high byte
  const int step = bitDepth == 16 ? 2 : 1;

  for (int y = 0; y < height; ++y) {
    const std::uint8_t* row = lines.data() + static_cast<std::size_t>(y) * rowBytes;
    for (int x = 0; x < width; ++x) {
      std::uint8_t r = 0, g = 0, b = 0, a = 255;
      if (colorType == 3) {
        int idx = 0;
        if (bitDepth == 8) {
          idx = row[x];
        } else {
          const int perByte = 8 / bitDepth;
          const int shift = 8 - bitDepth * (x % perByte + 1);
          idx = (row[x / perByte] >> shift) & ((1 << bitDepth) - 1);
        }
        if (static_cast<std::size_t>(idx) * 3 + 2 >= palette.size()) {
          error = "PNG palette index out of range";
          return false;
        }
        r = palette[static_cast<std::size_t>(idx) * 3];
        g = palette[static_cast<std::size_t>(idx) * 3 + 1];
        b = palette[static_cast<std::size_t>(idx) * 3 + 2];
        if (static_cast<std::size_t>(idx) < paletteAlpha.size()) a = paletteAlpha[idx];
      } else {
        const std::uint8_t* p = row + static_cast<std::size_t>(x) * channels * step;
        if (colorType == 0) {
          r = g = b = sample16(p);
        } else if (colorType == 4) {
          r = g = b = sample16(p);
          a = sample16(p + step);
        } else if (colorType == 2) {
          r = sample16(p);
          g = sample16(p + step);
          b = sample16(p + 2 * step);
        } else {  // 6
          r = sample16(p);
          g = sample16(p + step);
          b = sample16(p + 2 * step);
          a = sample16(p + 3 * step);
        }
      }
      std::uint8_t* dst = out.rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
      if (a == 255) {
        dst[0] = r;
        dst[1] = g;
        dst[2] = b;
      } else {
        const int inv = 255 - a;
        dst[0] = static_cast<std::uint8_t>((r * a + 255 * inv + 127) / 255);
        dst[1] = static_cast<std::uint8_t>((g * a + 255 * inv + 127) / 255);
        dst[2] = static_cast<std::uint8_t>((b * a + 255 * inv + 127) / 255);
      }
    }
  }
  return true;
}

namespace {

bool decodeBmp(const std::uint8_t* data, std::size_t size, Image& out, std::string& error) {
  if (size < 54 || data[0] != 'B' || data[1] != 'M') {
    error = "not a BMP file";
    return false;
  }
  const std::uint32_t offset = leU32(data + 10);
  const std::uint32_t headerSize = leU32(data + 14);
  if (headerSize < 40) {
    error = "unsupported BMP header";
    return false;
  }
  const int width = static_cast<int>(leU32(data + 18));
  const std::int32_t rawHeight = static_cast<std::int32_t>(leU32(data + 22));
  const int height = std::abs(rawHeight);
  const int bpp = data[28] | (data[29] << 8);
  const std::uint32_t compression = leU32(data + 30);
  if (compression != 0) {
    error = "compressed BMPs are not supported";
    return false;
  }
  if (bpp != 24 && bpp != 32) {
    error = "only 24- and 32-bit BMPs are supported";
    return false;
  }
  if (width <= 0 || height <= 0) {
    error = "BMP has non-positive dimensions";
    return false;
  }
  const std::size_t stride = ((static_cast<std::size_t>(width) * bpp / 8) + 3) & ~std::size_t{3};
  if (offset + stride * height > size) {
    error = "BMP pixel data is truncated";
    return false;
  }

  out.width = width;
  out.height = height;
  out.rgb.assign(static_cast<std::size_t>(width) * height * 3, 0);
  // A positive height means bottom-up row order, which is the BMP default.
  const bool bottomUp = rawHeight > 0;
  for (int y = 0; y < height; ++y) {
    const std::size_t srcRow = bottomUp ? static_cast<std::size_t>(height - 1 - y) : y;
    const std::uint8_t* row = data + offset + srcRow * stride;
    for (int x = 0; x < width; ++x) {
      const std::uint8_t* p = row + static_cast<std::size_t>(x) * (bpp / 8);
      std::uint8_t* dst = out.rgb.data() + (static_cast<std::size_t>(y) * width + x) * 3;
      dst[0] = p[2];  // BMP stores BGR
      dst[1] = p[1];
      dst[2] = p[0];
    }
  }
  return true;
}

bool decodePpm(const std::uint8_t* data, std::size_t size, Image& out, std::string& error) {
  if (size < 2 || data[0] != 'P' || data[1] != '6') {
    error = "not a binary PPM file";
    return false;
  }
  std::size_t pos = 2;
  int fields[3] = {0, 0, 0};
  for (int f = 0; f < 3;) {
    while (pos < size && (std::isspace(data[pos]) || data[pos] == '#')) {
      if (data[pos] == '#') {
        while (pos < size && data[pos] != '\n') ++pos;
      } else {
        ++pos;
      }
    }
    if (pos >= size || !std::isdigit(data[pos])) {
      error = "malformed PPM header";
      return false;
    }
    int v = 0;
    while (pos < size && std::isdigit(data[pos])) v = v * 10 + (data[pos++] - '0');
    fields[f++] = v;
  }
  ++pos;  // single whitespace byte after maxval
  const int width = fields[0], height = fields[1], maxval = fields[2];
  if (width <= 0 || height <= 0 || maxval != 255) {
    error = "unsupported PPM (only 8-bit P6 is handled)";
    return false;
  }
  const std::size_t need = static_cast<std::size_t>(width) * height * 3;
  if (pos + need > size) {
    error = "PPM pixel data is truncated";
    return false;
  }
  out.width = width;
  out.height = height;
  out.rgb.assign(data + pos, data + pos + need);
  return true;
}

}  // namespace

bool decodeImage(const std::uint8_t* data, std::size_t size, Image& out, std::string& error) {
  error.clear();
  if (size >= 8 && data[0] == 0x89 && data[1] == 'P') return decodePng(data, size, out, error);
  if (size >= 2 && data[0] == 'B' && data[1] == 'M') return decodeBmp(data, size, out, error);
  if (size >= 2 && data[0] == 'P' && data[1] == '6') return decodePpm(data, size, out, error);
  if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF) {
    // Baseline JPEG needs Huffman decoding, dequantization, an IDCT and YCbCr conversion — a
    // separate body of work with its own correctness surface. Named explicitly so the message
    // says what to do rather than "unknown format".
    error = "JPEG is not supported yet — convert to PNG first";
    return false;
  }
  error = "unrecognized image format";
  return false;
}

bool loadImage(const fs::path& path, Image& out, std::string& error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open '" + path.string() + "'";
    return false;
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
  if (bytes.empty()) {
    error = "'" + path.string() + "' is empty";
    return false;
  }
  if (!decodeImage(bytes.data(), bytes.size(), out, error)) {
    error = path.string() + ": " + error;
    return false;
  }
  return true;
}

}  // namespace qorvix::vision
