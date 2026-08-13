#include "qorvix/vision/inflate.hpp"

#include <algorithm>
#include <array>

namespace qorvix::vision {

namespace {

// ---- bit reader ------------------------------------------------------------------------------
// DEFLATE packs codes LSB-first within each byte, but Huffman codes are stored MSB-first within
// the code itself. Getting that asymmetry wrong is the classic way to produce a decoder that works
// on stored blocks and garbles everything else.
class BitReader {
 public:
  BitReader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}

  bool bits(int n, std::uint32_t& out) {
    while (bitCount_ < n) {
      if (pos_ >= size_) return false;
      bitBuf_ |= static_cast<std::uint32_t>(data_[pos_++]) << bitCount_;
      bitCount_ += 8;
    }
    out = bitBuf_ & ((1u << n) - 1u);
    bitBuf_ >>= n;
    bitCount_ -= n;
    return true;
  }

  void alignToByte() {
    const int drop = bitCount_ & 7;
    bitBuf_ >>= drop;
    bitCount_ -= drop;
  }

  // Reads whole bytes after alignToByte(), used by stored blocks.
  bool readBytes(std::uint8_t* dst, std::size_t n) {
    while (n > 0 && bitCount_ >= 8) {
      *dst++ = static_cast<std::uint8_t>(bitBuf_ & 0xFF);
      bitBuf_ >>= 8;
      bitCount_ -= 8;
      --n;
    }
    if (n > size_ - pos_) return false;
    std::copy(data_ + pos_, data_ + pos_ + n, dst);
    pos_ += n;
    return true;
  }

  std::size_t bytePos() const { return pos_; }
  int bufferedBits() const { return bitCount_; }

 private:
  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
  std::uint32_t bitBuf_ = 0;
  int bitCount_ = 0;
};

// ---- canonical Huffman decoding ---------------------------------------------------------------
// Built from code LENGTHS alone, per RFC 1951 §3.2.2: codes of the same length are consecutive in
// symbol order, and each length starts where the previous length's range ends (shifted). Decoding
// walks bit by bit, which is slower than a lookup table but is the form the spec is written in and
// therefore the form that can be checked against it line by line.
struct Huffman {
  std::array<int, 16> countByLen{};   // how many codes of each length
  std::array<int, 16> firstCode{};    // first canonical code of each length
  std::array<int, 16> firstIndex{};   // index into `symbols` where each length starts
  std::vector<int> symbols;

  bool build(const std::uint8_t* lengths, int n) {
    countByLen.fill(0);
    for (int i = 0; i < n; ++i) {
      if (lengths[i] > 15) return false;
      ++countByLen[lengths[i]];
    }
    countByLen[0] = 0;  // length 0 means "symbol unused"

    // Reject over-subscribed code sets: sum(count[l] * 2^-l) must be <= 1, or the code space
    // overflows and some bit pattern would decode to two different symbols.
    int left = 1;
    for (int len = 1; len <= 15; ++len) {
      left <<= 1;
      left -= countByLen[len];
      if (left < 0) return false;
    }

    int code = 0, index = 0;
    for (int len = 1; len <= 15; ++len) {
      firstCode[len] = code;
      firstIndex[len] = index;
      code += countByLen[len];
      index += countByLen[len];
      code <<= 1;
    }

    symbols.assign(static_cast<std::size_t>(index), 0);
    std::array<int, 16> next = firstIndex;
    for (int i = 0; i < n; ++i) {
      const int len = lengths[i];
      if (len != 0) symbols[static_cast<std::size_t>(next[len]++)] = i;
    }
    return true;
  }

  bool decode(BitReader& br, int& symbol) const {
    int code = 0;
    for (int len = 1; len <= 15; ++len) {
      std::uint32_t bit = 0;
      if (!br.bits(1, bit)) return false;
      code = (code << 1) | static_cast<int>(bit);  // Huffman codes are MSB-first
      const int count = countByLen[len];
      if (count > 0 && code - firstCode[len] < count) {
        symbol = symbols[static_cast<std::size_t>(firstIndex[len] + (code - firstCode[len]))];
        return true;
      }
    }
    return false;  // ran past 15 bits without a match: malformed
  }
};

// RFC 1951 §3.2.5 length/distance tables.
constexpr std::array<std::uint16_t, 29> kLenBase = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,
                                                    15, 17, 19, 23, 27, 31, 35, 43, 51,  59,
                                                    67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr std::array<std::uint8_t, 29> kLenExtra = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                                    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::array<std::uint16_t, 30> kDistBase = {
    1,    2,    3,    4,    5,    7,     9,     13,    17,  25,   33,   49,   65,    97,    129,
    193,  257,  385,  513,  769,  1025,  1537,  2049,  3073, 4097, 6145, 8193, 12289, 16385, 24577};
constexpr std::array<std::uint8_t, 30> kDistExtra = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5, 5, 6,
                                                     6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

void fixedTables(Huffman& lit, Huffman& dist) {
  // RFC 1951 §3.2.6: the fixed literal/length code is defined by a fixed length distribution.
  std::array<std::uint8_t, 288> litLen{};
  for (int i = 0; i < 144; ++i) litLen[i] = 8;
  for (int i = 144; i < 256; ++i) litLen[i] = 9;
  for (int i = 256; i < 280; ++i) litLen[i] = 7;
  for (int i = 280; i < 288; ++i) litLen[i] = 8;
  lit.build(litLen.data(), 288);

  std::array<std::uint8_t, 30> distLen{};
  distLen.fill(5);
  dist.build(distLen.data(), 30);
}

bool readDynamicTables(BitReader& br, Huffman& lit, Huffman& dist, std::string& error) {
  std::uint32_t hlit = 0, hdist = 0, hclen = 0;
  if (!br.bits(5, hlit) || !br.bits(5, hdist) || !br.bits(4, hclen)) {
    error = "truncated dynamic block header";
    return false;
  }
  const int nLit = static_cast<int>(hlit) + 257;
  const int nDist = static_cast<int>(hdist) + 1;
  const int nCode = static_cast<int>(hclen) + 4;

  // The code-length alphabet is itself Huffman-coded, and its lengths arrive in this permuted
  // order so that the common symbols come first and trailing zeros can be omitted.
  static constexpr std::array<int, 19> kOrder = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                                 11, 4,  12, 3, 13, 2, 14, 1, 15};
  std::array<std::uint8_t, 19> clen{};
  for (int i = 0; i < nCode; ++i) {
    std::uint32_t v = 0;
    if (!br.bits(3, v)) {
      error = "truncated code-length alphabet";
      return false;
    }
    clen[static_cast<std::size_t>(kOrder[i])] = static_cast<std::uint8_t>(v);
  }
  Huffman clHuff;
  if (!clHuff.build(clen.data(), 19)) {
    error = "invalid code-length alphabet";
    return false;
  }

  std::vector<std::uint8_t> lengths(static_cast<std::size_t>(nLit + nDist), 0);
  int i = 0;
  while (i < nLit + nDist) {
    int sym = 0;
    if (!clHuff.decode(br, sym)) {
      error = "truncated code-length sequence";
      return false;
    }
    if (sym < 16) {
      lengths[static_cast<std::size_t>(i++)] = static_cast<std::uint8_t>(sym);
      continue;
    }
    int repeat = 0;
    std::uint8_t value = 0;
    std::uint32_t extra = 0;
    if (sym == 16) {
      if (i == 0) {
        error = "repeat code with no previous length";
        return false;
      }
      if (!br.bits(2, extra)) {
        error = "truncated repeat";
        return false;
      }
      repeat = 3 + static_cast<int>(extra);
      value = lengths[static_cast<std::size_t>(i - 1)];
    } else if (sym == 17) {
      if (!br.bits(3, extra)) {
        error = "truncated repeat";
        return false;
      }
      repeat = 3 + static_cast<int>(extra);
    } else {
      if (!br.bits(7, extra)) {
        error = "truncated repeat";
        return false;
      }
      repeat = 11 + static_cast<int>(extra);
    }
    if (i + repeat > nLit + nDist) {
      error = "code-length repeat overruns the alphabet";
      return false;
    }
    for (int r = 0; r < repeat; ++r) lengths[static_cast<std::size_t>(i++)] = value;
  }

  if (!lit.build(lengths.data(), nLit) || !dist.build(lengths.data() + nLit, nDist)) {
    error = "invalid huffman table";
    return false;
  }
  return true;
}

}  // namespace

std::uint32_t adler32(const std::uint8_t* data, std::size_t size) {
  std::uint32_t a = 1, b = 0;
  for (std::size_t i = 0; i < size; ++i) {
    a = (a + data[i]) % 65521;
    b = (b + a) % 65521;
  }
  return (b << 16) | a;
}

bool inflateRaw(const std::uint8_t* data, std::size_t size, std::vector<std::uint8_t>& out,
                std::string& error, std::size_t sizeHint, std::size_t maxOutput) {
  error.clear();
  out.clear();
  if (sizeHint > 0 && sizeHint <= maxOutput) out.reserve(sizeHint);

  BitReader br(data, size);
  Huffman fixedLit, fixedDist;
  fixedTables(fixedLit, fixedDist);

  for (;;) {
    std::uint32_t bfinal = 0, btype = 0;
    if (!br.bits(1, bfinal) || !br.bits(2, btype)) {
      error = "truncated block header";
      return false;
    }

    if (btype == 0) {
      br.alignToByte();
      std::uint8_t hdr[4];
      if (!br.readBytes(hdr, 4)) {
        error = "truncated stored-block header";
        return false;
      }
      const std::size_t len = static_cast<std::size_t>(hdr[0]) | (static_cast<std::size_t>(hdr[1]) << 8);
      const std::size_t nlen = static_cast<std::size_t>(hdr[2]) | (static_cast<std::size_t>(hdr[3]) << 8);
      if ((len ^ 0xFFFF) != nlen) {
        error = "stored-block length check failed";
        return false;
      }
      if (out.size() + len > maxOutput) {
        error = "output exceeds the size limit";
        return false;
      }
      const std::size_t at = out.size();
      out.resize(at + len);
      if (len > 0 && !br.readBytes(out.data() + at, len)) {
        error = "truncated stored block";
        return false;
      }
    } else if (btype == 1 || btype == 2) {
      Huffman dynLit, dynDist;
      const Huffman* lit = &fixedLit;
      const Huffman* dist = &fixedDist;
      if (btype == 2) {
        if (!readDynamicTables(br, dynLit, dynDist, error)) return false;
        lit = &dynLit;
        dist = &dynDist;
      }

      for (;;) {
        int sym = 0;
        if (!lit->decode(br, sym)) {
          error = "truncated symbol stream";
          return false;
        }
        if (sym < 256) {
          if (out.size() + 1 > maxOutput) {
            error = "output exceeds the size limit";
            return false;
          }
          out.push_back(static_cast<std::uint8_t>(sym));
          continue;
        }
        if (sym == 256) break;  // end of block

        const int lenIdx = sym - 257;
        if (lenIdx >= static_cast<int>(kLenBase.size())) {
          error = "invalid length symbol";
          return false;
        }
        std::uint32_t extra = 0;
        if (kLenExtra[lenIdx] && !br.bits(kLenExtra[lenIdx], extra)) {
          error = "truncated length extra bits";
          return false;
        }
        const std::size_t length = kLenBase[lenIdx] + extra;

        int dsym = 0;
        if (!dist->decode(br, dsym) || dsym >= static_cast<int>(kDistBase.size())) {
          error = "invalid distance symbol";
          return false;
        }
        extra = 0;
        if (kDistExtra[dsym] && !br.bits(kDistExtra[dsym], extra)) {
          error = "truncated distance extra bits";
          return false;
        }
        const std::size_t distance = kDistBase[dsym] + extra;
        if (distance > out.size()) {
          error = "back-reference before the start of the stream";
          return false;
        }
        if (out.size() + length > maxOutput) {
          error = "output exceeds the size limit";
          return false;
        }

        // Byte-at-a-time on purpose: overlapping copies are legal and load-bearing (distance 1
        // with length 258 is how a run of one byte is encoded), so a memcpy would be wrong.
        std::size_t src = out.size() - distance;
        for (std::size_t i = 0; i < length; ++i) out.push_back(out[src + i]);
      }
    } else {
      error = "reserved block type 3";
      return false;
    }

    if (bfinal) break;
  }
  return true;
}

bool inflateZlib(const std::uint8_t* data, std::size_t size, std::vector<std::uint8_t>& out,
                 std::string& error, std::size_t sizeHint, std::size_t maxOutput) {
  error.clear();
  if (size < 6) {  // 2 header + at least 1 payload + 4 adler
    error = "zlib stream is too short";
    return false;
  }
  const std::uint8_t cmf = data[0], flg = data[1];
  if ((cmf & 0x0F) != 8) {
    error = "zlib compression method is not deflate";
    return false;
  }
  if ((static_cast<int>(cmf) * 256 + flg) % 31 != 0) {
    error = "zlib header check failed";
    return false;
  }
  if (flg & 0x20) {
    // FDICT: a preset dictionary the stream does not carry. PNG never sets it.
    error = "zlib preset dictionaries are not supported";
    return false;
  }

  if (!inflateRaw(data + 2, size - 6, out, error, sizeHint, maxOutput)) return false;

  const std::uint8_t* tail = data + size - 4;
  const std::uint32_t want = (static_cast<std::uint32_t>(tail[0]) << 24) |
                             (static_cast<std::uint32_t>(tail[1]) << 16) |
                             (static_cast<std::uint32_t>(tail[2]) << 8) |
                             static_cast<std::uint32_t>(tail[3]);
  const std::uint32_t got = adler32(out.data(), out.size());
  if (got != want) {
    // Verified rather than skipped: silently accepting corrupt pixels turns into a plausible
    // wrong embedding, which is far harder to notice than a rejected file.
    error = "zlib checksum mismatch (stream is corrupt)";
    return false;
  }
  return true;
}

}  // namespace qorvix::vision
