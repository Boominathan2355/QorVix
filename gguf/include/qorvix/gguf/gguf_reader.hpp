#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>

#include "qorvix/gguf/gguf_error.hpp"

namespace qorvix::gguf {

// Forward-only cursor over an in-memory buffer. All reads are little-endian (the GGUF wire
// format) and bounds-checked; integers/floats are assembled byte-by-byte so results are
// independent of host endianness. Length-prefixed strings are validated against the remaining
// buffer before any allocation, so a corrupt length can't trigger a huge alloc.
class ByteReader {
 public:
  explicit ByteReader(std::span<const std::byte> data) : data_(data) {}

  std::size_t position() const noexcept { return pos_; }
  std::size_t remaining() const noexcept { return data_.size() - pos_; }
  std::span<const std::byte> data() const noexcept { return data_; }

  void seek(std::size_t pos) {
    if (pos > data_.size()) throw GgufParseError("seek past end of buffer");
    pos_ = pos;
  }

  void require(std::size_t n) const {
    if (n > remaining()) {
      throw GgufParseError("unexpected end of buffer: need " + std::to_string(n) + " byte(s), " +
                           std::to_string(remaining()) + " remaining at offset " +
                           std::to_string(pos_));
    }
  }

  std::uint8_t readU8() { return readUInt<std::uint8_t>(); }
  std::int8_t readI8() { return static_cast<std::int8_t>(readUInt<std::uint8_t>()); }
  std::uint16_t readU16() { return readUInt<std::uint16_t>(); }
  std::int16_t readI16() { return static_cast<std::int16_t>(readUInt<std::uint16_t>()); }
  std::uint32_t readU32() { return readUInt<std::uint32_t>(); }
  std::int32_t readI32() { return static_cast<std::int32_t>(readUInt<std::uint32_t>()); }
  std::uint64_t readU64() { return readUInt<std::uint64_t>(); }
  std::int64_t readI64() { return static_cast<std::int64_t>(readUInt<std::uint64_t>()); }
  bool readBool() { return readU8() != 0; }

  float readF32() { return std::bit_cast<float>(readUInt<std::uint32_t>()); }
  double readF64() { return std::bit_cast<double>(readUInt<std::uint64_t>()); }

  // GGUF string: a length (width depends on version — 32-bit in v1, 64-bit in v2/v3) followed
  // by that many UTF-8 bytes (not null-terminated).
  std::string readString(bool lengthIs64Bit) {
    const std::uint64_t len = lengthIs64Bit ? readU64() : readU32();
    require(len);
    const auto* base = reinterpret_cast<const char*>(data_.data() + pos_);
    std::string s(base, static_cast<std::size_t>(len));
    pos_ += static_cast<std::size_t>(len);
    return s;
  }

 private:
  template <typename T>
  T readUInt() {
    require(sizeof(T));
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      value |= static_cast<T>(static_cast<std::uint8_t>(data_[pos_ + i])) << (8 * i);
    }
    pos_ += sizeof(T);
    return value;
  }

  std::span<const std::byte> data_;
  std::size_t pos_ = 0;
};

}  // namespace qorvix::gguf
