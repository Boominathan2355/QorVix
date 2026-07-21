#pragma once

// Test-only helper that serializes a GGUF v2/v3 byte buffer in memory, so the parser can be
// exercised without shipping large binary fixtures. Little-endian, mirroring the wire format.
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "qorvix/gguf/gguf_types.hpp"

namespace qorvix::gguf::test {

class GgufBuilder {
 public:
  explicit GgufBuilder(std::uint32_t version = 3) : version_(version) {}

  // ---- metadata ----
  GgufBuilder& u32(const std::string& key, std::uint32_t v) {
    kv(key, MetaType::UInt32);
    putU32(v);
    ++metaCount_;
    return *this;
  }
  GgufBuilder& u64(const std::string& key, std::uint64_t v) {
    kv(key, MetaType::UInt64);
    putU64(v);
    ++metaCount_;
    return *this;
  }
  GgufBuilder& f32(const std::string& key, float v) {
    kv(key, MetaType::Float32);
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    putU32(bits);
    ++metaCount_;
    return *this;
  }
  GgufBuilder& boolean(const std::string& key, bool v) {
    kv(key, MetaType::Bool);
    meta_.push_back(v ? std::byte{1} : std::byte{0});
    ++metaCount_;
    return *this;
  }
  GgufBuilder& str(const std::string& key, const std::string& v) {
    kv(key, MetaType::String);
    putString(meta_, v);
    ++metaCount_;
    return *this;
  }
  GgufBuilder& stringArray(const std::string& key, const std::vector<std::string>& values) {
    kv(key, MetaType::Array);
    putU32(static_cast<std::uint32_t>(MetaType::String));
    putLen(meta_, values.size());
    for (const auto& s : values) putString(meta_, s);
    ++metaCount_;
    return *this;
  }
  GgufBuilder& i32Array(const std::string& key, const std::vector<std::int32_t>& values) {
    kv(key, MetaType::Array);
    putU32(static_cast<std::uint32_t>(MetaType::Int32));
    putLen(meta_, values.size());
    for (std::int32_t v : values) putU32(static_cast<std::uint32_t>(v));
    ++metaCount_;
    return *this;
  }

  // ---- tensors ----
  // Registers a tensor with the given dims/type/offset (offset relative to the data section).
  GgufBuilder& tensor(const std::string& name, const std::vector<std::uint64_t>& dims,
                      std::uint32_t type, std::uint64_t offset) {
    putString(tensors_, name);
    putU32To(tensors_, static_cast<std::uint32_t>(dims.size()));
    for (std::uint64_t d : dims) putLenTo(tensors_, d);
    putU32To(tensors_, type);
    putU64To(tensors_, offset);
    ++tensorCount_;
    return *this;
  }

  // Serializes everything, appending `dataBytes` of zero-filled tensor data after the aligned
  // data offset (so extent validation has something to check).
  std::vector<std::byte> build(std::uint32_t alignment = 32, std::uint64_t dataBytes = 0) {
    std::vector<std::byte> out;
    putU32To(out, kMagic);
    putU32To(out, version_);
    putLenTo(out, tensorCount_);
    putLenTo(out, metaCount_);
    out.insert(out.end(), meta_.begin(), meta_.end());
    out.insert(out.end(), tensors_.begin(), tensors_.end());
    // pad to alignment
    while (alignment && out.size() % alignment != 0) out.push_back(std::byte{0});
    out.insert(out.end(), static_cast<std::size_t>(dataBytes), std::byte{0});
    return out;
  }

 private:
  bool wide() const { return version_ >= 2; }

  void kv(const std::string& key, MetaType type) {
    putString(meta_, key);
    putU32(static_cast<std::uint32_t>(type));
  }

  void putU32(std::uint32_t v) { putU32To(meta_, v); }
  void putU64(std::uint64_t v) { putU64To(meta_, v); }

  void putLen(std::vector<std::byte>& buf, std::uint64_t v) {
    if (wide()) putU64To(buf, v); else putU32To(buf, static_cast<std::uint32_t>(v));
  }
  void putLenTo(std::vector<std::byte>& buf, std::uint64_t v) { putLen(buf, v); }

  void putString(std::vector<std::byte>& buf, const std::string& s) {
    putLen(buf, s.size());
    for (char c : s) buf.push_back(static_cast<std::byte>(static_cast<unsigned char>(c)));
  }

  static void putU32To(std::vector<std::byte>& buf, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) buf.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
  }
  static void putU64To(std::vector<std::byte>& buf, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<std::byte>((v >> (8 * i)) & 0xFF));
  }

  std::uint32_t version_;
  std::uint64_t metaCount_ = 0;
  std::uint64_t tensorCount_ = 0;
  std::vector<std::byte> meta_;
  std::vector<std::byte> tensors_;
};

}  // namespace qorvix::gguf::test
