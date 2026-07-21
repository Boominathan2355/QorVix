#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "qorvix/gguf/gguf_error.hpp"
#include "qorvix/gguf/gguf_types.hpp"
#include "qorvix/gguf/mapped_file.hpp"

namespace qorvix::gguf {

struct GgufHeader {
  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint64_t tensorCount = 0;
  std::uint64_t metadataCount = 0;
};

// One entry from the tensor table. `offset` is relative to the start of the tensor-data section
// (dataOffset() in GgufFile); `nBytes` is the computed on-disk size from the type traits.
struct GgufTensor {
  std::string name;
  std::vector<std::uint64_t> dimensions;  // ggml order: dimensions[0] is the row length
  std::uint32_t typeRaw = 0;
  std::uint64_t offset = 0;
  std::uint64_t nElements = 0;
  std::uint64_t nBytes = 0;

  std::string typeName() const { return ggmlTypeName(typeRaw); }
};

// Convenience view of RoPE metadata for the detected architecture. Fields are nullopt when the
// corresponding key is absent (many models rely on architecture defaults).
struct RopeParams {
  std::optional<std::uint64_t> dimensionCount;   // <arch>.rope.dimension_count
  std::optional<double> freqBase;                // <arch>.rope.freq_base
  std::optional<std::string> scalingType;        // <arch>.rope.scaling.type
  std::optional<double> scalingFactor;           // <arch>.rope.scaling.factor
  std::optional<std::uint64_t> origContextLength; // <arch>.rope.scaling.original_context_length
};

// Parsed GGUF header, metadata, and tensor table. Tensor *data* is not read — only located.
// Construct via parse()/open(); both throw GgufParseError on malformed input.
class GgufFile {
 public:
  // Parses from an in-memory buffer (does not take ownership).
  static GgufFile parse(std::span<const std::byte> bytes);

  // Memory-maps `path` and parses it; the map is retained so tensor data stays reachable.
  static GgufFile open(const std::filesystem::path& path);

  const GgufHeader& header() const noexcept { return header_; }
  std::uint32_t alignment() const noexcept { return alignment_; }

  // Absolute file offset where the tensor-data section begins (after the tensor table, padded
  // up to alignment()). Tensor bytes live at dataOffset() + tensor.offset.
  std::uint64_t dataOffset() const noexcept { return dataOffset_; }

  const std::vector<GgufTensor>& tensors() const noexcept { return tensors_; }
  const GgufTensor* tensor(const std::string& name) const;

  // Raw metadata access, preserving on-disk order.
  const std::vector<std::pair<std::string, GgufValue>>& metadata() const noexcept {
    return metadata_;
  }
  const GgufValue* find(const std::string& key) const;

  // Typed metadata helpers (return nullopt when absent or the wrong type).
  std::optional<std::string> getString(const std::string& key) const;
  std::optional<std::uint64_t> getU64(const std::string& key) const;
  std::optional<double> getF64(const std::string& key) const;
  std::optional<bool> getBool(const std::string& key) const;

  // Detected architecture from general.architecture (empty if unspecified).
  const std::string& architecture() const noexcept { return architecture_; }
  std::optional<std::string> name() const { return getString("general.name"); }
  std::optional<std::uint64_t> fileType() const { return getU64("general.file_type"); }

  // Reads an "<architecture>.<suffix>" metadata key, e.g. ropeKey("attention.head_count").
  std::optional<std::uint64_t> archU64(const std::string& suffix) const;
  RopeParams rope() const;

  // The mmap backing open(); empty for parse(). Exposes tensor bytes for later phases.
  const MappedFile& mapping() const noexcept { return mapping_; }

 private:
  void parseInternal(std::span<const std::byte> bytes);

  GgufHeader header_;
  std::uint32_t alignment_ = kDefaultAlignment;
  std::uint64_t dataOffset_ = 0;
  std::string architecture_;

  std::vector<std::pair<std::string, GgufValue>> metadata_;
  std::unordered_map<std::string, std::size_t> metadataIndex_;
  std::vector<GgufTensor> tensors_;
  std::unordered_map<std::string, std::size_t> tensorIndex_;

  MappedFile mapping_;
};

}  // namespace qorvix::gguf
