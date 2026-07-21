#include "qorvix/gguf/gguf_file.hpp"

#include <limits>
#include <utility>

#include "qorvix/gguf/gguf_reader.hpp"

namespace qorvix::gguf {

namespace {

// Smallest number of bytes an array element of `type` can occupy — used to reject absurd array
// lengths before allocating. For strings it's just the length prefix (an empty string).
std::uint64_t minElementBytes(MetaType type, bool len64) {
  switch (type) {
    case MetaType::UInt8:
    case MetaType::Int8:
    case MetaType::Bool: return 1;
    case MetaType::UInt16:
    case MetaType::Int16: return 2;
    case MetaType::UInt32:
    case MetaType::Int32:
    case MetaType::Float32: return 4;
    case MetaType::UInt64:
    case MetaType::Int64:
    case MetaType::Float64: return 8;
    case MetaType::String: return len64 ? 8 : 4;
    case MetaType::Array: return 0;  // nested arrays are rejected before this is consulted
  }
  return 0;
}

GgufValue readScalar(ByteReader& r, MetaType type, bool len64) {
  switch (type) {
    case MetaType::UInt8: return GgufValue(type, r.readU8());
    case MetaType::Int8: return GgufValue(type, r.readI8());
    case MetaType::UInt16: return GgufValue(type, r.readU16());
    case MetaType::Int16: return GgufValue(type, r.readI16());
    case MetaType::UInt32: return GgufValue(type, r.readU32());
    case MetaType::Int32: return GgufValue(type, r.readI32());
    case MetaType::Float32: return GgufValue(type, r.readF32());
    case MetaType::Bool: return GgufValue(type, r.readBool());
    case MetaType::String: return GgufValue(type, r.readString(len64));
    case MetaType::UInt64: return GgufValue(type, r.readU64());
    case MetaType::Int64: return GgufValue(type, r.readI64());
    case MetaType::Float64: return GgufValue(type, r.readF64());
    case MetaType::Array:
      throw GgufParseError("nested arrays are not permitted in GGUF metadata");
  }
  throw GgufParseError("unknown metadata value type " +
                       std::to_string(static_cast<std::uint32_t>(type)));
}

GgufValue readValue(ByteReader& r, MetaType type, bool len64) {
  if (type != MetaType::Array) return readScalar(r, type, len64);

  const auto elementType = static_cast<MetaType>(r.readU32());
  if (elementType == MetaType::Array) {
    throw GgufParseError("nested arrays are not permitted in GGUF metadata");
  }
  const std::uint64_t length = len64 ? r.readU64() : r.readU32();

  // Fail fast (without allocating) if the declared length cannot possibly fit in the buffer.
  const std::uint64_t minBytes = minElementBytes(elementType, len64);
  if (minBytes > 0 && length > r.remaining() / minBytes) {
    throw GgufParseError("array length " + std::to_string(length) + " exceeds remaining buffer");
  }

  std::vector<GgufValue> elements;
  elements.reserve(static_cast<std::size_t>(length));
  for (std::uint64_t i = 0; i < length; ++i) {
    elements.push_back(readScalar(r, elementType, len64));
  }
  return GgufValue::makeArray(elementType, std::move(elements));
}

std::uint64_t productChecked(const std::vector<std::uint64_t>& dims) {
  std::uint64_t acc = 1;
  for (std::uint64_t d : dims) {
    if (d == 0) return 0;
    if (acc > std::numeric_limits<std::uint64_t>::max() / d) {
      throw GgufParseError("tensor element count overflows uint64");
    }
    acc *= d;
  }
  return acc;
}

std::uint64_t alignUp(std::uint64_t value, std::uint32_t alignment) {
  if (alignment == 0) return value;
  return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

void GgufFile::parseInternal(std::span<const std::byte> bytes) {
  ByteReader r(bytes);

  header_.magic = r.readU32();
  if (header_.magic != kMagic) {
    throw GgufParseError("not a GGUF file (bad magic)");
  }
  header_.version = r.readU32();
  if (header_.version < 1 || header_.version > 3) {
    throw GgufParseError("unsupported GGUF version " + std::to_string(header_.version));
  }
  const bool wide = header_.version >= 2;  // v2/v3 use 64-bit counts and string lengths

  header_.tensorCount = wide ? r.readU64() : r.readU32();
  header_.metadataCount = wide ? r.readU64() : r.readU32();

  // ---- metadata ----
  metadata_.reserve(static_cast<std::size_t>(header_.metadataCount));
  for (std::uint64_t i = 0; i < header_.metadataCount; ++i) {
    std::string key = r.readString(wide);
    const auto valueType = static_cast<MetaType>(r.readU32());
    GgufValue value = readValue(r, valueType, wide);
    metadataIndex_[key] = metadata_.size();
    metadata_.emplace_back(std::move(key), std::move(value));
  }

  if (auto align = getU64("general.alignment")) {
    alignment_ = static_cast<std::uint32_t>(*align);
    if (alignment_ == 0) alignment_ = kDefaultAlignment;
  }
  if (auto arch = getString("general.architecture")) architecture_ = *arch;

  // ---- tensor table ----
  tensors_.reserve(static_cast<std::size_t>(header_.tensorCount));
  for (std::uint64_t i = 0; i < header_.tensorCount; ++i) {
    GgufTensor t;
    t.name = r.readString(wide);

    const std::uint32_t nDims = r.readU32();
    if (nDims > 8) throw GgufParseError("tensor '" + t.name + "' has implausible rank " +
                                        std::to_string(nDims));
    t.dimensions.reserve(nDims);
    for (std::uint32_t d = 0; d < nDims; ++d) {
      t.dimensions.push_back(wide ? r.readU64() : r.readU32());
    }

    t.typeRaw = r.readU32();
    t.offset = r.readU64();
    t.nElements = productChecked(t.dimensions);

    if (const auto* traits = ggmlTypeTraits(t.typeRaw)) {
      if (traits->blockSize > 1 && t.nElements % traits->blockSize != 0) {
        throw GgufParseError("tensor '" + t.name + "' element count " +
                             std::to_string(t.nElements) + " is not a multiple of block size " +
                             std::to_string(traits->blockSize) + " for " + traits->name);
      }
      t.nBytes = t.nElements / traits->blockSize * traits->typeSize;
    } else {
      t.nBytes = 0;  // unknown type: recorded but size cannot be computed
    }

    if (t.offset % alignment_ != 0) {
      throw GgufParseError("tensor '" + t.name + "' offset " + std::to_string(t.offset) +
                           " is not aligned to " + std::to_string(alignment_));
    }

    tensorIndex_[t.name] = tensors_.size();
    tensors_.push_back(std::move(t));
  }

  dataOffset_ = alignUp(r.position(), alignment_);

  // Validate tensor extents only when the buffer actually carries the data section (a full file
  // or a fixture that includes data). Header-only buffers legitimately stop at dataOffset_.
  if (bytes.size() > dataOffset_) {
    const std::uint64_t dataBytes = bytes.size() - dataOffset_;
    for (const auto& t : tensors_) {
      if (t.nBytes == 0) continue;  // unknown type — nothing to check
      if (t.offset > dataBytes || t.nBytes > dataBytes - t.offset) {
        throw GgufParseError("tensor '" + t.name + "' data extent exceeds the tensor-data section");
      }
    }
  }
}

GgufFile GgufFile::parse(std::span<const std::byte> bytes) {
  GgufFile file;
  file.parseInternal(bytes);
  return file;
}

GgufFile GgufFile::open(const std::filesystem::path& path) {
  GgufFile file;
  if (!file.mapping_.open(path)) {
    throw GgufParseError("cannot open '" + path.string() + "': " + file.mapping_.error());
  }
  file.parseInternal(file.mapping_.bytes());
  return file;
}

const GgufValue* GgufFile::find(const std::string& key) const {
  auto it = metadataIndex_.find(key);
  return it == metadataIndex_.end() ? nullptr : &metadata_[it->second].second;
}

const GgufTensor* GgufFile::tensor(const std::string& name) const {
  auto it = tensorIndex_.find(name);
  return it == tensorIndex_.end() ? nullptr : &tensors_[it->second];
}

std::optional<std::string> GgufFile::getString(const std::string& key) const {
  if (const GgufValue* v = find(key)) {
    if (const std::string* s = v->asString()) return *s;
  }
  return std::nullopt;
}

std::optional<std::uint64_t> GgufFile::getU64(const std::string& key) const {
  if (const GgufValue* v = find(key)) return v->asU64();
  return std::nullopt;
}

std::optional<double> GgufFile::getF64(const std::string& key) const {
  if (const GgufValue* v = find(key)) return v->asF64();
  return std::nullopt;
}

std::optional<bool> GgufFile::getBool(const std::string& key) const {
  if (const GgufValue* v = find(key)) return v->asBool();
  return std::nullopt;
}

std::optional<std::uint64_t> GgufFile::archU64(const std::string& suffix) const {
  if (architecture_.empty()) return std::nullopt;
  return getU64(architecture_ + "." + suffix);
}

RopeParams GgufFile::rope() const {
  RopeParams p;
  if (architecture_.empty()) return p;
  const std::string a = architecture_ + ".";
  p.dimensionCount = getU64(a + "rope.dimension_count");
  p.freqBase = getF64(a + "rope.freq_base");
  p.scalingType = getString(a + "rope.scaling.type");
  p.scalingFactor = getF64(a + "rope.scaling.factor");
  p.origContextLength = getU64(a + "rope.scaling.original_context_length");
  return p;
}

}  // namespace qorvix::gguf
