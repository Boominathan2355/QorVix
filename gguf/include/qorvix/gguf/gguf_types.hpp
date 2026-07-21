#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace qorvix::gguf {

inline constexpr std::uint32_t kMagic = 0x46554747u;  // "GGUF" little-endian
inline constexpr std::uint32_t kDefaultAlignment = 32u;

// ---- ggml tensor element types -------------------------------------------------------------
// Numeric values are the on-disk ggml_type enum and must not change. Values 4 and 5 are the
// retired Q4_2/Q4_3 formats and are intentionally absent.
enum class GgmlType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
  IQ2_XXS = 16,
  IQ2_XS = 17,
  IQ3_XXS = 18,
  IQ1_S = 19,
  IQ4_NL = 20,
  IQ3_S = 21,
  IQ2_S = 22,
  IQ4_XS = 23,
  I8 = 24,
  I16 = 25,
  I32 = 26,
  I64 = 27,
  F64 = 28,
  IQ1_M = 29,
  BF16 = 30,
};

// Layout traits for a ggml type. A tensor is stored as blocks of `blockSize` elements, each
// block occupying `typeSize` bytes, so byte_size = n_elements / blockSize * typeSize.
struct GgmlTypeTraits {
  const char* name;
  std::uint32_t blockSize;
  std::uint32_t typeSize;
  bool quantized;
  bool specSupported;  // listed in SPEC.md's supported-quantizations set
};

// Returns traits for a raw on-disk type value, or nullptr if the value is unknown to this build
// (forward-compatible: unknown types are surfaced, not fatal).
const GgmlTypeTraits* ggmlTypeTraits(std::uint32_t type);

// Human-readable name; "UNKNOWN(n)" for unrecognized values.
std::string ggmlTypeName(std::uint32_t type);

// ---- metadata value model ------------------------------------------------------------------
// On-disk metadata value type tags (gguf_metadata_value_type), values 0..12 fixed by the format.
enum class MetaType : std::uint32_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

const char* metaTypeName(MetaType type);
bool isScalarMetaType(MetaType type);  // anything but Array

// A single metadata value: a scalar, a string, or an array of scalar/string values. Arrays are
// represented as a vector of scalar GgufValues (arrays are never nested in GGUF).
class GgufValue {
 public:
  using Scalar = std::variant<std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
                              std::uint32_t, std::int32_t, std::uint64_t, std::int64_t, float,
                              double, bool, std::string>;

  GgufValue() = default;
  explicit GgufValue(MetaType type, Scalar scalar)
      : type_(type), scalar_(std::move(scalar)) {}

  static GgufValue makeArray(MetaType elementType, std::vector<GgufValue> elements) {
    GgufValue v;
    v.type_ = MetaType::Array;
    v.arrayType_ = elementType;
    v.array_ = std::move(elements);
    return v;
  }

  MetaType type() const noexcept { return type_; }
  bool isArray() const noexcept { return type_ == MetaType::Array; }
  MetaType arrayElementType() const noexcept { return arrayType_; }
  const std::vector<GgufValue>& array() const noexcept { return array_; }
  const Scalar& scalar() const noexcept { return scalar_; }

  // Widening scalar accessors. Return nullopt for the wrong category (e.g. asU64 on a string,
  // or any accessor on an array). Integer/bool values widen into asU64/asI64; floats and any
  // integer widen into asF64 so numeric metadata can be read uniformly.
  std::optional<std::uint64_t> asU64() const;
  std::optional<std::int64_t> asI64() const;
  std::optional<double> asF64() const;
  std::optional<bool> asBool() const;
  const std::string* asString() const;

 private:
  MetaType type_ = MetaType::UInt32;
  Scalar scalar_{std::uint32_t{0}};
  MetaType arrayType_ = MetaType::UInt32;
  std::vector<GgufValue> array_;
};

}  // namespace qorvix::gguf
