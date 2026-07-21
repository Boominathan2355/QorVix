#include "qorvix/gguf/gguf_types.hpp"

namespace qorvix::gguf {

namespace {

// Indexed lookup keyed by raw type value. Block/type sizes match ggml's block structs; see
// gguf_types.hpp for the byte-size formula. `specSupported` marks the SPEC.md quant set
// (F32/F16/BF16/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K/Q8_0).
struct TypeRow {
  std::uint32_t value;
  GgmlTypeTraits traits;
};

constexpr TypeRow kTypes[] = {
    {0, {"F32", 1, 4, false, true}},
    {1, {"F16", 1, 2, false, true}},
    {2, {"Q4_0", 32, 18, true, true}},
    {3, {"Q4_1", 32, 20, true, false}},
    {6, {"Q5_0", 32, 22, true, true}},
    {7, {"Q5_1", 32, 24, true, false}},
    {8, {"Q8_0", 32, 34, true, true}},
    {9, {"Q8_1", 32, 36, true, false}},
    {10, {"Q2_K", 256, 84, true, false}},
    {11, {"Q3_K", 256, 110, true, false}},
    {12, {"Q4_K", 256, 144, true, true}},
    {13, {"Q5_K", 256, 176, true, true}},
    {14, {"Q6_K", 256, 210, true, true}},
    {15, {"Q8_K", 256, 292, true, false}},
    {16, {"IQ2_XXS", 256, 66, true, false}},
    {17, {"IQ2_XS", 256, 74, true, false}},
    {18, {"IQ3_XXS", 256, 98, true, false}},
    {19, {"IQ1_S", 256, 50, true, false}},
    {20, {"IQ4_NL", 32, 18, true, false}},
    {21, {"IQ3_S", 256, 110, true, false}},
    {22, {"IQ2_S", 256, 82, true, false}},
    {23, {"IQ4_XS", 256, 136, true, false}},
    {24, {"I8", 1, 1, false, false}},
    {25, {"I16", 1, 2, false, false}},
    {26, {"I32", 1, 4, false, false}},
    {27, {"I64", 1, 8, false, false}},
    {28, {"F64", 1, 8, false, false}},
    {29, {"IQ1_M", 256, 56, true, false}},
    {30, {"BF16", 1, 2, false, true}},
};

}  // namespace

const GgmlTypeTraits* ggmlTypeTraits(std::uint32_t type) {
  for (const auto& row : kTypes) {
    if (row.value == type) return &row.traits;
  }
  return nullptr;
}

std::string ggmlTypeName(std::uint32_t type) {
  if (const auto* t = ggmlTypeTraits(type)) return t->name;
  return "UNKNOWN(" + std::to_string(type) + ")";
}

const char* metaTypeName(MetaType type) {
  switch (type) {
    case MetaType::UInt8: return "uint8";
    case MetaType::Int8: return "int8";
    case MetaType::UInt16: return "uint16";
    case MetaType::Int16: return "int16";
    case MetaType::UInt32: return "uint32";
    case MetaType::Int32: return "int32";
    case MetaType::Float32: return "float32";
    case MetaType::Bool: return "bool";
    case MetaType::String: return "string";
    case MetaType::Array: return "array";
    case MetaType::UInt64: return "uint64";
    case MetaType::Int64: return "int64";
    case MetaType::Float64: return "float64";
  }
  return "invalid";
}

bool isScalarMetaType(MetaType type) { return type != MetaType::Array; }

std::optional<std::uint64_t> GgufValue::asU64() const {
  return std::visit(
      [](const auto& v) -> std::optional<std::uint64_t> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) return v ? 1u : 0u;
        else if constexpr (std::is_integral_v<T>) return static_cast<std::uint64_t>(v);
        else return std::nullopt;
      },
      scalar_);
}

std::optional<std::int64_t> GgufValue::asI64() const {
  return std::visit(
      [](const auto& v) -> std::optional<std::int64_t> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) return v ? 1 : 0;
        else if constexpr (std::is_integral_v<T>) return static_cast<std::int64_t>(v);
        else return std::nullopt;
      },
      scalar_);
}

std::optional<double> GgufValue::asF64() const {
  return std::visit(
      [](const auto& v) -> std::optional<double> {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, bool>) return std::nullopt;
        else if constexpr (std::is_arithmetic_v<T>) return static_cast<double>(v);
        else return std::nullopt;
      },
      scalar_);
}

std::optional<bool> GgufValue::asBool() const {
  if (const bool* b = std::get_if<bool>(&scalar_)) return *b;
  return std::nullopt;
}

const std::string* GgufValue::asString() const { return std::get_if<std::string>(&scalar_); }

}  // namespace qorvix::gguf
