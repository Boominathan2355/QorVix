#pragma once

// Shared internals of the GGUF weight loaders: resolving a tensor's bytes inside the file's mmap,
// borrowing a quantized matmul weight, and materializing a small F32 vector.
//
// These began as an anonymous namespace inside weights.cpp, moved to a private header when the
// BERT encoder loader needed the identical logic, and moved here when the CLIP vision tower became
// the third consumer — at which point reaching into another module's src/ directory was worse than
// admitting this is shared.
//
// `detail` is the contract: these are loader internals, not a stable API. Anything outside a
// weight loader should be using Weights / EncoderWeights / ClipWeights instead.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/dequant.hpp"
#include "qorvix/runtime/weights.hpp"

namespace qorvix::runtime::detail {

// Resolves a tensor's raw bytes inside the file's mmap. Sets error and returns nullptr if the
// tensor is missing, the element count is wrong, or the file wasn't memory-mapped.
inline const std::uint8_t* tensorBytes(const gguf::GgufFile& file, const std::string& name,
                                       std::size_t expectElements, const gguf::GgufTensor** outT,
                                       std::string& error) {
  const gguf::GgufTensor* t = file.tensor(name);
  if (!t) {
    error = "missing tensor '" + name + "'";
    return nullptr;
  }
  if (t->nElements != expectElements) {
    error = "tensor '" + name + "' has " + std::to_string(t->nElements) + " elements, expected " +
            std::to_string(expectElements);
    return nullptr;
  }
  const auto all = file.mapping().bytes();
  const std::uint64_t start = file.dataOffset() + t->offset;
  if (all.empty() || start + t->nBytes > all.size()) {
    error = "tensor '" + name + "' data is out of range (file not opened via mmap?)";
    return nullptr;
  }
  *outT = t;
  return reinterpret_cast<const std::uint8_t*>(all.data()) + start;
}

// Borrows a matmul weight [rows, cols] straight from the mmap, keeping it quantized.
inline bool loadMat(const gguf::GgufFile& file, const std::string& name, int rows, int cols,
                    WeightMat& out, std::string& error) {
  const gguf::GgufTensor* t = nullptr;
  const std::uint8_t* ptr =
      tensorBytes(file, name, static_cast<std::size_t>(rows) * cols, &t, error);
  if (!ptr) return false;
  if (!qmatmulSupports(t->typeRaw)) {
    error = "tensor '" + name + "' uses unsupported type " + t->typeName();
    return false;
  }
  out = WeightMat::quantized(ptr, t->typeRaw, rows, cols);
  return true;
}

// Copies a small norm/bias vector, dequantizing to F32 (norms are F32 in practice).
inline bool loadVec(const gguf::GgufFile& file, const std::string& name, int n,
                    std::vector<float>& out, std::string& error) {
  const gguf::GgufTensor* t = nullptr;
  const std::uint8_t* ptr = tensorBytes(file, name, static_cast<std::size_t>(n), &t, error);
  if (!ptr) return false;
  if (!canDequantize(t->typeRaw)) {
    error = "tensor '" + name + "' uses unsupported type " + t->typeName();
    return false;
  }
  out.resize(n);
  return dequantize(t->typeRaw, ptr, out.data(), static_cast<std::size_t>(n));
}

// Optional variants: an ABSENT tensor leaves `out` untouched and succeeds. A tensor that is
// present but malformed still fails with `error` set — "optional" must mean "may be absent", not
// "errors are ignored", or a corrupt position table would silently become a zero-filled one.
inline bool loadMatOpt(const gguf::GgufFile& file, const std::string& name, int rows, int cols,
                       WeightMat& out, std::string& error) {
  if (!file.tensor(name)) return true;
  return loadMat(file, name, rows, cols, out, error);
}

inline bool loadVecOpt(const gguf::GgufFile& file, const std::string& name, int n,
                       std::vector<float>& out, std::string& error) {
  if (!file.tensor(name)) return true;
  return loadVec(file, name, n, out, error);
}

inline std::string blk(int i, const char* suffix) {
  return "blk." + std::to_string(i) + "." + suffix;
}

}  // namespace qorvix::runtime::detail
