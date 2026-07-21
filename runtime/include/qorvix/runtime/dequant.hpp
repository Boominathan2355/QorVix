#pragma once

#include <cstddef>
#include <cstdint>

// Dequantization of GGUF tensor data to F32 for the CPU reference runtime. Block layouts and
// formulas mirror ggml exactly (values must match llama.cpp bit-for-bit). Native quantized
// kernels that skip this dequant step are Phase 6.
namespace qorvix::runtime {

// IEEE-754 half / bfloat16 to float. Exposed for tests and metadata (GGUF scales are fp16).
float fp16ToFloat(std::uint16_t h);
float bf16ToFloat(std::uint16_t b);

// True if `ggmlType` can be dequantized by dequantize() below.
bool canDequantize(std::uint32_t ggmlType);

// Dequantizes `nElements` values from `src` into `dst`. `nElements` must be a multiple of the
// type's block size (the caller validates via the GGUF tensor traits). Returns false for an
// unsupported type. `src` must hold the full quantized byte span for those elements.
bool dequantize(std::uint32_t ggmlType, const void* src, float* dst, std::size_t nElements);

}  // namespace qorvix::runtime
