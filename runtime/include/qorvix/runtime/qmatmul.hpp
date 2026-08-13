#pragma once

#include <cstddef>
#include <cstdint>

// Native quantized matmul (SPEC "Native Quantization Kernels": do not dequantize to F32 first).
// The weight matrix stays in its GGUF block form in memory (~1/8 the size of F32 for Q4_K); each
// row's blocks are dequantized one at a time into a tiny stack buffer and immediately dotted with
// the input, so the full F32 weight is never materialized. This is the CPU reference for the
// direct-quantized GEMV; the CUDA kernel form (Phase 8) computes the same thing on-device.
namespace qorvix::runtime {

// True if `qmatmul`/`dequantRow` support this ggml type (block-quantized or F32/F16/BF16).
bool qmatmulSupports(std::uint32_t ggmlType);

// out[r] = dot(dequant(weightRow r), x), for r in [0,rows). `weight` points at the tensor's raw
// bytes; row r occupies (cols/blockSize)*typeSize bytes. Requires cols % blockSize == 0.
// Returns false if the type is unsupported or cols isn't a block multiple.
bool qmatmul(float* out, const void* weight, std::uint32_t ggmlType, const float* x, int rows,
             int cols);

// Dequantizes a single row (`cols` elements) of a quantized matrix into `dst` — used for
// embedding lookup, where only one token's row is needed.
// Batched GEMV: out is [nVec, rows] row-major, X is [nVec, cols] row-major.
//
// The point is to dequantize each weight block ONCE and fold it into all nVec dot products,
// instead of re-streaming the whole matrix once per vector. qmatmul is a GEMV, so an encoder over
// N tokens — or a decoder prefilling an N-token prompt — pays N passes over every weight for work
// that needs one. At N=256 that is two orders of magnitude of avoidable memory traffic.
//
// Mathematically identical to calling qmatmul() once per vector: each output element accumulates
// the same blocks in the same order, so results are bit-identical, not merely close.
bool qmatmulN(float* out, const void* weight, std::uint32_t ggmlType, const float* X, int nVec,
              int rows, int cols);

bool dequantRow(const void* weight, std::uint32_t ggmlType, int cols, int row, float* dst);

}  // namespace qorvix::runtime
