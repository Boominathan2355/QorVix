#include "qorvix/runtime/qmatmul.hpp"

#include "qorvix/gguf/gguf_types.hpp"
#include "qorvix/runtime/cpu_features.hpp"
#include "qorvix/runtime/dequant.hpp"

namespace qorvix::runtime {

namespace {
// Largest block size across all supported types (K-quants use 256).
constexpr int kMaxBlock = 256;

// How many vectors qmatmulN accumulates per pass over a weight row. Bounds the stack accumulator
// (kVecTile floats) while keeping the number of passes over the weight matrix small.
constexpr int kVecTile = 64;

// Dispatched to the best SIMD kernel for the running CPU at startup (AVX2/NEON/scalar) — a portable
// build gets AVX2 with no -march=native. See runtime/cpu_features.cpp.
static inline float vecDotF32(const float* a, const float* b, int n) {
  return cpu::dotProductF32(a, b, n);
}
}  // namespace

bool qmatmulSupports(std::uint32_t ggmlType) {
  return canDequantize(ggmlType) && gguf::ggmlTypeTraits(ggmlType) != nullptr;
}

bool qmatmul(float* out, const void* weight, std::uint32_t ggmlType, const float* x, int rows,
             int cols) {
  const auto* traits = gguf::ggmlTypeTraits(ggmlType);
  if (!traits || !canDequantize(ggmlType)) return false;
  const int blockSize = static_cast<int>(traits->blockSize);
  if (blockSize <= 0 || blockSize > kMaxBlock || cols % blockSize != 0) return false;

  const int nBlocks = cols / blockSize;
  const std::size_t rowBytes = static_cast<std::size_t>(nBlocks) * traits->typeSize;
  const auto* base = static_cast<const std::uint8_t*>(weight);

  // Fill the scratch buffer with as many WHOLE BLOCKS as it holds, rather than one block per
  // call. For the K-quants (blockSize 256) this is the same single block as before, but the
  // small-block types were pathological: F16 and BF16 have blockSize 1, so a 384-column row meant
  // 384 dequantize() dispatches and 384 dot-product calls of length ONE — the SIMD kernel never
  // engaged, and per-call overhead dominated entirely. An F16 bge-small embed spent 5.5 s on four
  // tokens because of it. Batching to 256 elements cuts both call counts by 256x and lets
  // dotProductF32 reach its vector path.
  const int elemsPerBatch = (kMaxBlock / blockSize) * blockSize;  // <= 256, a multiple of blockSize

#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const std::uint8_t* rowPtr = base + static_cast<std::size_t>(r) * rowBytes;
    float buf[kMaxBlock];
    float acc = 0.0f;
    for (int e = 0; e < cols; e += elemsPerBatch) {
      const int n = cols - e < elemsPerBatch ? cols - e : elemsPerBatch;
      const std::size_t byteOff = static_cast<std::size_t>(e / blockSize) * traits->typeSize;
      dequantize(ggmlType, rowPtr + byteOff, buf, static_cast<std::size_t>(n));
      acc += vecDotF32(buf, x + e, n);
    }
    out[r] = acc;
  }
  return true;
}

bool qmatmulN(float* out, const void* weight, std::uint32_t ggmlType, const float* X, int nVec,
              int rows, int cols) {
  if (nVec <= 0) return false;
  if (nVec == 1) return qmatmul(out, weight, ggmlType, X, rows, cols);

  const auto* traits = gguf::ggmlTypeTraits(ggmlType);
  if (!traits || !canDequantize(ggmlType)) return false;
  const int blockSize = static_cast<int>(traits->blockSize);
  if (blockSize <= 0 || blockSize > kMaxBlock || cols % blockSize != 0) return false;

  const std::size_t rowBytes = static_cast<std::size_t>(cols / blockSize) * traits->typeSize;
  const auto* base = static_cast<const std::uint8_t*>(weight);
  const int elemsPerBatch = (kMaxBlock / blockSize) * blockSize;

#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const std::uint8_t* rowPtr = base + static_cast<std::size_t>(r) * rowBytes;
    // Tiled over vectors so the accumulator stays a bounded stack array. Each tile re-reads the
    // weight row once, so the matrix is streamed ceil(nVec/kVecTile) times instead of nVec —
    // 2 passes at nVec=256 rather than 256.
    for (int vBase = 0; vBase < nVec; vBase += kVecTile) {
      const int nv = nVec - vBase < kVecTile ? nVec - vBase : kVecTile;
      float acc[kVecTile] = {};
      float buf[kMaxBlock];
      for (int e = 0; e < cols; e += elemsPerBatch) {
        const int n = cols - e < elemsPerBatch ? cols - e : elemsPerBatch;
        const std::size_t byteOff = static_cast<std::size_t>(e / blockSize) * traits->typeSize;
        dequantize(ggmlType, rowPtr + byteOff, buf, static_cast<std::size_t>(n));
        for (int v = 0; v < nv; ++v) {
          acc[v] += vecDotF32(buf, X + static_cast<std::size_t>(vBase + v) * cols + e, n);
        }
      }
      for (int v = 0; v < nv; ++v) {
        out[static_cast<std::size_t>(vBase + v) * rows + r] = acc[v];
      }
    }
  }
  return true;
}

bool dequantRow(const void* weight, std::uint32_t ggmlType, int cols, int row, float* dst) {
  const auto* traits = gguf::ggmlTypeTraits(ggmlType);
  if (!traits || !canDequantize(ggmlType)) return false;
  const int blockSize = static_cast<int>(traits->blockSize);
  if (blockSize <= 0 || cols % blockSize != 0) return false;

  const std::size_t rowBytes = static_cast<std::size_t>(cols / blockSize) * traits->typeSize;
  const auto* rowPtr = static_cast<const std::uint8_t*>(weight) + static_cast<std::size_t>(row) * rowBytes;
  return dequantize(ggmlType, rowPtr, dst, static_cast<std::size_t>(cols));
}

}  // namespace qorvix::runtime
