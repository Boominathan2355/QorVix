#include "qorvix/runtime/qmatmul.hpp"

#include "qorvix/gguf/gguf_types.hpp"
#include "qorvix/runtime/dequant.hpp"

namespace qorvix::runtime {

namespace {
// Largest block size across all supported types (K-quants use 256).
constexpr int kMaxBlock = 256;
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

#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const std::uint8_t* rowPtr = base + static_cast<std::size_t>(r) * rowBytes;
    float buf[kMaxBlock];
    float acc = 0.0f;
    for (int b = 0; b < nBlocks; ++b) {
      // Dequantize one block into the stack buffer, then fold it into the running dot product.
      dequantize(ggmlType, rowPtr + static_cast<std::size_t>(b) * traits->typeSize, buf, blockSize);
      const float* xb = x + static_cast<std::size_t>(b) * blockSize;
      for (int i = 0; i < blockSize; ++i) acc += buf[i] * xb[i];
    }
    out[r] = acc;
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
