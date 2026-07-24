#include "qorvix/runtime/qmatmul.hpp"

#if defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "qorvix/gguf/gguf_types.hpp"
#include "qorvix/runtime/dequant.hpp"

namespace qorvix::runtime {

namespace {
// Largest block size across all supported types (K-quants use 256).
constexpr int kMaxBlock = 256;

static inline float vecDotF32(const float* a, const float* b, int n) {
#if defined(__AVX2__)
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 15 < n; i += 16) {
    __m256 va0 = _mm256_loadu_ps(a + i);
    __m256 vb0 = _mm256_loadu_ps(b + i);
    sum0 = _mm256_fmadd_ps(va0, vb0, sum0);
    __m256 va1 = _mm256_loadu_ps(a + i + 8);
    __m256 vb1 = _mm256_loadu_ps(b + i + 8);
    sum1 = _mm256_fmadd_ps(va1, vb1, sum1);
  }
  for (; i + 7 < n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    sum0 = _mm256_fmadd_ps(va, vb, sum0);
  }
  sum0 = _mm256_add_ps(sum0, sum1);
  alignas(32) float tmp[8];
  _mm256_storeu_ps(tmp, sum0);
  float res = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < n; ++i) res += a[i] * b[i];
  return res;
#else
  float res = 0.0f;
  for (int i = 0; i < n; ++i) res += a[i] * b[i];
  return res;
#endif
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

#pragma omp parallel for schedule(static)
  for (int r = 0; r < rows; ++r) {
    const std::uint8_t* rowPtr = base + static_cast<std::size_t>(r) * rowBytes;
    float buf[kMaxBlock];
    float acc = 0.0f;
    for (int b = 0; b < nBlocks; ++b) {
      // Dequantize one block into the stack buffer, then fold it into the running dot product.
      dequantize(ggmlType, rowPtr + static_cast<std::size_t>(b) * traits->typeSize, buf, blockSize);
      const float* xb = x + static_cast<std::size_t>(b) * blockSize;
      acc += vecDotF32(buf, xb, blockSize);
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
