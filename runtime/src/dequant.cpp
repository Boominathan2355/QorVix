#include "qorvix/runtime/dequant.hpp"

#include <cstring>

#include "qorvix/gguf/gguf_types.hpp"

namespace qorvix::runtime {

namespace {

using qorvix::gguf::GgmlType;

std::uint16_t readHalf(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

// Unpacks the 6-bit (scale, min) pair for sub-block j from a Q4_K/Q5_K 12-byte scales array.
// Verbatim from ggml's get_scale_min_k4.
void getScaleMinK4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

constexpr int kQK = 256;  // K-quant super-block size

// ---- legacy quants (32-element blocks) -----------------------------------------------------

void dequantQ8_0(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 34;  // half d + int8 qs[32]
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / 32; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const float d = fp16ToFloat(readHalf(p));
    const auto* qs = reinterpret_cast<const std::int8_t*>(p + 2);
    for (int i = 0; i < 32; ++i) dst[b * 32 + i] = d * qs[i];
  }
}

void dequantQ4_0(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 18;  // half d + uint8 qs[16]
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / 32; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const float d = fp16ToFloat(readHalf(p));
    const std::uint8_t* qs = p + 2;
    for (int i = 0; i < 16; ++i) {
      dst[b * 32 + i] = ((qs[i] & 0x0F) - 8) * d;
      dst[b * 32 + i + 16] = ((qs[i] >> 4) - 8) * d;
    }
  }
}

void dequantQ4_1(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 20;  // half d + half m + uint8 qs[16]
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / 32; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const float d = fp16ToFloat(readHalf(p));
    const float m = fp16ToFloat(readHalf(p + 2));
    const std::uint8_t* qs = p + 4;
    for (int i = 0; i < 16; ++i) {
      dst[b * 32 + i] = (qs[i] & 0x0F) * d + m;
      dst[b * 32 + i + 16] = (qs[i] >> 4) * d + m;
    }
  }
}

void dequantQ5_0(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 22;  // half d + uint8 qh[4] + uint8 qs[16]
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / 32; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const float d = fp16ToFloat(readHalf(p));
    std::uint32_t qh;
    std::memcpy(&qh, p + 2, 4);
    const std::uint8_t* qs = p + 6;
    for (int i = 0; i < 16; ++i) {
      const std::uint8_t xh0 = ((qh >> i) & 1) << 4;
      const std::uint8_t xh1 = ((qh >> (i + 16)) & 1) << 4;
      dst[b * 32 + i] = (((qs[i] & 0x0F) | xh0) - 16) * d;
      dst[b * 32 + i + 16] = (((qs[i] >> 4) | xh1) - 16) * d;
    }
  }
}

void dequantQ5_1(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 24;  // half d + half m + uint8 qh[4] + uint8 qs[16]
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / 32; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const float d = fp16ToFloat(readHalf(p));
    const float m = fp16ToFloat(readHalf(p + 2));
    std::uint32_t qh;
    std::memcpy(&qh, p + 4, 4);
    const std::uint8_t* qs = p + 8;
    for (int i = 0; i < 16; ++i) {
      const std::uint8_t xh0 = ((qh >> i) & 1) << 4;
      const std::uint8_t xh1 = ((qh >> (i + 16)) & 1) << 4;
      dst[b * 32 + i] = ((qs[i] & 0x0F) | xh0) * d + m;
      dst[b * 32 + i + 16] = ((qs[i] >> 4) | xh1) * d + m;
    }
  }
}

void dequantQ8_1(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 36;  // half d + half s + int8 qs[32]
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / 32; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const float d = fp16ToFloat(readHalf(p));
    const auto* qs = reinterpret_cast<const std::int8_t*>(p + 4);
    for (int i = 0; i < 32; ++i) dst[b * 32 + i] = d * qs[i];
  }
}

// ---- K-quants (256-element super-blocks) ---------------------------------------------------

void dequantQ4_K(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 144;  // half d + half dmin + scales[12] + qs[128]
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / kQK; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const float d = fp16ToFloat(readHalf(p));
    const float dmin = fp16ToFloat(readHalf(p + 2));
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* q = p + 16;
    float* y = dst + b * kQK;

    int is = 0;
    for (int j = 0; j < kQK; j += 64) {
      std::uint8_t sc, m;
      getScaleMinK4(is + 0, scales, sc, m);
      const float d1 = d * sc, m1 = dmin * m;
      getScaleMinK4(is + 1, scales, sc, m);
      const float d2 = d * sc, m2 = dmin * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

void dequantQ5_K(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 176;  // half d + half dmin + scales[12] + qh[32] + qs[128]
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / kQK; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const float d = fp16ToFloat(readHalf(p));
    const float dmin = fp16ToFloat(readHalf(p + 2));
    const std::uint8_t* scales = p + 4;
    const std::uint8_t* qh = p + 16;
    const std::uint8_t* ql = p + 48;
    float* y = dst + b * kQK;

    int is = 0;
    std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < kQK; j += 64) {
      std::uint8_t sc, m;
      getScaleMinK4(is + 0, scales, sc, m);
      const float d1 = d * sc, m1 = dmin * m;
      getScaleMinK4(is + 1, scales, sc, m);
      const float d2 = d * sc, m2 = dmin * m;
      for (int l = 0; l < 32; ++l)
        *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l)
        *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
  }
}

void dequantQ6_K(const std::uint8_t* src, float* dst, std::size_t n) {
  constexpr int blockBytes = 210;  // ql[128] + qh[64] + int8 scales[16] + half d
#pragma omp parallel for schedule(static)
  for (std::size_t b = 0; b < n / kQK; ++b) {
    const std::uint8_t* p = src + b * blockBytes;
    const std::uint8_t* ql = p;
    const std::uint8_t* qh = p + 128;
    const auto* scales = reinterpret_cast<const std::int8_t*>(p + 192);
    const float d = fp16ToFloat(readHalf(p + 208));
    float* y = dst + b * kQK;

    for (int seg = 0; seg < kQK; seg += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const std::int8_t q1 =
            static_cast<std::int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const std::int8_t q2 =
            static_cast<std::int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const std::int8_t q3 =
            static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const std::int8_t q4 =
            static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * scales[is + 0] * q1;
        y[l + 32] = d * scales[is + 2] * q2;
        y[l + 64] = d * scales[is + 4] * q3;
        y[l + 96] = d * scales[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      scales += 8;
    }
  }
}

void dequantF16(const std::uint8_t* src, float* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = fp16ToFloat(readHalf(src + i * 2));
}

void dequantBF16(const std::uint8_t* src, float* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = bf16ToFloat(readHalf(src + i * 2));
}

}  // namespace

float fp16ToFloat(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  std::uint32_t exp = (h >> 10) & 0x1F;
  std::uint32_t mant = h & 0x3FF;
  std::uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FF;
      f = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    f = sign | 0x7F800000u | (mant << 13);
  } else {
    exp = exp - 15 + 127;
    f = sign | (exp << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

float bf16ToFloat(std::uint16_t b) {
  const std::uint32_t f = static_cast<std::uint32_t>(b) << 16;
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

bool canDequantize(std::uint32_t t) {
  switch (static_cast<GgmlType>(t)) {
    case GgmlType::F32:
    case GgmlType::F16:
    case GgmlType::BF16:
    case GgmlType::Q8_0:
    case GgmlType::Q8_1:
    case GgmlType::Q4_0:
    case GgmlType::Q4_1:
    case GgmlType::Q5_0:
    case GgmlType::Q5_1:
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
    case GgmlType::Q6_K:
      return true;
    default:
      return false;
  }
}

bool dequantize(std::uint32_t t, const void* src, float* dst, std::size_t n) {
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  switch (static_cast<GgmlType>(t)) {
    case GgmlType::F32:
      std::memcpy(dst, src, n * sizeof(float));
      return true;
    case GgmlType::F16: dequantF16(bytes, dst, n); return true;
    case GgmlType::BF16: dequantBF16(bytes, dst, n); return true;
    case GgmlType::Q8_0: dequantQ8_0(bytes, dst, n); return true;
    case GgmlType::Q8_1: dequantQ8_1(bytes, dst, n); return true;
    case GgmlType::Q4_0: dequantQ4_0(bytes, dst, n); return true;
    case GgmlType::Q4_1: dequantQ4_1(bytes, dst, n); return true;
    case GgmlType::Q5_0: dequantQ5_0(bytes, dst, n); return true;
    case GgmlType::Q5_1: dequantQ5_1(bytes, dst, n); return true;
    case GgmlType::Q4_K: dequantQ4_K(bytes, dst, n); return true;
    case GgmlType::Q5_K: dequantQ5_K(bytes, dst, n); return true;
    case GgmlType::Q6_K: dequantQ6_K(bytes, dst, n); return true;
    default: return false;
  }
}

}  // namespace qorvix::runtime
