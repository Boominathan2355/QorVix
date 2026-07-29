#include "qorvix/runtime/cpu_features.hpp"

#include <thread>

// ---- architecture detection (compile time) --------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define QORVIX_ARCH_X86 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define QORVIX_ARCH_ARM64 1
#elif defined(__riscv)
#define QORVIX_ARCH_RISCV 1
#endif

#if defined(QORVIX_ARCH_X86) && (defined(__GNUC__) || defined(__clang__))
#define QORVIX_X86_SIMD 1  // GCC/Clang: build AVX2 kernels via target attributes, no -march needed
#include <immintrin.h>
#endif

#if defined(QORVIX_ARCH_ARM64)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif

#if defined(QORVIX_ARCH_X86) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace qorvix::runtime::cpu {
namespace {

// ---- dot-product kernels ------------------------------------------------------------------
float dot_scalar(const float* a, const float* b, int n) {
  float r = 0.0f;
  for (int i = 0; i < n; ++i) r += a[i] * b[i];
  return r;
}

#if defined(QORVIX_X86_SIMD)
// Compiled with AVX2+FMA codegen for THIS function only (the rest of the TU stays baseline), so a
// portable default build still contains this kernel and dispatches to it only when the CPU has
// AVX2. Accumulation order matches the previous -march=native path (tests already tolerate it).
__attribute__((target("avx2,fma"))) float dot_avx2(const float* a, const float* b, int n) {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 15 < n; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  for (; i + 7 < n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  s0 = _mm256_add_ps(s0, s1);
  alignas(32) float t[8];
  _mm256_storeu_ps(t, s0);
  float r = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
}

// 512-bit variant for CPUs with AVX-512F. Compiled here but can only ACTIVATE (and thus be
// verified) on AVX-512 hardware; the x86 reference environment is AVX2-only, so this stays
// code-complete but hardware-unverified.
__attribute__((target("avx512f"))) float dot_avx512(const float* a, const float* b, int n) {
  __m512 s = _mm512_setzero_ps();
  int i = 0;
  for (; i + 15 < n; i += 16) s = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s);
  float r = _mm512_reduce_add_ps(s);
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
}
#endif

#if defined(QORVIX_ARCH_ARM64)
// NEON is baseline on aarch64 (incl. Apple Silicon). Compiled but hardware-unverified in the x86
// reference environment.
float dot_neon(const float* a, const float* b, int n) {
  float32x4_t s = vdupq_n_f32(0.0f);
  int i = 0;
  for (; i + 3 < n; i += 4) s = vmlaq_f32(s, vld1q_f32(a + i), vld1q_f32(b + i));
  float r = vaddvq_f32(s);
  for (; i < n; ++i) r += a[i] * b[i];
  return r;
}
#endif

Features detect() {
  Features f;
  f.hardwareThreads = std::thread::hardware_concurrency();
  if (f.hardwareThreads == 0) f.hardwareThreads = 1;

#if defined(QORVIX_ARCH_X86)
  f.arch = "x86_64";
#if defined(__GNUC__) || defined(__clang__)
#if defined(__GNUC__) && !defined(__clang__)
  __builtin_cpu_init();
#endif
  f.sse2 = __builtin_cpu_supports("sse2");
  f.avx = __builtin_cpu_supports("avx");
  f.avx2 = __builtin_cpu_supports("avx2");
  f.fma = __builtin_cpu_supports("fma");
  f.avx512f = __builtin_cpu_supports("avx512f");
#elif defined(_MSC_VER)
  int r[4];
  __cpuid(r, 1);
  f.sse2 = (r[3] & (1 << 26)) != 0;
  f.avx = (r[2] & (1 << 28)) != 0;
  f.fma = (r[2] & (1 << 12)) != 0;
  __cpuidex(r, 7, 0);
  f.avx2 = (r[1] & (1 << 5)) != 0;
  f.avx512f = (r[1] & (1 << 16)) != 0;
#endif
#elif defined(QORVIX_ARCH_ARM64)
  f.arch = "aarch64";
  f.neon = true;  // mandatory on aarch64
#if defined(__linux__) && defined(AT_HWCAP)
#if defined(HWCAP_SVE)
  f.sve = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
#endif
#endif
#elif defined(QORVIX_ARCH_RISCV)
  f.arch = "riscv64";
#endif
  return f;
}

using DotFn = float (*)(const float*, const float*, int);

DotFn pickDot() {
  const Features& f = features();
#if defined(QORVIX_X86_SIMD)
  if (f.avx512f) return &dot_avx512;
  if (f.avx2 && f.fma) return &dot_avx2;
#endif
#if defined(QORVIX_ARCH_ARM64)
  if (f.neon) return &dot_neon;
#endif
  (void)f;
  return &dot_scalar;
}

const DotFn g_dot = pickDot();

}  // namespace

const Features& features() {
  static const Features f = detect();
  return f;
}

float dotProductF32(const float* a, const float* b, int n) { return g_dot(a, b, n); }

const char* activeDotKernel() {
#if defined(QORVIX_X86_SIMD)
  if (g_dot == &dot_avx512) return "avx512";
  if (g_dot == &dot_avx2) return "avx2";
#endif
#if defined(QORVIX_ARCH_ARM64)
  if (g_dot == &dot_neon) return "neon";
#endif
  return "scalar";
}

std::string summary() {
  const Features& f = features();
  std::string s = "arch=" + f.arch + " threads=" + std::to_string(f.hardwareThreads) + " | simd: ";
  if (f.arch == "x86_64") {
    s += std::string(f.sse2 ? "sse2 " : "") + (f.avx ? "avx " : "") + (f.avx2 ? "avx2 " : "") +
         (f.fma ? "fma " : "") + (f.avx512f ? "avx512f " : "");
  } else if (f.arch == "aarch64") {
    s += std::string(f.neon ? "neon " : "") + (f.sve ? "sve " : "");
  } else {
    s += "(scalar only) ";
  }
  s += "| active dot kernel: ";
  s += activeDotKernel();
  return s;
}

}  // namespace qorvix::runtime::cpu
