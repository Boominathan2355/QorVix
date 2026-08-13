#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <vector>

#include "qorvix/gguf/gguf_types.hpp"
#include "qorvix/runtime/dequant.hpp"
#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/qmatmul.hpp"

using namespace qorvix::runtime;
using qorvix::gguf::GgmlType;
using Catch::Matchers::WithinAbs;

namespace {

void pushHalf(std::vector<std::uint8_t>& v, std::uint16_t h) {
  v.push_back(h & 0xFF);
  v.push_back(h >> 8);
}

// Appends one Q8_0 block (d, then 32 int8 quants).
void pushQ8_0Block(std::vector<std::uint8_t>& v, std::uint16_t dHalf, int base) {
  pushHalf(v, dHalf);
  for (int i = 0; i < 32; ++i) v.push_back(static_cast<std::uint8_t>(base + i));
}

// Reference: dequantize the whole weight, then run the (verified) F32 matmul.
std::vector<float> reference(std::uint32_t type, const std::vector<std::uint8_t>& w,
                             const std::vector<float>& x, int rows, int cols) {
  std::vector<float> deq(static_cast<std::size_t>(rows) * cols);
  REQUIRE(dequantize(type, w.data(), deq.data(), deq.size()));
  std::vector<float> out(rows);
  ops::matmul(out.data(), deq.data(), x.data(), rows, cols);
  return out;
}

}  // namespace

TEST_CASE("qmatmul(Q8_0) matches dequant + F32 matmul", "[qmatmul]") {
  const int rows = 3, cols = 64;  // 2 blocks per row
  std::vector<std::uint8_t> w;
  for (int r = 0; r < rows; ++r) {
    pushQ8_0Block(w, 0x3800, r * 2);       // d = 0.5
    pushQ8_0Block(w, 0x3C00, r * 2 + 1);   // d = 1.0
  }
  std::vector<float> x(cols);
  for (int i = 0; i < cols; ++i) x[i] = 0.1f * ((i % 5) - 2);

  const auto ref = reference(static_cast<std::uint32_t>(GgmlType::Q8_0), w, x, rows, cols);
  std::vector<float> out(rows);
  REQUIRE(qmatmul(out.data(), w.data(), static_cast<std::uint32_t>(GgmlType::Q8_0), x.data(),
                  rows, cols));
  for (int r = 0; r < rows; ++r) REQUIRE_THAT(out[r], WithinAbs(ref[r], 1e-3f));
}

TEST_CASE("qmatmul(Q4_K) matches dequant + F32 matmul", "[qmatmul]") {
  const int rows = 2, cols = 256;  // one super-block per row
  std::vector<std::uint8_t> w;
  for (int r = 0; r < rows; ++r) {
    std::vector<std::uint8_t> block(144, 0);
    block[0] = 0x00;
    block[1] = 0x3C;  // d = 1.0
    for (int i = 0; i < 12; ++i) block[4 + i] = 1;   // scale fields = 1
    for (int i = 0; i < 128; ++i) block[16 + i] = static_cast<std::uint8_t>((r + i) & 0x0F);
    w.insert(w.end(), block.begin(), block.end());
  }
  std::vector<float> x(cols);
  for (int i = 0; i < cols; ++i) x[i] = 0.01f * ((i % 7) - 3);

  const auto ref = reference(static_cast<std::uint32_t>(GgmlType::Q4_K), w, x, rows, cols);
  std::vector<float> out(rows);
  REQUIRE(qmatmul(out.data(), w.data(), static_cast<std::uint32_t>(GgmlType::Q4_K), x.data(),
                  rows, cols));
  for (int r = 0; r < rows; ++r) REQUIRE_THAT(out[r], WithinAbs(ref[r], 1e-2f));
}

TEST_CASE("dequantRow extracts a single row", "[qmatmul]") {
  const int rows = 4, cols = 32;
  std::vector<std::uint8_t> w;
  for (int r = 0; r < rows; ++r) pushQ8_0Block(w, 0x3C00, r * 10);  // d=1.0, qs = r*10 + i

  std::vector<float> row(cols);
  REQUIRE(dequantRow(w.data(), static_cast<std::uint32_t>(GgmlType::Q8_0), cols, /*row=*/2, row.data()));
  for (int i = 0; i < cols; ++i) REQUIRE_THAT(row[i], WithinAbs(static_cast<float>(20 + i), 1e-3f));
}

TEST_CASE("qmatmul rejects bad shapes and types", "[qmatmul]") {
  std::vector<std::uint8_t> w(34, 0);
  std::vector<float> x(30, 1.0f), out(1);
  // cols not a multiple of the Q8_0 block size (32).
  REQUIRE_FALSE(qmatmul(out.data(), w.data(), static_cast<std::uint32_t>(GgmlType::Q8_0),
                        x.data(), 1, 30));
  // unsupported type.
  REQUIRE_FALSE(qmatmulSupports(static_cast<std::uint32_t>(GgmlType::Q2_K)));
}

namespace {
// A Q8_0 weight of `rows` rows x `blocks` blocks, with well-formed fp16 scales. Building the
// bytes arbitrarily is a trap: a random byte pair is often a NaN or Inf half, and then both the
// batched and unbatched paths produce NaN, which compares UNEQUAL to itself and fails an
// equality test that the code actually passes.
std::vector<std::uint8_t> q8Weight(int rows, int blocks) {
  static const std::uint16_t kScales[] = {0x3800, 0x3C00, 0x3400, 0x3E00};  // 0.5, 1.0, 0.25, 1.5
  std::vector<std::uint8_t> w;
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < blocks; ++b) pushQ8_0Block(w, kScales[(r + b) % 4], r * 3 + b);
  }
  return w;
}
}  // namespace

TEST_CASE("qmatmulN is bit-identical to qmatmul called once per vector", "[qmatmul]") {
  // The batched GEMV exists to stop re-streaming a weight matrix once per token. It accumulates
  // the same blocks in the same order per output element, so "close enough" is not the bar:
  // anything short of bit-identical would mean the batched path reassociated the sum, and an
  // embedding would then drift with how many tokens happened to be in the batch.
  constexpr int kRows = 12, kCols = 64, kVec = 5;  // 2 blocks per row
  const auto w = q8Weight(kRows, kCols / 32);
  std::vector<float> X(static_cast<std::size_t>(kVec) * kCols);
  for (std::size_t i = 0; i < X.size(); ++i) X[i] = 0.01f * static_cast<float>(i % 23) - 0.1f;

  const auto type = static_cast<std::uint32_t>(GgmlType::Q8_0);
  std::vector<float> batched(static_cast<std::size_t>(kVec) * kRows, 0.0f);
  REQUIRE(qmatmulN(batched.data(), w.data(), type, X.data(), kVec, kRows, kCols));

  for (int v = 0; v < kVec; ++v) {
    std::vector<float> single(kRows, 0.0f);
    REQUIRE(qmatmul(single.data(), w.data(), type, X.data() + v * kCols, kRows, kCols));
    for (int r = 0; r < kRows; ++r) {
      REQUIRE(batched[static_cast<std::size_t>(v) * kRows + r] == single[r]);
    }
  }
}

TEST_CASE("qmatmulN handles more vectors than its internal tile", "[qmatmul]") {
  // Above the tile width the weight row is streamed more than once; the accumulator must still
  // land in the right output slots.
  constexpr int kRows = 4, kCols = 32, kVec = 70;  // kVec > the 64-wide tile
  const auto w = q8Weight(kRows, kCols / 32);
  std::vector<float> X(static_cast<std::size_t>(kVec) * kCols);
  for (std::size_t i = 0; i < X.size(); ++i) X[i] = static_cast<float>(i % 7) * 0.05f;

  const auto type = static_cast<std::uint32_t>(GgmlType::Q8_0);
  std::vector<float> batched(static_cast<std::size_t>(kVec) * kRows, -1.0f);
  REQUIRE(qmatmulN(batched.data(), w.data(), type, X.data(), kVec, kRows, kCols));

  for (int v = 0; v < kVec; ++v) {
    std::vector<float> single(kRows, 0.0f);
    REQUIRE(qmatmul(single.data(), w.data(), type, X.data() + v * kCols, kRows, kCols));
    for (int r = 0; r < kRows; ++r) {
      REQUIRE(batched[static_cast<std::size_t>(v) * kRows + r] == single[r]);
    }
  }
}

TEST_CASE("qmatmulN with one vector matches qmatmul", "[qmatmul]") {
  constexpr int kRows = 3, kCols = 32;
  const auto w = q8Weight(kRows, kCols / 32);
  std::vector<float> x(kCols, 0.25f);
  const auto type = static_cast<std::uint32_t>(GgmlType::Q8_0);

  std::vector<float> a(kRows), b(kRows);
  REQUIRE(qmatmulN(a.data(), w.data(), type, x.data(), 1, kRows, kCols));
  REQUIRE(qmatmul(b.data(), w.data(), type, x.data(), kRows, kCols));
  REQUIRE(a == b);
}
