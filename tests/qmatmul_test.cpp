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
