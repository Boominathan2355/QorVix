#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <vector>

#include "qorvix/gguf/gguf_types.hpp"
#include "qorvix/runtime/dequant.hpp"

using namespace qorvix::runtime;
using qorvix::gguf::GgmlType;
using Catch::Matchers::WithinAbs;

namespace {
void pushHalf(std::vector<std::uint8_t>& v, std::uint16_t h) {
  v.push_back(h & 0xFF);
  v.push_back(h >> 8);
}
constexpr std::uint16_t kHalf_1_0 = 0x3C00;  // 1.0
constexpr std::uint16_t kHalf_0_5 = 0x3800;  // 0.5
constexpr std::uint16_t kHalf_2_0 = 0x4000;  // 2.0
}  // namespace

TEST_CASE("fp16/bf16 conversion of known encodings", "[dequant]") {
  REQUIRE_THAT(fp16ToFloat(kHalf_1_0), WithinAbs(1.0f, 0.0f));
  REQUIRE_THAT(fp16ToFloat(kHalf_0_5), WithinAbs(0.5f, 0.0f));
  REQUIRE_THAT(fp16ToFloat(0x0000), WithinAbs(0.0f, 0.0f));
  REQUIRE_THAT(fp16ToFloat(0xC000), WithinAbs(-2.0f, 0.0f));  // -2.0
  REQUIRE_THAT(bf16ToFloat(0x3F80), WithinAbs(1.0f, 0.0f));   // 1.0
  REQUIRE_THAT(bf16ToFloat(0x4040), WithinAbs(3.0f, 0.0f));   // 3.0
}

TEST_CASE("Q8_0 dequantizes d * qs", "[dequant]") {
  // One block of 32: d = 0.5, qs = 0,1,2,...,31 -> value[i] = 0.5 * i.
  std::vector<std::uint8_t> buf;
  pushHalf(buf, kHalf_0_5);
  for (int i = 0; i < 32; ++i) buf.push_back(static_cast<std::uint8_t>(i));

  std::vector<float> out(32);
  REQUIRE(dequantize(static_cast<std::uint32_t>(GgmlType::Q8_0), buf.data(), out.data(), 32));
  for (int i = 0; i < 32; ++i) REQUIRE_THAT(out[i], WithinAbs(0.5f * i, 1e-6f));
}

TEST_CASE("Q8_0 handles negative int8 quants", "[dequant]") {
  std::vector<std::uint8_t> buf;
  pushHalf(buf, kHalf_1_0);
  for (int i = 0; i < 32; ++i) buf.push_back(static_cast<std::uint8_t>(-i));  // 0,-1,-2,...
  std::vector<float> out(32);
  dequantize(static_cast<std::uint32_t>(GgmlType::Q8_0), buf.data(), out.data(), 32);
  for (int i = 0; i < 32; ++i) REQUIRE_THAT(out[i], WithinAbs(static_cast<float>(-i), 1e-6f));
}

TEST_CASE("Q4_0 dequantizes (nibble - 8) * d with split layout", "[dequant]") {
  // d = 1.0; every qs byte = 0x80 -> low nibble 0 -> -8, high nibble 8 -> 0.
  // So dst[0..15] = -8, dst[16..31] = 0.
  std::vector<std::uint8_t> buf;
  pushHalf(buf, kHalf_1_0);
  for (int i = 0; i < 16; ++i) buf.push_back(0x80);
  std::vector<float> out(32);
  dequantize(static_cast<std::uint32_t>(GgmlType::Q4_0), buf.data(), out.data(), 32);
  for (int i = 0; i < 16; ++i) REQUIRE_THAT(out[i], WithinAbs(-8.0f, 1e-6f));
  for (int i = 16; i < 32; ++i) REQUIRE_THAT(out[i], WithinAbs(0.0f, 1e-6f));
}

TEST_CASE("Q4_1 dequantizes nibble * d + m", "[dequant]") {
  // d = 2.0, m = 0.5; qs byte = 0x31 -> low nibble 1 -> 2.5, high nibble 3 -> 6.5.
  std::vector<std::uint8_t> buf;
  pushHalf(buf, kHalf_2_0);
  pushHalf(buf, kHalf_0_5);
  for (int i = 0; i < 16; ++i) buf.push_back(0x31);
  std::vector<float> out(32);
  dequantize(static_cast<std::uint32_t>(GgmlType::Q4_1), buf.data(), out.data(), 32);
  for (int i = 0; i < 16; ++i) REQUIRE_THAT(out[i], WithinAbs(1 * 2.0f + 0.5f, 1e-6f));
  for (int i = 16; i < 32; ++i) REQUIRE_THAT(out[i], WithinAbs(3 * 2.0f + 0.5f, 1e-6f));
}

TEST_CASE("Q5_0 combines the 5th bit from qh", "[dequant]") {
  // d = 1.0; qs all 0x00, qh with bit 0 set -> element 0 gets high bit: (0 | 16) - 16 = 0.
  // With qh = 0, element 0 = (0) - 16 = -16.
  std::vector<std::uint8_t> buf;
  pushHalf(buf, kHalf_1_0);
  std::uint32_t qh = 0;  // all high bits clear
  for (int k = 0; k < 4; ++k) buf.push_back(static_cast<std::uint8_t>(qh >> (8 * k)));
  for (int i = 0; i < 16; ++i) buf.push_back(0x00);
  std::vector<float> out(32);
  dequantize(static_cast<std::uint32_t>(GgmlType::Q5_0), buf.data(), out.data(), 32);
  for (int i = 0; i < 32; ++i) REQUIRE_THAT(out[i], WithinAbs(-16.0f, 1e-6f));

  // Now set qh bit 0 -> element 0 becomes (0 | 16) - 16 = 0.
  buf[2] = 0x01;
  dequantize(static_cast<std::uint32_t>(GgmlType::Q5_0), buf.data(), out.data(), 32);
  REQUIRE_THAT(out[0], WithinAbs(0.0f, 1e-6f));
  REQUIRE_THAT(out[1], WithinAbs(-16.0f, 1e-6f));
}

TEST_CASE("Q6_K super-block with uniform inputs", "[dequant]") {
  // ql=0, qh=0 -> every quant = (0) - 32 = -32. scales all 1, d = 1.0 -> all outputs = -32.
  std::vector<std::uint8_t> buf(210, 0);
  for (int i = 0; i < 16; ++i) buf[192 + i] = 1;  // int8 scales = 1
  buf[208] = kHalf_1_0 & 0xFF;
  buf[209] = kHalf_1_0 >> 8;
  std::vector<float> out(256);
  REQUIRE(dequantize(static_cast<std::uint32_t>(GgmlType::Q6_K), buf.data(), out.data(), 256));
  for (int i = 0; i < 256; ++i) REQUIRE_THAT(out[i], WithinAbs(-32.0f, 1e-4f));

  // Set ql[0] low nibble to 0xF -> q1 for element 0 = 15 - 32 = -17, scaled by d*scale(1) = -17.
  buf[0] = 0x0F;
  dequantize(static_cast<std::uint32_t>(GgmlType::Q6_K), buf.data(), out.data(), 256);
  REQUIRE_THAT(out[0], WithinAbs(-17.0f, 1e-4f));
}

TEST_CASE("Q4_K super-block with uniform scales", "[dequant]") {
  // d = 1.0, dmin = 0.0; scales[0..3] give 6-bit scale=1,min=0 for sub-blocks; qs nibbles = 0.
  // With dmin=0 the min term vanishes; qs=0 -> all outputs 0.
  std::vector<std::uint8_t> buf(144, 0);
  buf[0] = kHalf_1_0 & 0xFF;
  buf[1] = kHalf_1_0 >> 8;  // d = 1.0
  // dmin (buf[2..3]) = 0.0
  for (int i = 0; i < 12; ++i) buf[4 + i] = 1;  // scale/min 6-bit fields = 1 (min ignored, dmin=0)
  std::vector<float> out(256);
  REQUIRE(dequantize(static_cast<std::uint32_t>(GgmlType::Q4_K), buf.data(), out.data(), 256));
  for (int i = 0; i < 256; ++i) REQUIRE_THAT(out[i], WithinAbs(0.0f, 1e-4f));

  // Set first qs byte low nibble to 5 -> element 0 = d*scale(1)*5 - dmin*..(0) = 5.
  buf[16] = 0x05;
  dequantize(static_cast<std::uint32_t>(GgmlType::Q4_K), buf.data(), out.data(), 256);
  REQUIRE_THAT(out[0], WithinAbs(5.0f, 1e-4f));
}

TEST_CASE("canDequantize matches the supported set", "[dequant]") {
  REQUIRE(canDequantize(static_cast<std::uint32_t>(GgmlType::Q4_K)));
  REQUIRE(canDequantize(static_cast<std::uint32_t>(GgmlType::F16)));
  REQUIRE_FALSE(canDequantize(static_cast<std::uint32_t>(GgmlType::Q2_K)));
  REQUIRE_FALSE(canDequantize(static_cast<std::uint32_t>(GgmlType::IQ2_XXS)));
}
