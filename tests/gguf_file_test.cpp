#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

#include "gguf_builder.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/gguf/gguf_reader.hpp"

using qorvix::gguf::GgufFile;
using qorvix::gguf::GgufParseError;
using qorvix::gguf::GgmlType;
using qorvix::gguf::test::GgufBuilder;

namespace {

// A representative small model: llama-ish metadata, an array, and two tensors (F32 + Q4_K).
std::vector<std::byte> sampleModel(std::uint32_t version = 3) {
  GgufBuilder b(version);
  b.str("general.architecture", "llama")
      .str("general.name", "tiny-test")
      .u32("general.alignment", 32)
      .u32("general.file_type", 15)
      .u64("llama.context_length", 4096)
      .u32("llama.attention.head_count", 32)
      .f32("llama.rope.freq_base", 10000.0f)
      .u32("llama.rope.dimension_count", 128)
      .boolean("tokenizer.ggml.add_bos_token", true)
      .stringArray("tokenizer.ggml.tokens", {"<s>", "</s>", "hello", "world"})
      .i32Array("tokenizer.ggml.token_type", {1, 1, 1, 1});
  // token_embd: [64, 4] F32 -> 256 elems * 4 bytes = 1024 bytes at offset 0
  b.tensor("token_embd.weight", {64, 4}, static_cast<std::uint32_t>(GgmlType::F32), 0);
  // ffn: [256] Q4_K -> 256 elems / 256 block * 144 = 144 bytes at offset 1024 (32-aligned)
  b.tensor("blk.0.ffn_down.weight", {256}, static_cast<std::uint32_t>(GgmlType::Q4_K), 1024);
  return b.build(/*alignment=*/32, /*dataBytes=*/1024 + 144);
}

}  // namespace

TEST_CASE("parses header, metadata, and typed accessors", "[gguf]") {
  const auto bytes = sampleModel();
  const auto f = GgufFile::parse(bytes);

  REQUIRE(f.header().version == 3);
  REQUIRE(f.header().metadataCount == 11);
  REQUIRE(f.header().tensorCount == 2);
  REQUIRE(f.architecture() == "llama");
  REQUIRE(f.name() == "tiny-test");
  REQUIRE(f.fileType() == 15);
  REQUIRE(f.alignment() == 32);

  REQUIRE(f.getU64("llama.context_length") == 4096);
  REQUIRE(f.archU64("attention.head_count") == 32);
  REQUIRE(f.getBool("tokenizer.ggml.add_bos_token") == true);
}

TEST_CASE("parses RoPE metadata via the arch prefix", "[gguf]") {
  const auto bytes = sampleModel();
  const auto f = GgufFile::parse(bytes);
  const auto rope = f.rope();
  REQUIRE(rope.dimensionCount == 128);
  REQUIRE(rope.freqBase.has_value());
  REQUIRE(*rope.freqBase == 10000.0);
}

TEST_CASE("parses arrays", "[gguf]") {
  const auto bytes = sampleModel();
  const auto f = GgufFile::parse(bytes);
  const auto* tokens = f.find("tokenizer.ggml.tokens");
  REQUIRE(tokens != nullptr);
  REQUIRE(tokens->isArray());
  REQUIRE(tokens->array().size() == 4);
  REQUIRE(*tokens->array()[2].asString() == "hello");
}

TEST_CASE("parses tensor table with computed sizes and data offset", "[gguf]") {
  const auto bytes = sampleModel();
  const auto f = GgufFile::parse(bytes);

  REQUIRE(f.tensors().size() == 2);
  const auto* embd = f.tensor("token_embd.weight");
  REQUIRE(embd != nullptr);
  REQUIRE(embd->dimensions == std::vector<std::uint64_t>{64, 4});
  REQUIRE(embd->nElements == 256);
  REQUIRE(embd->nBytes == 1024);
  REQUIRE(embd->typeName() == "F32");

  const auto* ffn = f.tensor("blk.0.ffn_down.weight");
  REQUIRE(ffn != nullptr);
  REQUIRE(ffn->nElements == 256);
  REQUIRE(ffn->nBytes == 144);  // 1 Q4_K block
  REQUIRE(ffn->offset == 1024);

  // Data section begins 32-aligned after the tensor table.
  REQUIRE(f.dataOffset() % 32 == 0);
  REQUIRE(f.dataOffset() < bytes.size());
}

TEST_CASE("v2 files parse identically", "[gguf]") {
  const auto f = GgufFile::parse(sampleModel(2));
  REQUIRE(f.header().version == 2);
  REQUIRE(f.architecture() == "llama");
  REQUIRE(f.tensors().size() == 2);
}

TEST_CASE("rejects a bad magic", "[gguf]") {
  std::vector<std::byte> bytes(16, std::byte{0});
  REQUIRE_THROWS_AS(GgufFile::parse(bytes), GgufParseError);
}

TEST_CASE("rejects an unsupported version", "[gguf]") {
  GgufBuilder b(99);
  REQUIRE_THROWS_AS(GgufFile::parse(b.build()), GgufParseError);
}

TEST_CASE("rejects truncation mid-metadata", "[gguf]") {
  auto bytes = sampleModel();
  bytes.resize(40);  // cut off partway through
  REQUIRE_THROWS_AS(GgufFile::parse(bytes), GgufParseError);
}

TEST_CASE("rejects a misaligned tensor offset", "[gguf]") {
  GgufBuilder b(3);
  b.str("general.architecture", "llama").u32("general.alignment", 32);
  b.tensor("bad.weight", {32}, static_cast<std::uint32_t>(GgmlType::F32), 7);  // 7 % 32 != 0
  REQUIRE_THROWS_AS(GgufFile::parse(b.build()), GgufParseError);
}

TEST_CASE("rejects a block-size mismatch", "[gguf]") {
  GgufBuilder b(3);
  b.str("general.architecture", "llama");
  // 100 is not a multiple of Q4_K's 256-element block.
  b.tensor("bad.weight", {100}, static_cast<std::uint32_t>(GgmlType::Q4_K), 0);
  REQUIRE_THROWS_AS(GgufFile::parse(b.build()), GgufParseError);
}
