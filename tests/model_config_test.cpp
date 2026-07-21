#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <vector>

#include "gguf_builder.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/model_config.hpp"

using qorvix::gguf::GgufFile;
using qorvix::gguf::test::GgufBuilder;
using namespace qorvix::runtime;

namespace {
std::vector<std::byte> configModel(const std::string& arch = "llama") {
  GgufBuilder b(3);
  b.str("general.architecture", arch)
      .u32(arch + ".embedding_length", 8)
      .u32(arch + ".block_count", 2)
      .u32(arch + ".feed_forward_length", 16)
      .u32(arch + ".context_length", 128)
      .u32(arch + ".attention.head_count", 4)
      .u32(arch + ".attention.head_count_kv", 2)
      .f32(arch + ".rope.freq_base", 500000.0f)
      .f32(arch + ".attention.layer_norm_rms_epsilon", 1e-6f)
      .stringArray("tokenizer.ggml.tokens", {"a", "b", "c", "d", "e"});
  return b.build();
}
}  // namespace

TEST_CASE("configFromGguf derives Llama hyperparameters", "[model_config]") {
  const auto bytes = configModel();
  const auto file = GgufFile::parse(bytes);
  std::string err;
  const auto cfg = configFromGguf(file, err);

  REQUIRE(err.empty());
  REQUIRE(cfg.valid());
  REQUIRE(cfg.architecture == "llama");
  REQUIRE(cfg.vocabSize == 5);
  REQUIRE(cfg.embeddingLength == 8);
  REQUIRE(cfg.blockCount == 2);
  REQUIRE(cfg.feedForwardLength == 16);
  REQUIRE(cfg.headCount == 4);
  REQUIRE(cfg.headCountKv == 2);
  REQUIRE(cfg.headDim() == 2);
  REQUIRE(cfg.kvDim() == 4);
  REQUIRE(cfg.contextLength == 128);
  REQUIRE(cfg.ropeFreqBase == 500000.0f);
}

TEST_CASE("head_count_kv defaults to head_count (MHA) when absent", "[model_config]") {
  GgufBuilder b(3);
  b.str("general.architecture", "llama")
      .u32("llama.embedding_length", 8)
      .u32("llama.block_count", 1)
      .u32("llama.feed_forward_length", 16)
      .u32("llama.attention.head_count", 4)
      .stringArray("tokenizer.ggml.tokens", {"a", "b"});
  const auto file = GgufFile::parse(b.build());
  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE(cfg.valid());
  REQUIRE(cfg.headCountKv == 4);
}

TEST_CASE("unsupported architectures are rejected", "[model_config]") {
  const auto file = GgufFile::parse(configModel("bert"));
  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE_FALSE(cfg.valid());
  REQUIRE_FALSE(err.empty());
}
