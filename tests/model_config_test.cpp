#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
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
  // 'bert' used to be the example here, before Phase 11a added the encoder family. Vision and
  // audio architectures still have no loader, so they are the honest examples now.
  const auto file = GgufFile::parse(configModel("whisper"));
  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE_FALSE(cfg.valid());
  REQUIRE_FALSE(err.empty());
}

TEST_CASE("a llama gguf is still a decoder with decoder defaults", "[model_config]") {
  // Regression guard on widening the allowlist: none of the encoder fields may change behaviour
  // for a decoder, or every model that worked before Phase 11a would quietly shift.
  const auto file = GgufFile::parse(configModel("llama"));
  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE(cfg.valid());
  REQUIRE(cfg.family == ArchFamily::Decoder);
  REQUIRE_FALSE(cfg.isEncoder());
  REQUIRE(cfg.causal);
  REQUIRE(cfg.pooling == PoolingType::None);
  REQUIRE(cfg.ffnGated);
  REQUIRE_FALSE(cfg.attnBias);
  REQUIRE_FALSE(cfg.postNorm);
  REQUIRE_FALSE(cfg.hasPositionEmbd);
  REQUIRE(cfg.tokenTypeCount == 0);
  REQUIRE(cfg.normEpsilon == 1e-6f);  // still read from the rms key
}

namespace {
// Mirrors the metadata of a real bge-small-en-v1.5 GGUF (verified against the file on disk),
// scaled down. The tensors are registered metadata-only — configFromGguf probes tensor PRESENCE
// for the position table / gate / bias, never their contents, so no data section is needed.
std::vector<std::byte> encoderModel(std::uint32_t poolingType = 2, bool withBias = true) {
  GgufBuilder b(3);
  b.str("general.architecture", "bert")
      .u32("bert.embedding_length", 8)
      .u32("bert.block_count", 2)
      .u32("bert.feed_forward_length", 16)
      .u32("bert.context_length", 128)
      .u32("bert.attention.head_count", 4)
      .boolean("bert.attention.causal", false)
      .u32("bert.pooling_type", poolingType)
      .f32("bert.attention.layer_norm_epsilon", 1e-12f)
      .u32("tokenizer.ggml.token_type_count", 2)
      .stringArray("tokenizer.ggml.tokens", {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hi"});
  b.tensor("position_embd.weight", {8, 128}, 0, 0);
  if (withBias) b.tensor("blk.0.attn_q.bias", {8}, 0, 4096);
  return b.build(32, 8192);
}
}  // namespace

TEST_CASE("a bert gguf yields a bidirectional encoder config with cls pooling", "[model_config]") {
  const auto file = GgufFile::parse(encoderModel(/*poolingType=*/2));
  std::string err;
  const auto cfg = configFromGguf(file, err);

  REQUIRE(err.empty());
  REQUIRE(cfg.valid());
  REQUIRE(cfg.family == ArchFamily::Encoder);
  REQUIRE(cfg.isEncoder());
  REQUIRE_FALSE(cfg.causal);
  REQUIRE(cfg.pooling == PoolingType::Cls);
  REQUIRE(cfg.tokenTypeCount == 2);
  REQUIRE(cfg.postNorm);
  REQUIRE(cfg.hasPositionEmbd);
  REQUIRE(cfg.attnBias);
  REQUIRE_FALSE(cfg.ffnGated);  // no blk.0.ffn_gate.weight tensor
  REQUIRE(cfg.headCountKv == 4);  // MHA: no head_count_kv key
}

TEST_CASE("pooling_type 1 selects mean pooling", "[model_config]") {
  // bge-small ships pooling_type 2 and all-MiniLM ships 1; reading the key rather than hardcoding
  // per-architecture is what makes both work from the same loader.
  const auto file = GgufFile::parse(encoderModel(/*poolingType=*/1));
  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE(cfg.valid());
  REQUIRE(cfg.pooling == PoolingType::Mean);
}

TEST_CASE("the encoder reads attention.layer_norm_epsilon, not the rms key", "[model_config]") {
  // BERT trains at 1e-12 and writes layer_norm_epsilon; reading only the Llama-family rms key
  // left encoders silently on the 1e-5 default, which shifts every vector without failing.
  const auto file = GgufFile::parse(encoderModel());
  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE(cfg.valid());
  REQUIRE(cfg.normEpsilon == 1e-12f);
}

TEST_CASE("an encoder without attention bias tensors reports attnBias false", "[model_config]") {
  const auto file = GgufFile::parse(encoderModel(/*poolingType=*/2, /*withBias=*/false));
  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE(cfg.valid());
  REQUIRE_FALSE(cfg.attnBias);
}
