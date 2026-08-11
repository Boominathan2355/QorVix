#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gguf_builder.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/encoder_weights.hpp"
#include "qorvix/runtime/model_config.hpp"

namespace fs = std::filesystem;
using qorvix::gguf::GgufFile;
using qorvix::gguf::test::GgufBuilder;
using namespace qorvix::runtime;

namespace {

// These tests go through a real file on disk rather than GgufFile::parse(span), which every other
// GGUF test uses. That is not a style choice: parse() never populates the GgufFile's mapping_, and
// the weight loaders resolve tensors through mapping().bytes() — so an in-memory GGUF fails with
// "data is out of range (file not opened via mmap?)" before any loader logic runs. That is also
// why GgufBuilder::tensorF32 has existed unused since it was written; this is its first caller.
fs::path writeGguf(const std::string& name, const std::vector<std::byte>& bytes) {
  const fs::path dir = fs::temp_directory_path() / "qorvix_encoder_weights_test";
  fs::create_directories(dir);
  const fs::path path = dir / (name + ".gguf");
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  return path;
}

// A miniature BERT: 2 layers, d=4, 2 heads, ffn=8, vocab=6, ctx=8. Shapes mirror a real
// bge-small GGUF exactly (verified against the file on disk) — only the magnitudes shrink.
struct TinyBert {
  static constexpr int kD = 4, kFfn = 8, kVocab = 6, kCtx = 8, kLayers = 2, kHeads = 2;

  // `omit` drops one tensor by name so the negative paths can name what went missing.
  static std::vector<std::byte> bytes(const std::string& omit = "", bool withBias = true,
                                      bool withTokenTypes = true) {
    GgufBuilder b(3);
    b.str("general.architecture", "bert")
        .u32("bert.embedding_length", kD)
        .u32("bert.block_count", kLayers)
        .u32("bert.feed_forward_length", kFfn)
        .u32("bert.context_length", kCtx)
        .u32("bert.attention.head_count", kHeads)
        .boolean("bert.attention.causal", false)
        .u32("bert.pooling_type", 2)
        .f32("bert.attention.layer_norm_epsilon", 1e-12f)
        .stringArray("tokenizer.ggml.tokens", {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "b"});
    if (withTokenTypes) b.u32("tokenizer.ggml.token_type_count", 2);

    auto mat = [&](const std::string& name, int rows, int cols) {
      if (name == omit) return;
      // GGUF dims are in ggml order (dimensions[0] fastest), i.e. {cols, rows} for a row-major
      // [rows, cols] matrix — the layout the real files use.
      b.tensorF32(name, {static_cast<std::uint64_t>(cols), static_cast<std::uint64_t>(rows)},
                  std::vector<float>(static_cast<std::size_t>(rows) * cols, 0.25f));
    };
    auto vec = [&](const std::string& name, int n) {
      if (name == omit) return;
      b.tensorF32(name, {static_cast<std::uint64_t>(n)}, std::vector<float>(n, 0.5f));
    };

    mat("token_embd.weight", kVocab, kD);
    vec("token_embd_norm.weight", kD);
    vec("token_embd_norm.bias", kD);
    if (withTokenTypes) mat("token_types.weight", 2, kD);
    mat("position_embd.weight", kCtx, kD);

    for (int i = 0; i < kLayers; ++i) {
      const std::string p = "blk." + std::to_string(i) + ".";
      mat(p + "attn_q.weight", kD, kD);
      mat(p + "attn_k.weight", kD, kD);
      mat(p + "attn_v.weight", kD, kD);
      mat(p + "attn_output.weight", kD, kD);
      if (withBias) {
        vec(p + "attn_q.bias", kD);
        vec(p + "attn_k.bias", kD);
        vec(p + "attn_v.bias", kD);
        vec(p + "attn_output.bias", kD);
      }
      vec(p + "attn_output_norm.weight", kD);
      vec(p + "attn_output_norm.bias", kD);
      mat(p + "ffn_up.weight", kFfn, kD);
      vec(p + "ffn_up.bias", kFfn);
      mat(p + "ffn_down.weight", kD, kFfn);
      vec(p + "ffn_down.bias", kD);
      vec(p + "layer_output_norm.weight", kD);
      vec(p + "layer_output_norm.bias", kD);
    }
    return b.build();
  }
};

}  // namespace

TEST_CASE("a synthetic bert gguf loads encoder weights with biases", "[embeddings]") {
  const fs::path path = writeGguf("tiny_bert", TinyBert::bytes());
  const auto file = GgufFile::open(path);

  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE(err.empty());
  REQUIRE(cfg.isEncoder());
  REQUIRE(cfg.attnBias);
  REQUIRE(cfg.hasPositionEmbd);

  const auto w = loadEncoderWeights(file, cfg, err);
  REQUIRE(w.has_value());
  REQUIRE(err.empty());

  REQUIRE(w->layers.size() == TinyBert::kLayers);
  REQUIRE(w->tokenEmbd.rows == TinyBert::kVocab);
  REQUIRE(w->tokenEmbd.cols == TinyBert::kD);
  REQUIRE(w->positionEmbd.rows == TinyBert::kCtx);
  REQUIRE(w->tokenTypes.rows == 2);
  REQUIRE(w->embdNorm.size() == TinyBert::kD);
  REQUIRE(w->embdNormB.size() == TinyBert::kD);

  // Every layer, not just the first — a loop that fills layer 0 and leaves the rest empty would
  // otherwise pass.
  for (const auto& L : w->layers) {
    REQUIRE(L.wq.valid());
    REQUIRE(L.wo.valid());
    REQUIRE(L.bq.size() == TinyBert::kD);
    REQUIRE(L.bo.size() == TinyBert::kD);
    REQUIRE(L.attnNorm.size() == TinyBert::kD);
    REQUIRE(L.attnNormB.size() == TinyBert::kD);
    REQUIRE(L.ffnUp.rows == TinyBert::kFfn);
    REQUIRE(L.ffnUpB.size() == TinyBert::kFfn);
    REQUIRE(L.ffnDown.rows == TinyBert::kD);
    REQUIRE(L.ffnDownB.size() == TinyBert::kD);
    REQUIRE(L.ffnNorm.size() == TinyBert::kD);
    REQUIRE_FALSE(L.ffnGate.valid());  // plain BERT has no gate
  }
}

TEST_CASE("a missing encoder tensor names itself in the error", "[embeddings]") {
  // layer_output_norm is the post-FFN norm — the one tensor a loader written from the decoder's
  // shape would not think to look for at all.
  const fs::path path = writeGguf("tiny_bert_missing", TinyBert::bytes("blk.1.layer_output_norm.weight"));
  const auto file = GgufFile::open(path);

  std::string err;
  const auto cfg = configFromGguf(file, err);
  const auto w = loadEncoderWeights(file, cfg, err);
  REQUIRE_FALSE(w.has_value());
  REQUIRE(err.find("blk.1.layer_output_norm.weight") != std::string::npos);
}

TEST_CASE("an encoder without token types loads with an invalid token type matrix",
          "[embeddings]") {
  const fs::path path =
      writeGguf("tiny_bert_no_types", TinyBert::bytes("", /*withBias=*/true, /*withTokenTypes=*/false));
  const auto file = GgufFile::open(path);

  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE(cfg.tokenTypeCount == 0);

  const auto w = loadEncoderWeights(file, cfg, err);
  REQUIRE(w.has_value());
  REQUIRE_FALSE(w->tokenTypes.valid());  // absent, not zero-filled
}

TEST_CASE("an encoder without attention biases loads with empty bias vectors", "[embeddings]") {
  const fs::path path = writeGguf("tiny_bert_no_bias", TinyBert::bytes("", /*withBias=*/false));
  const auto file = GgufFile::open(path);

  std::string err;
  const auto cfg = configFromGguf(file, err);
  REQUIRE_FALSE(cfg.attnBias);

  const auto w = loadEncoderWeights(file, cfg, err);
  REQUIRE(w.has_value());
  REQUIRE(w->layers.front().bq.empty());
  // wmatmulBias treats an empty bias as "no bias", so this stays a valid model rather than a
  // half-loaded one.
  REQUIRE(w->layers.front().wq.valid());
}

TEST_CASE("loadEncoderWeights refuses a decoder config", "[embeddings]") {
  // The mirror of the encoder guard in TextModel::fromGguf: neither loader may quietly accept the
  // other family's model and fail later on a tensor name.
  const fs::path path = writeGguf("tiny_bert_for_decoder", TinyBert::bytes());
  const auto file = GgufFile::open(path);

  std::string err;
  auto cfg = configFromGguf(file, err);
  cfg.family = ArchFamily::Decoder;  // pretend
  const auto w = loadEncoderWeights(file, cfg, err);
  REQUIRE_FALSE(w.has_value());
  REQUIRE(err.find("decoder config") != std::string::npos);
}
