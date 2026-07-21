#include "qorvix/runtime/model_config.hpp"

#include <array>

#include "qorvix/gguf/gguf_file.hpp"

namespace qorvix::runtime {

namespace {

// Decoder families that share the Llama-style block layout this loader handles. Others (e.g.
// vision/audio/diffusion architectures) are rejected until their loaders exist.
constexpr std::array<const char*, 6> kSupportedArch = {"llama",   "qwen2",   "qwen2moe",
                                                       "mistral", "gemma",   "phi3"};

bool isSupported(const std::string& arch) {
  for (const char* a : kSupportedArch) {
    if (arch == a) return true;
  }
  return false;
}

}  // namespace

ModelConfig configFromGguf(const gguf::GgufFile& file, std::string& error) {
  error.clear();
  ModelConfig cfg;
  cfg.architecture = file.architecture();
  if (cfg.architecture.empty()) {
    error = "GGUF has no general.architecture";
    return cfg;
  }
  if (!isSupported(cfg.architecture)) {
    error = "architecture '" + cfg.architecture + "' is not a supported decoder family yet";
    return cfg;
  }

  const std::string a = cfg.architecture + ".";
  auto u32 = [&](const std::string& suffix, std::uint32_t fallback) -> std::uint32_t {
    if (auto v = file.getU64(a + suffix)) return static_cast<std::uint32_t>(*v);
    return fallback;
  };

  // Vocab size: prefer the tokenizer token list length, fall back to the embedding row count
  // (filled by the weights loader if still zero).
  if (const auto* tokens = file.find("tokenizer.ggml.tokens"); tokens && tokens->isArray()) {
    cfg.vocabSize = static_cast<std::uint32_t>(tokens->array().size());
  }
  if (auto v = file.getU64(a + "vocab_size")) cfg.vocabSize = static_cast<std::uint32_t>(*v);

  cfg.contextLength = u32("context_length", 0);
  cfg.embeddingLength = u32("embedding_length", 0);
  cfg.blockCount = u32("block_count", 0);
  cfg.feedForwardLength = u32("feed_forward_length", 0);
  cfg.headCount = u32("attention.head_count", 0);
  cfg.headCountKv = u32("attention.head_count_kv", cfg.headCount);  // MHA if unspecified

  cfg.ropeDimensionCount = u32("rope.dimension_count", cfg.headDim());
  if (auto v = file.getF64(a + "rope.freq_base")) cfg.ropeFreqBase = static_cast<float>(*v);
  if (auto v = file.getF64(a + "attention.layer_norm_rms_epsilon"))
    cfg.normEpsilon = static_cast<float>(*v);

  if (!cfg.valid() && error.empty()) {
    error = "GGUF metadata is missing required hyperparameters for '" + cfg.architecture + "'";
  }
  return cfg;
}

}  // namespace qorvix::runtime
