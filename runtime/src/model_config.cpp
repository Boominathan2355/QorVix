#include "qorvix/runtime/model_config.hpp"

#include <array>

#include "qorvix/gguf/gguf_file.hpp"

namespace qorvix::runtime {

namespace {

struct ArchEntry {
  const char* name;
  ArchFamily family;
};

// The block layouts this loader handles. Decoder = Llama-style (pre-norm RMSNorm, SwiGLU, rope,
// causal). Encoder = BERT-style (post-norm LayerNorm with bias, GELU FFN, bidirectional). Others
// (vision/audio/diffusion architectures) are rejected here rather than failing deep in the weight
// loader on a missing tensor.
constexpr std::array<ArchEntry, 8> kSupportedArch = {{
    {"llama", ArchFamily::Decoder},
    {"qwen2", ArchFamily::Decoder},
    {"qwen2moe", ArchFamily::Decoder},
    {"mistral", ArchFamily::Decoder},
    {"gemma", ArchFamily::Decoder},
    {"phi3", ArchFamily::Decoder},
    {"bert", ArchFamily::Encoder},
    {"nomic-bert", ArchFamily::Encoder},
}};

const ArchEntry* findArch(const std::string& arch) {
  for (const ArchEntry& e : kSupportedArch) {
    if (arch == e.name) return &e;
  }
  return nullptr;
}

}  // namespace

const char* poolingName(PoolingType p) {
  switch (p) {
    case PoolingType::None: return "none";
    case PoolingType::Mean: return "mean";
    case PoolingType::Cls: return "cls";
    case PoolingType::Last: return "last";
  }
  return "none";
}

bool parsePooling(const std::string& s, PoolingType& out) {
  if (s == "none") out = PoolingType::None;
  else if (s == "mean") out = PoolingType::Mean;
  else if (s == "cls") out = PoolingType::Cls;
  else if (s == "last") out = PoolingType::Last;
  else return false;
  return true;
}

ModelConfig configFromGguf(const gguf::GgufFile& file, std::string& error) {
  error.clear();
  ModelConfig cfg;
  cfg.architecture = file.architecture();
  if (cfg.architecture.empty()) {
    error = "GGUF has no general.architecture";
    return cfg;
  }
  const ArchEntry* entry = findArch(cfg.architecture);
  if (!entry) {
    error = "architecture '" + cfg.architecture + "' is not a supported model family yet";
    return cfg;
  }
  cfg.family = entry->family;

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

  // Llama-family GGUFs write ...rms_epsilon; BERT writes ...layer_norm_epsilon. No file writes
  // both. Reading only the rms key silently left every encoder on the 1e-5 default when the model
  // was trained at 1e-12 — a difference that shifts every vector without failing anything.
  if (auto v = file.getF64(a + "attention.layer_norm_rms_epsilon")) {
    cfg.normEpsilon = static_cast<float>(*v);
  } else if (auto v2 = file.getF64(a + "attention.layer_norm_epsilon")) {
    cfg.normEpsilon = static_cast<float>(*v2);
  }

  cfg.causal = file.getBool(a + "attention.causal").value_or(!cfg.isEncoder());
  if (auto v = file.getU64(a + "pooling_type")) {
    cfg.pooling = static_cast<PoolingType>(static_cast<std::uint32_t>(*v));
  } else if (cfg.isEncoder()) {
    cfg.pooling = PoolingType::Mean;  // the safe default when a conversion omits the key
  }
  // Note the prefix: this key is tokenizer-scoped, not architecture-scoped, in real files.
  if (auto v = file.getU64("tokenizer.ggml.token_type_count")) {
    cfg.tokenTypeCount = static_cast<std::uint32_t>(*v);
  }

  // Derived from tensor presence rather than metadata: llama.cpp writes no key for any of these,
  // and the tensor is the ground truth anyway. configFromGguf already has the whole file.
  cfg.hasPositionEmbd = file.tensor("position_embd.weight") != nullptr;
  if (cfg.isEncoder()) {
    cfg.ffnGated = file.tensor("blk.0.ffn_gate.weight") != nullptr;
    cfg.attnBias = file.tensor("blk.0.attn_q.bias") != nullptr;
    cfg.postNorm = true;
  }

  if (!cfg.valid() && error.empty()) {
    error = "GGUF metadata is missing required hyperparameters for '" + cfg.architecture + "'";
  }
  return cfg;
}

}  // namespace qorvix::runtime
