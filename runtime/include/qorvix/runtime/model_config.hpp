#pragma once

#include <cstdint>
#include <string>

#include "qorvix/runtime/ops.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::runtime {

// Which block layout an architecture uses. Decoder = Llama-style (pre-norm RMSNorm, SwiGLU, rope,
// causal attention, an LM head). Encoder = BERT-style (post-norm LayerNorm with bias, GELU FFN,
// bidirectional attention, no LM head — the output is a pooled vector, not vocab logits).
enum class ArchFamily { Decoder, Encoder };

// Pooling mode for encoder models, matching GGUF's `<arch>.pooling_type` values.
enum class PoolingType : std::uint32_t { None = 0, Mean = 1, Cls = 2, Last = 3 };

// Hyperparameters for a transformer, derived from GGUF metadata via the architecture-prefixed
// keys. The shared fields cover both families; the encoder-specific block below defaults to
// values that leave every decoder path byte-identical to before encoders existed.
struct ModelConfig {
  std::string architecture;
  std::uint32_t vocabSize = 0;
  std::uint32_t contextLength = 0;
  std::uint32_t embeddingLength = 0;   // d_model
  std::uint32_t blockCount = 0;        // number of transformer layers
  std::uint32_t feedForwardLength = 0; // FFN hidden size
  std::uint32_t headCount = 0;         // query heads
  std::uint32_t headCountKv = 0;       // key/value heads (== headCount for MHA, < for GQA/MQA)
  std::uint32_t ropeDimensionCount = 0; // dims per head that rope rotates (default: headDim)
  float ropeFreqBase = 10000.0f;
  float normEpsilon = 1e-5f;
  ops::RopeMode ropeMode = ops::RopeMode::Neox;

  // ---- family + encoder fields ----
  ArchFamily family = ArchFamily::Decoder;
  bool causal = true;                          // false for BERT (bidirectional attention)
  PoolingType pooling = PoolingType::None;     // how token states collapse to one vector
  std::uint32_t tokenTypeCount = 0;            // BERT segment embeddings (2), 0 if absent
  bool hasPositionEmbd = false;                // learned position table vs rope
  bool ffnGated = true;                        // SwiGLU gate+up vs a single GELU FFN
  bool attnBias = false;                       // q/k/v/o carry .bias tensors
  bool postNorm = false;                       // LayerNorm after sublayer+residual, not before

  bool isEncoder() const { return family == ArchFamily::Encoder; }
  std::uint32_t headDim() const { return headCount ? embeddingLength / headCount : 0; }
  std::uint32_t kvDim() const { return headCountKv * headDim(); }
  bool valid() const {
    return vocabSize && embeddingLength && blockCount && headCount && headCountKv &&
           feedForwardLength && headCount % headCountKv == 0 &&
           embeddingLength % headCount == 0;
  }
};

// Human-readable pooling name, for `qorvix model-info` and the --pooling CLI flag.
const char* poolingName(PoolingType p);
bool parsePooling(const std::string& s, PoolingType& out);

// Builds a ModelConfig from a parsed GGUF file. Reads "<arch>.*" metadata keys with sensible
// fallbacks. `error` is set (and the result is !valid()) when a required key is missing or the
// architecture isn't a supported decoder family.
ModelConfig configFromGguf(const gguf::GgufFile& file, std::string& error);

}  // namespace qorvix::runtime
