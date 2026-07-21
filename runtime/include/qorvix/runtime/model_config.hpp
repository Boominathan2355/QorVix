#pragma once

#include <cstdint>
#include <string>

#include "qorvix/runtime/ops.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::runtime {

// Hyperparameters for a decoder-only transformer (Llama-family: llama, qwen2, mistral, gemma,
// ...). Derived from GGUF metadata via the architecture-prefixed keys. Fields common to that
// family; architecture quirks (partial rope, attention bias, logit softcap) are added as the
// loader grows to support more models.
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

  std::uint32_t headDim() const { return headCount ? embeddingLength / headCount : 0; }
  std::uint32_t kvDim() const { return headCountKv * headDim(); }
  bool valid() const {
    return vocabSize && embeddingLength && blockCount && headCount && headCountKv &&
           feedForwardLength && headCount % headCountKv == 0 &&
           embeddingLength % headCount == 0;
  }
};

// Builds a ModelConfig from a parsed GGUF file. Reads "<arch>.*" metadata keys with sensible
// fallbacks. `error` is set (and the result is !valid()) when a required key is missing or the
// architecture isn't a supported decoder family.
ModelConfig configFromGguf(const gguf::GgufFile& file, std::string& error);

}  // namespace qorvix::runtime
