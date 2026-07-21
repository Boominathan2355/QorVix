#pragma once

#include <optional>
#include <string>
#include <vector>

#include "qorvix/runtime/model_config.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::runtime {

// Per-layer weight matrices, all dequantized to F32 and row-major. Matmul convention:
// out = W * x with W shaped [out_features, in_features] (the GGUF storage order).
struct LayerWeights {
  std::vector<float> attnNorm;      // [d_model]
  std::vector<float> wq;            // [n_heads*head_dim, d_model]
  std::vector<float> wk;            // [n_kv*head_dim, d_model]
  std::vector<float> wv;            // [n_kv*head_dim, d_model]
  std::vector<float> wo;            // [d_model, n_heads*head_dim]
  std::vector<float> ffnNorm;       // [d_model]
  std::vector<float> ffnGate;       // [ffn, d_model]
  std::vector<float> ffnUp;         // [ffn, d_model]
  std::vector<float> ffnDown;       // [d_model, ffn]
};

// Full model weights. `output` is the LM head [vocab, d_model]; when a model ties embeddings
// (no output.weight tensor) it is left empty and callers use tokenEmbd instead.
struct Weights {
  std::vector<float> tokenEmbd;     // [vocab, d_model]
  std::vector<LayerWeights> layers; // [block_count]
  std::vector<float> outputNorm;    // [d_model]
  std::vector<float> output;        // [vocab, d_model] or empty (tied to tokenEmbd)

  const std::vector<float>& lmHead() const { return output.empty() ? tokenEmbd : output; }
};

// Loads and dequantizes every weight tensor named by the Llama-family GGUF convention
// (token_embd, blk.N.attn_*/ffn_*, output_norm, output). Requires a file opened via
// GgufFile::open (memory-mapped) so tensor data is reachable. Returns nullopt with `error` set
// on a missing/unsupported tensor.
std::optional<Weights> loadWeights(const gguf::GgufFile& file, const ModelConfig& cfg,
                                   std::string& error);

}  // namespace qorvix::runtime
