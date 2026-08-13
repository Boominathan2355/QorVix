#include "qorvix/runtime/weights.hpp"

#include <cstddef>
#include <cstdint>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/dequant.hpp"

#include "qorvix/runtime/tensor_load.hpp"

namespace qorvix::runtime {

using detail::blk;
using detail::loadMat;
using detail::loadVec;

std::optional<Weights> loadWeights(const gguf::GgufFile& file, const ModelConfig& cfg,
                                   std::string& error) {
  error.clear();
  Weights w;

  const int d = static_cast<int>(cfg.embeddingLength);
  const int kv = static_cast<int>(cfg.kvDim());
  const int ffn = static_cast<int>(cfg.feedForwardLength);
  const int vocab = static_cast<int>(cfg.vocabSize);

  if (!loadMat(file, "token_embd.weight", vocab, d, w.tokenEmbd, error)) return std::nullopt;
  if (!loadVec(file, "output_norm.weight", d, w.outputNorm, error)) return std::nullopt;
  if (file.tensor("output.weight")) {
    if (!loadMat(file, "output.weight", vocab, d, w.output, error)) return std::nullopt;
  }

  w.layers.resize(cfg.blockCount);
  for (std::uint32_t i = 0; i < cfg.blockCount; ++i) {
    LayerWeights& L = w.layers[i];
    if (!loadVec(file, blk(i, "attn_norm.weight"), d, L.attnNorm, error) ||
        !loadMat(file, blk(i, "attn_q.weight"), d, d, L.wq, error) ||
        !loadMat(file, blk(i, "attn_k.weight"), kv, d, L.wk, error) ||
        !loadMat(file, blk(i, "attn_v.weight"), kv, d, L.wv, error) ||
        !loadMat(file, blk(i, "attn_output.weight"), d, d, L.wo, error) ||
        !loadVec(file, blk(i, "ffn_norm.weight"), d, L.ffnNorm, error) ||
        !loadMat(file, blk(i, "ffn_gate.weight"), ffn, d, L.ffnGate, error) ||
        !loadMat(file, blk(i, "ffn_up.weight"), ffn, d, L.ffnUp, error) ||
        !loadMat(file, blk(i, "ffn_down.weight"), d, ffn, L.ffnDown, error)) {
      return std::nullopt;
    }
  }
  return w;
}

}  // namespace qorvix::runtime
