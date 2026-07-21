#include "qorvix/runtime/weights.hpp"

#include <cstddef>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/dequant.hpp"

namespace qorvix::runtime {

namespace {

// Dequantizes one named tensor into `out`. Returns false (and sets error) if the tensor is
// absent, has an unexpected element count, or uses a quant we can't dequantize.
bool loadTensor(const gguf::GgufFile& file, const std::string& name, std::size_t expectElements,
                std::vector<float>& out, std::string& error) {
  const gguf::GgufTensor* t = file.tensor(name);
  if (!t) {
    error = "missing tensor '" + name + "'";
    return false;
  }
  if (expectElements != 0 && t->nElements != expectElements) {
    error = "tensor '" + name + "' has " + std::to_string(t->nElements) + " elements, expected " +
            std::to_string(expectElements);
    return false;
  }
  if (!canDequantize(t->typeRaw)) {
    error = "tensor '" + name + "' uses unsupported type " + t->typeName();
    return false;
  }

  const auto all = file.mapping().bytes();
  const std::uint64_t start = file.dataOffset() + t->offset;
  if (all.empty() || start + t->nBytes > all.size()) {
    error = "tensor '" + name + "' data is out of range (file not opened via mmap?)";
    return false;
  }

  out.resize(t->nElements);
  if (!dequantize(t->typeRaw, all.data() + start, out.data(), t->nElements)) {
    error = "failed to dequantize '" + name + "'";
    return false;
  }
  return true;
}

std::string blk(int i, const char* suffix) {
  return "blk." + std::to_string(i) + "." + suffix;
}

}  // namespace

std::optional<Weights> loadWeights(const gguf::GgufFile& file, const ModelConfig& cfg,
                                   std::string& error) {
  error.clear();
  Weights w;

  const std::size_t d = cfg.embeddingLength;
  const std::size_t kv = cfg.kvDim();
  const std::size_t ffn = cfg.feedForwardLength;
  const std::size_t vocab = cfg.vocabSize;

  if (!loadTensor(file, "token_embd.weight", vocab * d, w.tokenEmbd, error)) return std::nullopt;
  if (!loadTensor(file, "output_norm.weight", d, w.outputNorm, error)) return std::nullopt;

  // Optional LM head; tied to the embedding table when absent.
  if (file.tensor("output.weight")) {
    if (!loadTensor(file, "output.weight", vocab * d, w.output, error)) return std::nullopt;
  }

  w.layers.resize(cfg.blockCount);
  for (std::uint32_t i = 0; i < cfg.blockCount; ++i) {
    LayerWeights& L = w.layers[i];
    if (!loadTensor(file, blk(i, "attn_norm.weight"), d, L.attnNorm, error) ||
        !loadTensor(file, blk(i, "attn_q.weight"), d * d, L.wq, error) ||
        !loadTensor(file, blk(i, "attn_k.weight"), kv * d, L.wk, error) ||
        !loadTensor(file, blk(i, "attn_v.weight"), kv * d, L.wv, error) ||
        !loadTensor(file, blk(i, "attn_output.weight"), d * d, L.wo, error) ||
        !loadTensor(file, blk(i, "ffn_norm.weight"), d, L.ffnNorm, error) ||
        !loadTensor(file, blk(i, "ffn_gate.weight"), ffn * d, L.ffnGate, error) ||
        !loadTensor(file, blk(i, "ffn_up.weight"), ffn * d, L.ffnUp, error) ||
        !loadTensor(file, blk(i, "ffn_down.weight"), d * ffn, L.ffnDown, error)) {
      return std::nullopt;
    }
  }

  return w;
}

}  // namespace qorvix::runtime
