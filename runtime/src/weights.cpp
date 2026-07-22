#include "qorvix/runtime/weights.hpp"

#include <cstddef>
#include <cstdint>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/dequant.hpp"

namespace qorvix::runtime {

namespace {

// Resolves a tensor's raw bytes inside the file's mmap. Sets error and returns nullptr if the
// tensor is missing, the element count is wrong, or the file wasn't memory-mapped.
const std::uint8_t* tensorBytes(const gguf::GgufFile& file, const std::string& name,
                                std::size_t expectElements, const gguf::GgufTensor** outT,
                                std::string& error) {
  const gguf::GgufTensor* t = file.tensor(name);
  if (!t) {
    error = "missing tensor '" + name + "'";
    return nullptr;
  }
  if (t->nElements != expectElements) {
    error = "tensor '" + name + "' has " + std::to_string(t->nElements) + " elements, expected " +
            std::to_string(expectElements);
    return nullptr;
  }
  const auto all = file.mapping().bytes();
  const std::uint64_t start = file.dataOffset() + t->offset;
  if (all.empty() || start + t->nBytes > all.size()) {
    error = "tensor '" + name + "' data is out of range (file not opened via mmap?)";
    return nullptr;
  }
  *outT = t;
  return reinterpret_cast<const std::uint8_t*>(all.data()) + start;
}

// Borrows a matmul weight [rows, cols] straight from the mmap, keeping it quantized.
bool loadMat(const gguf::GgufFile& file, const std::string& name, int rows, int cols,
             WeightMat& out, std::string& error) {
  const gguf::GgufTensor* t = nullptr;
  const std::uint8_t* ptr =
      tensorBytes(file, name, static_cast<std::size_t>(rows) * cols, &t, error);
  if (!ptr) return false;
  if (!qmatmulSupports(t->typeRaw)) {
    error = "tensor '" + name + "' uses unsupported type " + t->typeName();
    return false;
  }
  out = WeightMat::quantized(ptr, t->typeRaw, rows, cols);
  return true;
}

// Copies a small norm/bias vector, dequantizing to F32 (norms are F32 in practice).
bool loadVec(const gguf::GgufFile& file, const std::string& name, int n, std::vector<float>& out,
             std::string& error) {
  const gguf::GgufTensor* t = nullptr;
  const std::uint8_t* ptr = tensorBytes(file, name, static_cast<std::size_t>(n), &t, error);
  if (!ptr) return false;
  if (!canDequantize(t->typeRaw)) {
    error = "tensor '" + name + "' uses unsupported type " + t->typeName();
    return false;
  }
  out.resize(n);
  return dequantize(t->typeRaw, ptr, out.data(), static_cast<std::size_t>(n));
}

std::string blk(int i, const char* suffix) {
  return "blk." + std::to_string(i) + "." + suffix;
}

}  // namespace

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
