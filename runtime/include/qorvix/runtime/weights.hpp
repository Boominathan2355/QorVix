#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/qmatmul.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::runtime {

// A matmul weight matrix [rows, cols] (out_features x in_features), stored either as owned F32
// (tests / in-memory models) or as borrowed quantized GGUF bytes (real models — the pointer
// aliases the model's mmap, kept alive by TextModel). This is what lets weights stay ~4-bit in
// memory: the forward pass runs qmatmul directly on the quantized bytes.
struct WeightMat {
  int rows = 0;
  int cols = 0;
  std::uint32_t type = 0;                // GgmlType::F32 == 0
  const std::uint8_t* quant = nullptr;   // borrowed quantized bytes when set
  std::vector<float> owned;              // owned F32 when quant == nullptr

  static WeightMat f32(std::vector<float> data, int rows, int cols) {
    WeightMat w;
    w.rows = rows;
    w.cols = cols;
    w.owned = std::move(data);
    return w;
  }
  static WeightMat quantized(const std::uint8_t* p, std::uint32_t type, int rows, int cols) {
    WeightMat w;
    w.rows = rows;
    w.cols = cols;
    w.type = type;
    w.quant = p;
    return w;
  }

  bool valid() const { return quant != nullptr || !owned.empty(); }
};

// out[rows] = W * x[cols], dispatching to the quantized or F32 kernel.
inline void wmatmul(float* out, const WeightMat& w, const float* x) {
  if (w.quant) {
    qmatmul(out, w.quant, w.type, x, w.rows, w.cols);
  } else {
    ops::matmul(out, w.owned.data(), x, w.rows, w.cols);
  }
}

// Writes row `r` (cols elements) of an embedding table into dst.
inline void embeddingRow(const WeightMat& w, int r, float* dst) {
  if (w.quant) {
    dequantRow(w.quant, w.type, w.cols, r, dst);
  } else {
    const float* src = w.owned.data() + static_cast<std::size_t>(r) * w.cols;
    for (int i = 0; i < w.cols; ++i) dst[i] = src[i];
  }
}

// Per-layer weights: matmuls as WeightMat (quantized or F32), norms as owned F32 (small, F32 in
// GGUF, used elementwise in rmsnorm).
struct LayerWeights {
  std::vector<float> attnNorm;   // [d_model]
  WeightMat wq;                  // [n_heads*head_dim, d_model]
  WeightMat wk;                  // [n_kv*head_dim, d_model]
  WeightMat wv;                  // [n_kv*head_dim, d_model]
  WeightMat wo;                  // [d_model, n_heads*head_dim]
  std::vector<float> ffnNorm;    // [d_model]
  WeightMat ffnGate;             // [ffn, d_model]
  WeightMat ffnUp;               // [ffn, d_model]
  WeightMat ffnDown;             // [d_model, ffn]
};

struct Weights {
  WeightMat tokenEmbd;              // [vocab, d_model]
  std::vector<LayerWeights> layers;
  std::vector<float> outputNorm;   // [d_model]
  WeightMat output;                // [vocab, d_model]; invalid => tied to tokenEmbd

  const WeightMat& lmHead() const { return output.valid() ? output : tokenEmbd; }
};

// Loads weights from an opened (mmap'd) GGUF, keeping matmul weights quantized (borrowing the
// file's mapping — the file must outlive the returned Weights, which TextModel guarantees) and
// copying the small F32 norms. Returns nullopt with `error` set on a missing/unsupported tensor.
std::optional<Weights> loadWeights(const gguf::GgufFile& file, const ModelConfig& cfg,
                                   std::string& error);

}  // namespace qorvix::runtime
