#pragma once

#include <optional>
#include <string>
#include <vector>

#include "qorvix/runtime/model_config.hpp"
#include "qorvix/runtime/weights.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::runtime {

// BERT-family per-layer weights.
//
// A separate struct from LayerWeights rather than eight more fields on it. A decoder layer is
// pre-norm RMSNorm + SwiGLU with no bias anywhere; an encoder layer is post-norm LayerNorm + GELU
// with bias on every projection. Widening LayerWeights would put eight permanently-empty vectors
// in every decoder layer and blur what a decoder layer is — and the two families already have
// separate model classes, so separate weight structs cost nothing.
//
// Tensor names below are the llama.cpp BERT convention, verified against real
// bge-small-en-v1.5 and all-MiniLM-L6-v2 GGUFs rather than assumed.
struct EncoderLayerWeights {
  // Attention projections, all [d, d] for MHA. Biases are empty when cfg.attnBias is false.
  WeightMat wq, wk, wv, wo;                  // blk.N.attn_{q,k,v,output}.weight
  std::vector<float> bq, bk, bv, bo;         // blk.N.attn_{q,k,v,output}.bias
  // Post-attention LayerNorm, applied AFTER the residual add.
  std::vector<float> attnNorm, attnNormB;    // blk.N.attn_output_norm.{weight,bias}

  WeightMat ffnUp;                           // blk.N.ffn_up.weight     [ffn, d]
  std::vector<float> ffnUpB;                 // blk.N.ffn_up.bias       [ffn]
  WeightMat ffnGate;                         // blk.N.ffn_gate.weight — invalid unless cfg.ffnGated
  WeightMat ffnDown;                         // blk.N.ffn_down.weight   [d, ffn]
  std::vector<float> ffnDownB;               // blk.N.ffn_down.bias     [d]
  // Post-FFN LayerNorm, likewise after the residual add.
  std::vector<float> ffnNorm, ffnNormB;      // blk.N.layer_output_norm.{weight,bias}
};

struct EncoderWeights {
  WeightMat tokenEmbd;                       // token_embd.weight     [vocab, d]
  WeightMat tokenTypes;                      // token_types.weight    [tokenTypeCount, d]
  WeightMat positionEmbd;                    // position_embd.weight  [ctx, d]; invalid => rope
  std::vector<float> embdNorm, embdNormB;    // token_embd_norm.{weight,bias}  [d]
  std::vector<EncoderLayerWeights> layers;
};

// Loads BERT-family weights from an opened (mmap'd) GGUF, keeping matmul weights quantized and
// borrowing the file's mapping — the file must outlive the result, which BertModel guarantees.
// Requires cfg.isEncoder(). Returns nullopt with `error` naming the offending tensor.
//
// Deliberately NOT loaded: output_norm.*, output.weight, and the cls.* MLM/NSP heads. Some
// conversions carry them; an embedding model never uses them.
std::optional<EncoderWeights> loadEncoderWeights(const gguf::GgufFile& file, const ModelConfig& cfg,
                                                 std::string& error);

}  // namespace qorvix::runtime
