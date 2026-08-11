#include "qorvix/runtime/encoder_weights.hpp"

#include "qorvix/gguf/gguf_file.hpp"

#include "weights_detail.hpp"

namespace qorvix::runtime {

using detail::blk;
using detail::loadMat;
using detail::loadMatOpt;
using detail::loadVec;
using detail::loadVecOpt;

std::optional<EncoderWeights> loadEncoderWeights(const gguf::GgufFile& file, const ModelConfig& cfg,
                                                 std::string& error) {
  error.clear();
  if (!cfg.isEncoder()) {
    error = "loadEncoderWeights called with a decoder config ('" + cfg.architecture + "')";
    return std::nullopt;
  }
  if (!cfg.valid()) {
    error = "invalid encoder config";
    return std::nullopt;
  }

  const int d = static_cast<int>(cfg.embeddingLength);
  const int ffn = static_cast<int>(cfg.feedForwardLength);
  const int vocab = static_cast<int>(cfg.vocabSize);
  const int ctx = static_cast<int>(cfg.contextLength);

  EncoderWeights w;
  if (!loadMat(file, "token_embd.weight", vocab, d, w.tokenEmbd, error)) return std::nullopt;

  // The embedding LayerNorm. Present on every BERT conversion seen so far, but optional rather
  // than required: nomic-bert applies rope and skips it, and a missing norm is better handled as
  // "identity" than as a hard failure on a model whose weights are otherwise complete.
  if (!loadVecOpt(file, "token_embd_norm.weight", d, w.embdNorm, error)) return std::nullopt;
  if (!loadVecOpt(file, "token_embd_norm.bias", d, w.embdNormB, error)) return std::nullopt;

  if (cfg.tokenTypeCount > 0) {
    if (!loadMatOpt(file, "token_types.weight", static_cast<int>(cfg.tokenTypeCount), d,
                    w.tokenTypes, error)) {
      return std::nullopt;
    }
  }
  if (cfg.hasPositionEmbd) {
    // Required, not optional, when the config said the tensor is there — cfg derived that flag
    // from the tensor's presence, so failing here means the file changed underneath us.
    if (!loadMat(file, "position_embd.weight", ctx, d, w.positionEmbd, error)) return std::nullopt;
  }

  w.layers.resize(cfg.blockCount);
  for (int i = 0; i < static_cast<int>(cfg.blockCount); ++i) {
    EncoderLayerWeights& L = w.layers[i];
    const bool ok =
        loadMat(file, blk(i, "attn_q.weight"), d, d, L.wq, error) &&
        loadMat(file, blk(i, "attn_k.weight"), d, d, L.wk, error) &&
        loadMat(file, blk(i, "attn_v.weight"), d, d, L.wv, error) &&
        loadMat(file, blk(i, "attn_output.weight"), d, d, L.wo, error) &&
        loadVec(file, blk(i, "attn_output_norm.weight"), d, L.attnNorm, error) &&
        loadVec(file, blk(i, "attn_output_norm.bias"), d, L.attnNormB, error) &&
        loadMat(file, blk(i, "ffn_up.weight"), ffn, d, L.ffnUp, error) &&
        loadMat(file, blk(i, "ffn_down.weight"), d, ffn, L.ffnDown, error) &&
        loadVec(file, blk(i, "layer_output_norm.weight"), d, L.ffnNorm, error) &&
        loadVec(file, blk(i, "layer_output_norm.bias"), d, L.ffnNormB, error);
    if (!ok) return std::nullopt;

    // Biases: required when the config saw them on layer 0, so a file with layer 0 biased and
    // layer 7 not is an error rather than a silently biasless layer.
    if (cfg.attnBias) {
      const bool okBias = loadVec(file, blk(i, "attn_q.bias"), d, L.bq, error) &&
                          loadVec(file, blk(i, "attn_k.bias"), d, L.bk, error) &&
                          loadVec(file, blk(i, "attn_v.bias"), d, L.bv, error) &&
                          loadVec(file, blk(i, "attn_output.bias"), d, L.bo, error);
      if (!okBias) return std::nullopt;
    }
    // The FFN biases are independent of attnBias — nomic-bert has neither, plain BERT has both,
    // and nothing rules out a conversion with one and not the other.
    if (!loadVecOpt(file, blk(i, "ffn_up.bias"), ffn, L.ffnUpB, error)) return std::nullopt;
    if (!loadVecOpt(file, blk(i, "ffn_down.bias"), d, L.ffnDownB, error)) return std::nullopt;

    if (cfg.ffnGated) {
      if (!loadMat(file, blk(i, "ffn_gate.weight"), ffn, d, L.ffnGate, error)) return std::nullopt;
    }
  }

  return w;
}

}  // namespace qorvix::runtime
