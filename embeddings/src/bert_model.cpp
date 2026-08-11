#include "qorvix/embeddings/bert_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/pooling.hpp"

namespace qorvix::embeddings {

namespace rt = qorvix::runtime;

bool IEmbeddingEngine::embedBatch(const std::vector<std::vector<int>>& batch,
                                  std::vector<std::vector<float>>& out, std::string& error) {
  out.clear();
  out.reserve(batch.size());
  for (const auto& tokens : batch) {
    std::vector<float> v;
    if (!embed(tokens, v, error)) return false;
    out.push_back(std::move(v));
  }
  return true;
}

BertModel::BertModel(rt::ModelConfig cfg, rt::EncoderWeights weights, std::uint32_t maxSeq)
    : cfg_(std::move(cfg)), w_(std::move(weights)), maxSeq_(maxSeq) {
  const std::size_t d = cfg_.embeddingLength;
  const std::size_t ffn = cfg_.feedForwardLength;
  const std::size_t n = maxSeq_;

  states_.assign(n * d, 0.0f);
  norm_.assign(n * d, 0.0f);
  q_.assign(n * d, 0.0f);
  k_.assign(n * d, 0.0f);
  v_.assign(n * d, 0.0f);
  attn_.assign(n * d, 0.0f);
  tmp_.assign(n * d, 0.0f);
  scores_.assign(static_cast<std::size_t>(cfg_.headCount) * n, 0.0f);
  ffn_.assign(n * ffn, 0.0f);
  if (cfg_.ffnGated) ffnGate_.assign(n * ffn, 0.0f);
}

std::optional<BertModel> BertModel::fromGguf(gguf::GgufFile file, std::string& error,
                                             std::uint32_t maxSeq) {
  rt::ModelConfig cfg = rt::configFromGguf(file, error);
  if (!cfg.valid()) return std::nullopt;
  if (!cfg.isEncoder()) {
    error = "'" + cfg.architecture +
            "' is a decoder model — use `qorvix generate`, not the embedding path";
    return std::nullopt;
  }
  auto weights = rt::loadEncoderWeights(file, cfg, error);
  if (!weights) return std::nullopt;

  // The learned position table is the hard ceiling: there is no row beyond contextLength, so a
  // longer request cannot be served at any cost. Rope-based encoders could extrapolate, but not
  // meaningfully, so both are clamped the same way.
  const std::uint32_t cap = cfg.contextLength ? cfg.contextLength : 512;
  const std::uint32_t seq = (maxSeq == 0 || maxSeq > cap) ? cap : maxSeq;

  BertModel model(std::move(cfg), std::move(*weights), seq);
  model.file_ = std::make_unique<gguf::GgufFile>(std::move(file));
  return model;
}

void BertModel::attention(const rt::EncoderLayerWeights& L, int n) {
  const int d = static_cast<int>(cfg_.embeddingLength);
  const int nHeads = static_cast<int>(cfg_.headCount);
  const int headDim = static_cast<int>(cfg_.headDim());
  const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

  // Project every position first, then attend. Interleaving projection with attention would work
  // identically, but this shape is what a batched GEMV (qmatmulN) drops into as a single change:
  // one pass over each weight matrix instead of one pass per token.
  for (int t = 0; t < n; ++t) {
    const float* x = norm_.data() + static_cast<std::size_t>(t) * d;
    rt::wmatmulBias(q_.data() + static_cast<std::size_t>(t) * d, L.wq, x, L.bq);
    rt::wmatmulBias(k_.data() + static_cast<std::size_t>(t) * d, L.wk, x, L.bk);
    rt::wmatmulBias(v_.data() + static_cast<std::size_t>(t) * d, L.wv, x, L.bv);
  }

  for (int h = 0; h < nHeads; ++h) {
    const int off = h * headDim;
    float* scores = scores_.data() + static_cast<std::size_t>(h) * maxSeq_;
    for (int i = 0; i < n; ++i) {
      const float* qi = q_.data() + static_cast<std::size_t>(i) * d + off;

      // Bidirectional: every query attends to every key. The decoder's causal mask is implicit in
      // TextModel::attention's `t <= pos` loop bound; here the bound is simply `t < n`. That one
      // character is the whole difference, and it is why bert_model_test asserts that changing the
      // LAST token moves the FIRST token's state.
      for (int t = 0; t < n; ++t) {
        const float* kt = k_.data() + static_cast<std::size_t>(t) * d + off;
        float dot = 0.0f;
        for (int e = 0; e < headDim; ++e) dot += qi[e] * kt[e];
        scores[t] = dot * scale;
      }
      rt::ops::softmax(scores, n);

      float* outRow = attn_.data() + static_cast<std::size_t>(i) * d + off;
      for (int e = 0; e < headDim; ++e) outRow[e] = 0.0f;
      for (int t = 0; t < n; ++t) {
        const float wgt = scores[t];
        const float* vt = v_.data() + static_cast<std::size_t>(t) * d + off;
        for (int e = 0; e < headDim; ++e) outRow[e] += wgt * vt[e];
      }
    }
  }
}

bool BertModel::encode(const std::vector<int>& tokens, std::string& error) {
  const int n = static_cast<int>(tokens.size());
  const int d = static_cast<int>(cfg_.embeddingLength);
  const int ffn = static_cast<int>(cfg_.feedForwardLength);
  const float eps = cfg_.normEpsilon;

  if (n == 0) {
    error = "cannot embed an empty token sequence";
    return false;
  }
  if (n > static_cast<int>(maxSeq_)) {
    error = "sequence of " + std::to_string(n) + " tokens exceeds the model's limit of " +
            std::to_string(maxSeq_);
    return false;
  }
  for (int id : tokens) {
    if (id < 0 || id >= static_cast<int>(cfg_.vocabSize)) {
      error = "token id " + std::to_string(id) + " is outside the vocabulary";
      return false;
    }
  }

  // ---- embeddings: token + segment + position, then LayerNorm ----
  for (int t = 0; t < n; ++t) {
    float* x = states_.data() + static_cast<std::size_t>(t) * d;
    rt::embeddingRow(w_.tokenEmbd, tokens[t], x);

    // Segment 0 for every token: these are single-sequence embeddings, so the second segment is
    // never used. Skipping the add entirely would be wrong — segment 0's row is not zero.
    if (w_.tokenTypes.valid()) {
      rt::embeddingRow(w_.tokenTypes, 0, tmp_.data());
      rt::ops::add(x, tmp_.data(), d);
    }
    if (w_.positionEmbd.valid()) {
      rt::embeddingRow(w_.positionEmbd, t, tmp_.data());
      rt::ops::add(x, tmp_.data(), d);
    } else if (cfg_.ropeDimensionCount > 0) {
      // nomic-bert applies rope to Q/K instead of a learned table; that variant is not verified
      // yet (see the roadmap), so refuse rather than emit a plausible wrong vector.
      error = "rope-based encoders (nomic-bert) are not supported yet";
      return false;
    }

    if (!w_.embdNorm.empty()) {
      rt::ops::layernorm(x, x, w_.embdNorm.data(),
                         w_.embdNormB.empty() ? nullptr : w_.embdNormB.data(), d, eps);
    }
  }

  // ---- transformer layers: post-norm, GELU FFN ----
  for (const auto& L : w_.layers) {
    // BERT is post-norm: the LayerNorm comes AFTER the sublayer and its residual, not before it.
    // The input to the attention projections is therefore the raw residual stream, so norm_ here
    // is just a copy — kept as a separate buffer so attention() reads a stable input while
    // states_ is being updated in place.
    std::copy(states_.begin(), states_.begin() + static_cast<std::size_t>(n) * d, norm_.begin());
    attention(L, n);

    for (int t = 0; t < n; ++t) {
      float* x = states_.data() + static_cast<std::size_t>(t) * d;
      rt::wmatmulBias(tmp_.data(), L.wo, attn_.data() + static_cast<std::size_t>(t) * d, L.bo);
      rt::ops::add(x, tmp_.data(), d);  // residual
      rt::ops::layernorm(x, x, L.attnNorm.data(),
                         L.attnNormB.empty() ? nullptr : L.attnNormB.data(), d, eps);
    }

    for (int t = 0; t < n; ++t) {
      float* x = states_.data() + static_cast<std::size_t>(t) * d;
      float* up = ffn_.data() + static_cast<std::size_t>(t) * ffn;
      rt::wmatmulBias(up, L.ffnUp, x, L.ffnUpB);
      if (cfg_.ffnGated) {
        float* gate = ffnGate_.data() + static_cast<std::size_t>(t) * ffn;
        rt::wmatmul(gate, L.ffnGate, x);
        rt::ops::geluInPlace(gate, ffn);
        for (int i = 0; i < ffn; ++i) up[i] *= gate[i];
      } else {
        rt::ops::geluInPlace(up, ffn);
      }
      rt::wmatmulBias(tmp_.data(), L.ffnDown, up, L.ffnDownB);
      rt::ops::add(x, tmp_.data(), d);  // residual
      rt::ops::layernorm(x, x, L.ffnNorm.data(),
                         L.ffnNormB.empty() ? nullptr : L.ffnNormB.data(), d, eps);
    }
  }
  return true;
}

bool BertModel::embedWith(const std::vector<int>& tokens, PoolingType pooling, bool normalize,
                          std::vector<float>& out, std::string& error) {
  error.clear();
  if (!encode(tokens, error)) return false;

  const int n = static_cast<int>(tokens.size());
  const int d = static_cast<int>(cfg_.embeddingLength);
  out.assign(d, 0.0f);

  switch (pooling) {
    case PoolingType::Cls:
      rt::ops::clsPool(out.data(), states_.data(), n, d);
      break;
    case PoolingType::Last:
      rt::ops::lastPool(out.data(), states_.data(), n, d);
      break;
    case PoolingType::Mean:
    case PoolingType::None:
      // None means "the model declares no pooling"; mean is the conventional fallback and what
      // sentence-transformers applies when a model ships no pooling config.
      rt::ops::meanPool(out.data(), states_.data(), n, d);
      break;
  }
  if (normalize) rt::ops::l2Normalize(out.data(), d);
  return true;
}

bool BertModel::embed(const std::vector<int>& tokens, std::vector<float>& out,
                      std::string& error) {
  return embedWith(tokens, cfg_.pooling, defaultNormalize(), out, error);
}

bool BertModel::embedTokens(const std::vector<int>& tokens, std::vector<float>& out,
                            std::string& error) {
  error.clear();
  if (!encode(tokens, error)) return false;
  const std::size_t n = tokens.size();
  const std::size_t d = cfg_.embeddingLength;
  out.assign(states_.begin(), states_.begin() + n * d);
  return true;
}

}  // namespace qorvix::embeddings
