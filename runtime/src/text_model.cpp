#include "qorvix/runtime/text_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/ops.hpp"

namespace qorvix::runtime {

namespace {
constexpr int kKvTokensPerPage = 256;

memory::KvCacheConfig kvConfig(const ModelConfig& c) {
  return memory::KvCacheConfig{static_cast<int>(c.blockCount), static_cast<int>(c.kvDim()),
                               kKvTokensPerPage};
}

// Pool sized to hold `maxSessions` sequences of maxSeq tokens across all layers.
std::size_t kvPoolBytes(const ModelConfig& c, std::uint32_t maxSeq, std::uint32_t maxSessions) {
  const std::size_t pagesPerLayer = (maxSeq + kKvTokensPerPage - 1) / kKvTokensPerPage;
  const std::size_t pages =
      static_cast<std::size_t>(c.blockCount) * pagesPerLayer * std::max<std::uint32_t>(1, maxSessions);
  const std::size_t pageFloats = static_cast<std::size_t>(kKvTokensPerPage) * c.kvDim() * 2;
  return pages * pageFloats * sizeof(float);
}
}  // namespace

TextModel::TextModel(ModelConfig config, Weights weights, std::uint32_t maxSeqLen,
                     std::uint32_t maxSessions)
    : cfg_(std::move(config)),
      w_(std::move(weights)),
      maxSeq_(maxSeqLen),
      kv_(kvConfig(cfg_), kvPoolBytes(cfg_, maxSeqLen, maxSessions)) {
  const std::size_t d = cfg_.embeddingLength;
  const std::size_t kv = cfg_.kvDim();
  const std::size_t ffn = cfg_.feedForwardLength;

  session_ = kv_.open();

  x_.assign(d, 0.0f);
  xn_.assign(d, 0.0f);
  q_.assign(static_cast<std::size_t>(cfg_.headCount) * cfg_.headDim(), 0.0f);
  k_.assign(kv, 0.0f);
  v_.assign(kv, 0.0f);
  attn_.assign(static_cast<std::size_t>(cfg_.headCount) * cfg_.headDim(), 0.0f);
  ffnGate_.assign(ffn, 0.0f);
  ffnUp_.assign(ffn, 0.0f);
  ffnAct_.assign(ffn, 0.0f);
  tmpDModel_.assign(d, 0.0f);
  logits_.assign(cfg_.vocabSize, 0.0f);
}

std::optional<TextModel> TextModel::fromGguf(gguf::GgufFile file, std::string& error,
                                            std::uint32_t maxSeqLen, std::uint32_t maxSessions) {
  ModelConfig cfg = configFromGguf(file, error);
  if (!cfg.valid()) return std::nullopt;
  auto weights = loadWeights(file, cfg, error);  // borrows pointers into file's mmap
  if (!weights) return std::nullopt;

  TextModel model(std::move(cfg), std::move(*weights), maxSeqLen, maxSessions);
  // Take ownership of the file so the mmap (and the borrowed weight pointers) outlive this call.
  // Moving GgufFile transfers the same mapping address, so the borrowed pointers stay valid.
  model.file_ = std::make_unique<gguf::GgufFile>(std::move(file));
  return model;
}

void TextModel::attention(memory::SessionId session, const LayerWeights& L, int layer, int pos) {
  const int headDim = static_cast<int>(cfg_.headDim());
  const int nHeads = static_cast<int>(cfg_.headCount);
  const int nKv = static_cast<int>(cfg_.headCountKv);
  const int kvDim = static_cast<int>(cfg_.kvDim());
  const int group = nHeads / nKv;  // query heads per kv head (GQA)
  const float invSqrt = 1.0f / std::sqrt(static_cast<float>(headDim));

  // Store the freshly projected K/V for this position into the paged cache.
  float* kRow = kv_.kSlot(session, layer, pos);
  float* vRow = kv_.vSlot(session, layer, pos);
  for (int i = 0; i < kvDim; ++i) {
    kRow[i] = k_[i];
    vRow[i] = v_[i];
  }

  std::vector<float> scores(pos + 1);
  for (int h = 0; h < nHeads; ++h) {
    const float* qh = q_.data() + static_cast<std::size_t>(h) * headDim;
    const int kvHead = h / group;
    const std::size_t headOff = static_cast<std::size_t>(kvHead) * headDim;

    for (int t = 0; t <= pos; ++t) {
      const float* kt = kv_.kSlot(session, layer, t) + headOff;
      float dot = 0.0f;
      for (int d = 0; d < headDim; ++d) dot += qh[d] * kt[d];
      scores[t] = dot * invSqrt;
    }
    ops::softmax(scores.data(), pos + 1);

    float* outH = attn_.data() + static_cast<std::size_t>(h) * headDim;
    for (int d = 0; d < headDim; ++d) outH[d] = 0.0f;
    for (int t = 0; t <= pos; ++t) {
      const float* vt = kv_.vSlot(session, layer, t) + headOff;
      const float s = scores[t];
      for (int d = 0; d < headDim; ++d) outH[d] += s * vt[d];
    }
  }
}

const std::vector<float>& TextModel::forward(memory::SessionId session, int token, int pos) {
  const int d = static_cast<int>(cfg_.embeddingLength);
  const int headDim = static_cast<int>(cfg_.headDim());

  // Grow the paged KV cache so slot `pos` exists in every layer (normally appends one token).
  while (kv_.length(session) <= pos) {
    if (!kv_.appendToken(session)) break;  // pool exhausted; attention will clamp to what exists
  }

  // Embedding lookup (dequantizes one row when the table is quantized).
  embeddingRow(w_.tokenEmbd, token, x_.data());

  for (std::uint32_t layer = 0; layer < cfg_.blockCount; ++layer) {
    const LayerWeights& L = w_.layers[layer];

    // --- attention block ---
    ops::rmsnorm(xn_.data(), x_.data(), L.attnNorm.data(), d, cfg_.normEpsilon);
    wmatmul(q_.data(), L.wq, xn_.data());
    wmatmul(k_.data(), L.wk, xn_.data());
    wmatmul(v_.data(), L.wv, xn_.data());

    ops::rope(q_.data(), static_cast<int>(cfg_.headCount), headDim, pos, cfg_.ropeFreqBase,
              cfg_.ropeMode);
    ops::rope(k_.data(), static_cast<int>(cfg_.headCountKv), headDim, pos, cfg_.ropeFreqBase,
              cfg_.ropeMode);

    attention(session, L, static_cast<int>(layer), pos);

    wmatmul(tmpDModel_.data(), L.wo, attn_.data());
    ops::add(x_.data(), tmpDModel_.data(), d);

    // --- feed-forward block (SwiGLU) ---
    ops::rmsnorm(xn_.data(), x_.data(), L.ffnNorm.data(), d, cfg_.normEpsilon);
    wmatmul(ffnGate_.data(), L.ffnGate, xn_.data());
    wmatmul(ffnUp_.data(), L.ffnUp, xn_.data());
    ops::swiglu(ffnAct_.data(), ffnGate_.data(), ffnUp_.data(),
                static_cast<int>(cfg_.feedForwardLength));
    wmatmul(tmpDModel_.data(), L.ffnDown, ffnAct_.data());
    ops::add(x_.data(), tmpDModel_.data(), d);
  }

  // Final norm + LM head.
  ops::rmsnorm(xn_.data(), x_.data(), w_.outputNorm.data(), d, cfg_.normEpsilon);
  wmatmul(logits_.data(), w_.lmHead(), xn_.data());

  return logits_;
}

std::vector<int> TextModel::generateGreedy(const std::vector<int>& prompt, int maxNew) {
  reset();
  std::vector<int> generated;
  int pos = 0;
  int next = 0;

  for (std::size_t i = 0; i < prompt.size() && pos < static_cast<int>(maxSeq_); ++i, ++pos) {
    const auto& logits = forward(prompt[i], pos);
    next = ops::argmax(logits.data(), static_cast<int>(cfg_.vocabSize));
  }

  for (int n = 0; n < maxNew && pos < static_cast<int>(maxSeq_); ++n, ++pos) {
    generated.push_back(next);
    const auto& logits = forward(next, pos);
    next = ops::argmax(logits.data(), static_cast<int>(cfg_.vocabSize));
  }
  return generated;
}

}  // namespace qorvix::runtime
