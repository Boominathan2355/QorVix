#include "qorvix/audio/whisper_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

#include "qorvix/gguf/gguf_error.hpp"
#include "qorvix/runtime/ops.hpp"

namespace qorvix::audio {

namespace rt = qorvix::runtime;

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
// Whisper's timestamp tokens are 20 ms apart, by construction of the vocabulary.
constexpr float kTimestampStep = 0.02f;

}  // namespace

WhisperModel::WhisperModel(WhisperConfig cfg, WhisperWeights weights, tokenizer::Tokenizer tok)
    : cfg_(std::move(cfg)), w_(std::move(weights)), tok_(std::move(tok)) {
  resolveSpecials();

  const std::size_t d = cfg_.dModel;
  const std::size_t n = cfg_.encCtx;
  const std::size_t frames = n * 2;  // the stem's second convolution has stride 2
  const std::size_t encFfn = cfg_.encFfn;
  const std::size_t decFfn = cfg_.decFfn;

  encStates_.assign(n * d, 0.0f);
  encNorm_.assign(n * d, 0.0f);
  encQ_.assign(n * d, 0.0f);
  encK_.assign(n * d, 0.0f);
  encV_.assign(n * d, 0.0f);
  encAttn_.assign(n * d, 0.0f);
  encTmp_.assign(n * std::max(d, encFfn), 0.0f);
  encFfn_.assign(n * encFfn, 0.0f);
  encScores_.assign(static_cast<std::size_t>(cfg_.encHeads) * n, 0.0f);
  // One buffer for both convolutions: they run in sequence, and the second is the larger
  // (d*3 columns over encCtx frames beats melBins*3 over twice as many).
  convCols_.assign(std::max(frames * cfg_.melBins * 3, n * d * 3), 0.0f);
  conv1Out_.assign(frames * d, 0.0f);

  crossK_.assign(cfg_.decLayers, std::vector<float>(n * d, 0.0f));
  crossV_.assign(cfg_.decLayers, std::vector<float>(n * d, 0.0f));
  selfK_.assign(cfg_.decLayers, std::vector<float>(static_cast<std::size_t>(cfg_.decCtx) * d, 0.0f));
  selfV_.assign(cfg_.decLayers, std::vector<float>(static_cast<std::size_t>(cfg_.decCtx) * d, 0.0f));

  x_.assign(d, 0.0f);
  dNorm_.assign(d, 0.0f);
  dQ_.assign(d, 0.0f);
  dK_.assign(d, 0.0f);
  dV_.assign(d, 0.0f);
  dAttn_.assign(d, 0.0f);
  dTmp_.assign(std::max(d, decFfn), 0.0f);
  dFfn_.assign(decFfn, 0.0f);
  // Wide enough for the longer of the two attentions: cross-attention scores span every encoder
  // position, self-attention only the tokens generated so far.
  dScores_.assign(static_cast<std::size_t>(cfg_.decHeads) * std::max<std::size_t>(n, cfg_.decCtx),
                  0.0f);
  logits_.assign(cfg_.vocabSize, 0.0f);
}

std::optional<WhisperModel> WhisperModel::fromGguf(gguf::GgufFile file, std::string& error) {
  WhisperConfig cfg = whisperConfigFromGguf(file, error);
  if (!cfg.valid() || !error.empty()) {
    if (error.empty()) error = "invalid whisper config";
    return std::nullopt;
  }
  auto weights = loadWhisperWeights(file, cfg, error);
  if (!weights) return std::nullopt;
  // The vocabulary is byte-level BPE (gpt2 keys), which the existing Tokenizer already implements —
  // Whisper needed no new tokenizer, only the protocol tokens resolved below.
  auto tok = tokenizer::Tokenizer::fromGguf(file, error);
  if (!tok) return std::nullopt;

  WhisperModel model(std::move(cfg), std::move(*weights), std::move(*tok));
  model.file_ = std::make_unique<gguf::GgufFile>(std::move(file));
  if (!model.validateSpecials(error)) return std::nullopt;
  return model;
}

std::optional<WhisperModel> WhisperModel::fromPath(const std::filesystem::path& path,
                                                   std::string& error) {
  error.clear();
  {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      error = "cannot open '" + path.string() + "'";
      return std::nullopt;
    }
    char magic[4] = {};
    in.read(magic, sizeof magic);
    // 0x67676d6c spelled out on disk. Every "whisper GGUF" on the Hub is this container renamed,
    // so the case is named with its fix rather than surfacing as "bad magic" from the parser.
    if (in.gcount() == 4 && std::memcmp(magic, "lmgg", 4) == 0) {
      error = "'" + path.string() +
              "' is whisper.cpp's legacy ggml container (magic \"lmgg\"), not GGUF — convert the "
              "HuggingFace checkpoint instead: python scripts/convert_whisper_to_gguf.py "
              "openai/whisper-tiny models/whisper-tiny-f32.gguf";
      return std::nullopt;
    }
  }
  try {
    return fromGguf(gguf::GgufFile::open(path), error);
  } catch (const gguf::GgufParseError& e) {
    error = std::string("cannot read '") + path.string() + "': " + e.what();
    return std::nullopt;
  }
}

void WhisperModel::resolveSpecials() {
  auto id = [&](const char* name) { return tok_.tokenToId(name); };
  ids_.sot = id("<|startoftranscript|>");
  ids_.eot = id("<|endoftext|>");
  ids_.transcribe = id("<|transcribe|>");
  ids_.translate = id("<|translate|>");
  ids_.noTimestamps = id("<|notimestamps|>");
  ids_.timestampBegin = id("<|0.00|>");
  // The releases disagree on the name of this one and nothing here depends on it, so it is
  // resolved permissively rather than required.
  ids_.noSpeech = id("<|nocaptions|>");
  if (ids_.noSpeech < 0) ids_.noSpeech = id("<|nospeech|>");
  // Languages sit strictly between <|startoftranscript|> and <|translate|>: derived from the two
  // anchors so the number of languages is whatever the file says it is.
  if (ids_.translate > ids_.sot + 1) {
    ids_.langFirst = ids_.sot + 1;
    ids_.langLast = ids_.translate - 1;
  }
}

bool WhisperModel::validateSpecials(std::string& error) const {
  error.clear();
  if (!ids_.valid()) {
    error = "vocabulary is missing Whisper's protocol tokens (sot " + std::to_string(ids_.sot) +
            ", eot " + std::to_string(ids_.eot) + ", <|0.00|> " +
            std::to_string(ids_.timestampBegin) + ")";
    return false;
  }
  if (ids_.eot != cfg_.textTokenEnd) {
    error = "whisper.text_token_end (" + std::to_string(cfg_.textTokenEnd) +
            ") disagrees with <|endoftext|> (" + std::to_string(ids_.eot) + ")";
    return false;
  }
  return true;
}

MelConfig WhisperModel::melConfig() const {
  // Sample rate, window, hop and chunk length are fixed by Whisper; only the mel count varies
  // (80 through large-v2, 128 on large-v3), and it comes from the file.
  MelConfig c;
  c.nMels = static_cast<int>(cfg_.melBins);
  return c;
}

// ---- encoder ---------------------------------------------------------------------------------

void WhisperModel::encoderAttention(const WhisperEncoderLayer& L, int n) {
  const int d = static_cast<int>(cfg_.dModel);
  const int nHeads = static_cast<int>(cfg_.encHeads);
  const int hd = static_cast<int>(cfg_.encHeadDim());
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  const float* xs = encNorm_.data();
  rt::wmatmulNBias(encQ_.data(), L.attn.wq, xs, n, L.attn.bq);
  rt::wmatmulNBias(encK_.data(), L.attn.wk, xs, n, {});  // k_proj has no bias in Whisper
  rt::wmatmulNBias(encV_.data(), L.attn.wv, xs, n, L.attn.bv);

  // Fully bidirectional: audio has no causal structure inside a window, every position attends to
  // every position. This n^2 term over 1500 positions is the encoder's dominant cost, and each
  // head owns its own score row, so the heads parallelize without sharing anything. The pragma is
  // a no-op unless the build found OpenMP.
#pragma omp parallel for schedule(static)
  for (int h = 0; h < nHeads; ++h) {
    const int off = h * hd;
    float* scores = encScores_.data() + static_cast<std::size_t>(h) * cfg_.encCtx;
    for (int i = 0; i < n; ++i) {
      const float* qi = encQ_.data() + static_cast<std::size_t>(i) * d + off;
      for (int t = 0; t < n; ++t) {
        const float* kt = encK_.data() + static_cast<std::size_t>(t) * d + off;
        float dot = 0.0f;
        for (int e = 0; e < hd; ++e) dot += qi[e] * kt[e];
        scores[t] = dot * scale;
      }
      rt::ops::softmax(scores, n);

      float* outRow = encAttn_.data() + static_cast<std::size_t>(i) * d + off;
      for (int e = 0; e < hd; ++e) outRow[e] = 0.0f;
      for (int t = 0; t < n; ++t) {
        const float wgt = scores[t];
        const float* vt = encV_.data() + static_cast<std::size_t>(t) * d + off;
        for (int e = 0; e < hd; ++e) outRow[e] += wgt * vt[e];
      }
    }
  }
}

bool WhisperModel::encode(const std::vector<float>& logMel, std::string& error) {
  error.clear();
  encoded_ = false;
  const int d = static_cast<int>(cfg_.dModel);
  const int mels = static_cast<int>(cfg_.melBins);
  const int n = static_cast<int>(cfg_.encCtx);
  const int frames = n * 2;
  const int encFfn = static_cast<int>(cfg_.encFfn);
  const float eps = cfg_.normEpsilon;

  const std::size_t expect = static_cast<std::size_t>(mels) * frames;
  if (logMel.size() != expect) {
    error = "expected a [" + std::to_string(mels) + " x " + std::to_string(frames) +
            "] log-mel window (" + std::to_string(expect) + " values), got " +
            std::to_string(logMel.size());
    return false;
  }

  // ---- convolutional stem ----
  // Conv1d with kernel 3 and padding 1, then the same with stride 2. Each output frame is one dot
  // product over an (in_channels x 3) window, so the window is gathered into a column and the whole
  // convolution becomes the batched GEMV every other matmul here uses. Column index is c*3 + k,
  // which is the order the converter kept from PyTorch's [out, in, k] layout.
  for (int t = 0; t < frames; ++t) {
    float* col = convCols_.data() + static_cast<std::size_t>(t) * mels * 3;
    for (int c = 0; c < mels; ++c) {
      for (int k = 0; k < 3; ++k) {
        const int src = t + k - 1;  // padding 1
        col[static_cast<std::size_t>(c) * 3 + k] =
            (src < 0 || src >= frames) ? 0.0f : logMel[static_cast<std::size_t>(c) * frames + src];
      }
    }
  }
  rt::wmatmulNBias(conv1Out_.data(), w_.conv1, convCols_.data(), frames, w_.conv1B);
  rt::ops::geluInPlace(conv1Out_.data(), frames * d);

  for (int t = 0; t < n; ++t) {
    float* col = convCols_.data() + static_cast<std::size_t>(t) * d * 3;
    for (int c = 0; c < d; ++c) {
      for (int k = 0; k < 3; ++k) {
        const int src = 2 * t + k - 1;  // stride 2, padding 1
        col[static_cast<std::size_t>(c) * 3 + k] =
            (src < 0 || src >= frames) ? 0.0f
                                       : conv1Out_[static_cast<std::size_t>(src) * d + c];
      }
    }
  }
  rt::wmatmulNBias(encStates_.data(), w_.conv2, convCols_.data(), n, w_.conv2B);
  rt::ops::geluInPlace(encStates_.data(), n * d);

  // Sinusoidal positions, read from the file rather than regenerated (see the converter).
  for (int t = 0; t < n; ++t) {
    rt::embeddingRow(w_.encPos, t, encTmp_.data());
    rt::ops::add(encStates_.data() + static_cast<std::size_t>(t) * d, encTmp_.data(), d);
  }

  // ---- pre-norm blocks ----
  for (const auto& L : w_.enc) {
    for (int t = 0; t < n; ++t) {
      const float* x = encStates_.data() + static_cast<std::size_t>(t) * d;
      rt::ops::layernorm(encNorm_.data() + static_cast<std::size_t>(t) * d, x, L.attnNormW.data(),
                         L.attnNormB.data(), d, eps);
    }
    encoderAttention(L, n);
    rt::wmatmulNBias(encTmp_.data(), L.attn.wo, encAttn_.data(), n, L.attn.bo);
    for (int t = 0; t < n; ++t) {
      rt::ops::add(encStates_.data() + static_cast<std::size_t>(t) * d,
                   encTmp_.data() + static_cast<std::size_t>(t) * d, d);
    }

    for (int t = 0; t < n; ++t) {
      const float* x = encStates_.data() + static_cast<std::size_t>(t) * d;
      rt::ops::layernorm(encNorm_.data() + static_cast<std::size_t>(t) * d, x, L.ffnNormW.data(),
                         L.ffnNormB.data(), d, eps);
    }
    rt::wmatmulNBias(encFfn_.data(), L.ffnUp, encNorm_.data(), n, L.ffnUpB);
    // Exact erf GELU, the same variant Phase 11a settled empirically against fp32 — not the tanh
    // approximation and not CLIP's quick-GELU.
    rt::ops::geluInPlace(encFfn_.data(), n * encFfn);
    rt::wmatmulNBias(encTmp_.data(), L.ffnDown, encFfn_.data(), n, L.ffnDownB);
    for (int t = 0; t < n; ++t) {
      rt::ops::add(encStates_.data() + static_cast<std::size_t>(t) * d,
                   encTmp_.data() + static_cast<std::size_t>(t) * d, d);
    }
  }

  for (int t = 0; t < n; ++t) {
    float* x = encStates_.data() + static_cast<std::size_t>(t) * d;
    rt::ops::layernorm(x, x, w_.encNormW.data(), w_.encNormB.data(), d, eps);
  }

  // ---- cross-attention keys and values, once per clip ----
  // This is the whole reason the encoder runs separately from the decode loop: K and V over every
  // encoder position are fixed for the clip, so they are computed here and read by every step.
  // Recomputing them per token would multiply the encoder-side projection cost by the transcript
  // length and change no number in the output.
  for (std::size_t l = 0; l < w_.dec.size(); ++l) {
    const WhisperDecoderLayer& L = w_.dec[l];
    rt::wmatmulNBias(crossK_[l].data(), L.cross.wk, encStates_.data(), n, {});
    rt::wmatmulNBias(crossV_[l].data(), L.cross.wv, encStates_.data(), n, L.cross.bv);
  }

  encoded_ = true;
  decLen_ = 0;
  return true;
}

// ---- decoder ---------------------------------------------------------------------------------

void WhisperModel::resetDecoder() { decLen_ = 0; }

void WhisperModel::selfAttention(const WhisperDecoderLayer& L, int layer, int pos) {
  const int d = static_cast<int>(cfg_.dModel);
  const int nHeads = static_cast<int>(cfg_.decHeads);
  const int hd = static_cast<int>(cfg_.decHeadDim());
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  rt::wmatmulBias(dQ_.data(), L.attn.wq, dNorm_.data(), L.attn.bq);
  rt::wmatmulBias(dK_.data(), L.attn.wk, dNorm_.data(), {});
  rt::wmatmulBias(dV_.data(), L.attn.wv, dNorm_.data(), L.attn.bv);

  // Causality is structural here, not a mask: the cache holds positions 0..pos and nothing later
  // exists yet, so there is nothing to mask out.
  float* kCache = selfK_[static_cast<std::size_t>(layer)].data();
  float* vCache = selfV_[static_cast<std::size_t>(layer)].data();
  std::copy_n(dK_.data(), d, kCache + static_cast<std::size_t>(pos) * d);
  std::copy_n(dV_.data(), d, vCache + static_cast<std::size_t>(pos) * d);

  const int len = pos + 1;
  const std::size_t stride = std::max<std::size_t>(cfg_.encCtx, cfg_.decCtx);
  for (int h = 0; h < nHeads; ++h) {
    const int off = h * hd;
    float* scores = dScores_.data() + static_cast<std::size_t>(h) * stride;
    const float* qh = dQ_.data() + off;
    for (int t = 0; t < len; ++t) {
      const float* kt = kCache + static_cast<std::size_t>(t) * d + off;
      float dot = 0.0f;
      for (int e = 0; e < hd; ++e) dot += qh[e] * kt[e];
      scores[t] = dot * scale;
    }
    rt::ops::softmax(scores, len);
    float* outRow = dAttn_.data() + off;
    for (int e = 0; e < hd; ++e) outRow[e] = 0.0f;
    for (int t = 0; t < len; ++t) {
      const float wgt = scores[t];
      const float* vt = vCache + static_cast<std::size_t>(t) * d + off;
      for (int e = 0; e < hd; ++e) outRow[e] += wgt * vt[e];
    }
  }
}

void WhisperModel::crossAttention(const WhisperDecoderLayer& L, int layer) {
  const int d = static_cast<int>(cfg_.dModel);
  const int nHeads = static_cast<int>(cfg_.decHeads);
  const int hd = static_cast<int>(cfg_.decHeadDim());
  const int n = static_cast<int>(cfg_.encCtx);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  // Only the query comes from the text stream; K and V were fixed when the audio was encoded.
  rt::wmatmulBias(dQ_.data(), L.cross.wq, dNorm_.data(), L.cross.bq);

  const float* kCache = crossK_[static_cast<std::size_t>(layer)].data();
  const float* vCache = crossV_[static_cast<std::size_t>(layer)].data();
  const std::size_t stride = std::max<std::size_t>(cfg_.encCtx, cfg_.decCtx);
#pragma omp parallel for schedule(static)
  for (int h = 0; h < nHeads; ++h) {
    const int off = h * hd;
    float* scores = dScores_.data() + static_cast<std::size_t>(h) * stride;
    const float* qh = dQ_.data() + off;
    for (int t = 0; t < n; ++t) {
      const float* kt = kCache + static_cast<std::size_t>(t) * d + off;
      float dot = 0.0f;
      for (int e = 0; e < hd; ++e) dot += qh[e] * kt[e];
      scores[t] = dot * scale;
    }
    rt::ops::softmax(scores, n);
    float* outRow = dAttn_.data() + off;
    for (int e = 0; e < hd; ++e) outRow[e] = 0.0f;
    for (int t = 0; t < n; ++t) {
      const float wgt = scores[t];
      const float* vt = vCache + static_cast<std::size_t>(t) * d + off;
      for (int e = 0; e < hd; ++e) outRow[e] += wgt * vt[e];
    }
  }
}

bool WhisperModel::step(int token, int pos, std::string& error) {
  error.clear();
  if (!encoded_) {
    // Attending to a zero-filled cross cache would still produce fluent, confident text — about
    // nothing. Refused rather than allowed to look like a working transcript.
    error = "encode() must run before a decoder step";
    return false;
  }
  if (token < 0 || token >= static_cast<int>(cfg_.vocabSize)) {
    error = "token id " + std::to_string(token) + " is outside the vocabulary";
    return false;
  }
  if (pos < 0 || pos >= static_cast<int>(cfg_.decCtx)) {
    error = "position " + std::to_string(pos) + " is beyond the decoder's position table (" +
            std::to_string(cfg_.decCtx) + ")";
    return false;
  }
  if (pos != decLen_) {
    error = "decoder positions must be filled in order: expected " + std::to_string(decLen_) +
            ", got " + std::to_string(pos);
    return false;
  }

  const int d = static_cast<int>(cfg_.dModel);
  const int decFfn = static_cast<int>(cfg_.decFfn);
  const float eps = cfg_.normEpsilon;

  // Token embedding plus the LEARNED position row — the decoder's positions are a trained table,
  // unlike the encoder's sinusoids.
  rt::embeddingRow(w_.tokenEmbd, token, x_.data());
  rt::embeddingRow(w_.decPos, pos, dTmp_.data());
  rt::ops::add(x_.data(), dTmp_.data(), d);

  for (std::size_t l = 0; l < w_.dec.size(); ++l) {
    const WhisperDecoderLayer& L = w_.dec[l];

    rt::ops::layernorm(dNorm_.data(), x_.data(), L.attnNormW.data(), L.attnNormB.data(), d, eps);
    selfAttention(L, static_cast<int>(l), pos);
    rt::wmatmulBias(dTmp_.data(), L.attn.wo, dAttn_.data(), L.attn.bo);
    rt::ops::add(x_.data(), dTmp_.data(), d);

    rt::ops::layernorm(dNorm_.data(), x_.data(), L.crossNormW.data(), L.crossNormB.data(), d, eps);
    crossAttention(L, static_cast<int>(l));
    rt::wmatmulBias(dTmp_.data(), L.cross.wo, dAttn_.data(), L.cross.bo);
    rt::ops::add(x_.data(), dTmp_.data(), d);

    rt::ops::layernorm(dNorm_.data(), x_.data(), L.ffnNormW.data(), L.ffnNormB.data(), d, eps);
    rt::wmatmulBias(dFfn_.data(), L.ffnUp, dNorm_.data(), L.ffnUpB);
    rt::ops::geluInPlace(dFfn_.data(), decFfn);
    rt::wmatmulBias(dTmp_.data(), L.ffnDown, dFfn_.data(), L.ffnDownB);
    rt::ops::add(x_.data(), dTmp_.data(), d);
  }

  rt::ops::layernorm(x_.data(), x_.data(), w_.decNormW.data(), w_.decNormB.data(), d, eps);
  // The LM head is the token embedding, tied and biasless — asserted at conversion time.
  rt::wmatmul(logits_.data(), w_.tokenEmbd, x_.data());

  decLen_ = pos + 1;
  return true;
}

// ---- protocol and greedy decode ---------------------------------------------------------------

int WhisperModel::languageToken(const std::string& code) const {
  if (code.empty()) return -1;
  const int id = tok_.tokenToId("<|" + code + "|>");
  if (id < 0) return -1;
  if (ids_.langFirst >= 0 && (id < ids_.langFirst || id > ids_.langLast)) return -1;
  return id;
}

std::string WhisperModel::languageCode(int token) const {
  if (token < 0) return {};
  const std::string& t = tok_.idToToken(token);
  if (t.size() > 4 && t.compare(0, 2, "<|") == 0 && t.compare(t.size() - 2, 2, "|>") == 0) {
    return t.substr(2, t.size() - 4);
  }
  return {};
}

std::string WhisperModel::decodeText(const std::vector<int>& tokens) const {
  std::vector<int> text;
  text.reserve(tokens.size());
  for (int id : tokens) {
    if (id < cfg_.textTokenEnd) text.push_back(id);
  }
  return tok_.decode(text, /*skipSpecial=*/true);
}

std::vector<int> WhisperModel::promptTokens(const TranscribeOptions& opt, std::string& error) const {
  error.clear();
  std::vector<int> out{ids_.sot};
  if (cfg_.multilingual) {
    const std::string code = opt.language.empty() ? std::string("en") : opt.language;
    const int lang = languageToken(code);
    if (lang < 0) {
      error = "unknown language '" + code + "' — this vocabulary has no <|" + code + "|> token";
      return {};
    }
    out.push_back(lang);
    const int task = opt.translate ? ids_.translate : ids_.transcribe;
    if (task < 0) {
      error = "vocabulary has no <|transcribe|>/<|translate|> token";
      return {};
    }
    out.push_back(task);
  } else if (opt.translate) {
    // An English-only model has no language tokens at all, so "translate to English" has no way
    // to be expressed and no meaning; refused rather than silently transcribing instead.
    error = "this model is English-only and cannot translate";
    return {};
  }
  if (!opt.timestamps) {
    if (ids_.noTimestamps < 0) {
      error = "vocabulary has no <|notimestamps|> token";
      return {};
    }
    out.push_back(ids_.noTimestamps);
  }
  return out;
}

int WhisperModel::pickToken(bool firstGenerated, bool timestamps) {
  const int vocab = static_cast<int>(cfg_.vocabSize);
  // The model's own list: ids the released checkpoints never emit as text.
  for (int id : cfg_.suppressTokens) {
    if (id >= 0 && id < vocab) logits_[static_cast<std::size_t>(id)] = kNegInf;
  }
  // At the first generated position only: a leading space and an immediate <|endoftext|>, which
  // would end the transcript before it began.
  if (firstGenerated) {
    for (int id : cfg_.beginSuppressTokens) {
      if (id >= 0 && id < vocab) logits_[static_cast<std::size_t>(id)] = kNegInf;
    }
  }
  // The protocol tokens are the model's control plane, not vocabulary: sot, the languages, the
  // tasks and <|notimestamps|> must never appear mid-transcript. <|endoftext|> always may, and the
  // timestamps may when timestamps were asked for.
  for (int id = cfg_.textTokenEnd; id < vocab; ++id) {
    if (id == ids_.eot) continue;
    if (timestamps && id >= ids_.timestampBegin) continue;
    logits_[static_cast<std::size_t>(id)] = kNegInf;
  }
  return rt::ops::argmax(logits_.data(), vocab);
}

bool WhisperModel::detectLanguage(std::string& code, std::string& error) {
  error.clear();
  code.clear();
  if (!encoded_) {
    error = "encode() must run before language detection";
    return false;
  }
  if (!cfg_.multilingual) {
    code = "en";  // an English-only model has nothing to detect
    return true;
  }
  if (ids_.langFirst < 0) {
    error = "vocabulary carries no language tokens";
    return false;
  }
  // Whisper's own procedure: one step from <|startoftranscript|> alone, then argmax restricted to
  // the language ids. Restricting matters — the unrestricted argmax at that position is a text
  // token, because the model is also predicting what the transcript will say.
  resetDecoder();
  if (!step(ids_.sot, 0, error)) return false;
  int best = ids_.langFirst;
  float bestVal = logits_[static_cast<std::size_t>(ids_.langFirst)];
  for (int id = ids_.langFirst + 1; id <= ids_.langLast; ++id) {
    if (logits_[static_cast<std::size_t>(id)] > bestVal) {
      bestVal = logits_[static_cast<std::size_t>(id)];
      best = id;
    }
  }
  resetDecoder();
  code = languageCode(best);
  if (code.empty()) {
    error = "language token " + std::to_string(best) + " has an unreadable name";
    return false;
  }
  return true;
}

bool WhisperModel::generate(const TranscribeOptions& opt, TranscribeResult& out,
                            std::string& error) {
  error.clear();
  out = TranscribeResult{};
  if (!encoded_) {
    error = "encode() must run before generate()";
    return false;
  }

  TranscribeOptions o = opt;
  if (o.language.empty()) {
    std::string detected;
    if (!detectLanguage(detected, error)) return false;
    o.language = detected;
    out.languageDetected = cfg_.multilingual;
  }
  out.language = o.language;

  const std::vector<int> prompt = promptTokens(o, error);
  if (prompt.empty()) return false;

  resetDecoder();
  int pos = 0;
  for (int t : prompt) {
    if (!step(t, pos, error)) return false;
    ++pos;
  }
  out.tokens = prompt;

  const int budget = o.maxTokens > 0 ? o.maxTokens : static_cast<int>(cfg_.decCtx);
  const int limit = std::min<int>(static_cast<int>(cfg_.decCtx),
                                  static_cast<int>(prompt.size()) + budget);
  bool first = true;
  bool sawEot = false;
  while (pos < limit) {
    const int next = pickToken(first, o.timestamps);
    first = false;
    if (next == ids_.eot) {
      sawEot = true;
      break;
    }
    out.tokens.push_back(next);
    if (!step(next, pos, error)) return false;
    ++pos;
  }
  // Reported rather than hidden: a transcript cut off by the position table is a different thing
  // from one the model chose to end, and only the caller can decide whether to re-run on a
  // shorter window.
  out.hitTokenLimit = !sawEot;
  out.text = decodeText(out.tokens);

  if (o.timestamps) {
    // Whisper emits <|t0|> text <|t1|> <|t1|> text <|t2|> …, so timestamp tokens alternate between
    // opening and closing a segment.
    float start = -1.0f;
    std::vector<int> pending;
    for (int id : out.tokens) {
      if (id < cfg_.textTokenEnd) {
        pending.push_back(id);
        continue;
      }
      if (id < ids_.timestampBegin) continue;  // sot/language/task/notimestamps
      const float time = static_cast<float>(id - ids_.timestampBegin) * kTimestampStep;
      if (start < 0.0f) {
        start = time;
        pending.clear();
      } else {
        out.segments.push_back({start, time, decodeText(pending)});
        pending.clear();
        start = -1.0f;
      }
    }
    // A segment left open by <|endoftext|> (or by the token limit) is kept with an unknown end
    // rather than dropped, since its text is real.
    if (start >= 0.0f && !pending.empty()) {
      out.segments.push_back({start, start, decodeText(pending)});
    }
  }
  return true;
}

bool WhisperModel::transcribe(const std::vector<float>& logMel, const TranscribeOptions& opt,
                              TranscribeResult& out, std::string& error) {
  if (!encode(logMel, error)) return false;
  return generate(opt, out, error);
}

}  // namespace qorvix::audio
