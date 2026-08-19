#include "qorvix/audio/whisper_weights.hpp"

#include "qorvix/gguf/gguf_file.hpp"

// Reuses the runtime's tensor-load helpers rather than duplicating the mmap borrowing,
// element-count checking and quantized-type gating — the same call the BERT and CLIP loaders make.
#include "qorvix/runtime/tensor_load.hpp"

namespace qorvix::audio {

namespace rt = qorvix::runtime;
using rt::detail::loadMat;
using rt::detail::loadVec;

namespace {

std::string encBlk(int i, const char* suffix) {
  return "enc.blk." + std::to_string(i) + "." + suffix;
}

std::string decBlk(int i, const char* suffix) {
  return "dec.blk." + std::to_string(i) + "." + suffix;
}

// q, k, v, out with biases on everything except k. Called with `prefix` = "enc.blk.0.attn",
// "dec.blk.0.cross_attn" and so on, so self- and cross-attention load through one path.
bool loadAttn(const gguf::GgufFile& file, const std::string& prefix, int d,
              WhisperAttnWeights& out, std::string& error) {
  return loadMat(file, prefix + "_q.weight", d, d, out.wq, error) &&
         loadVec(file, prefix + "_q.bias", d, out.bq, error) &&
         loadMat(file, prefix + "_k.weight", d, d, out.wk, error) &&
         loadMat(file, prefix + "_v.weight", d, d, out.wv, error) &&
         loadVec(file, prefix + "_v.bias", d, out.bv, error) &&
         loadMat(file, prefix + "_out.weight", d, d, out.wo, error) &&
         loadVec(file, prefix + "_out.bias", d, out.bo, error);
}

bool loadNorm(const gguf::GgufFile& file, const std::string& prefix, int d,
              std::vector<float>& w, std::vector<float>& b, std::string& error) {
  return loadVec(file, prefix + ".weight", d, w, error) &&
         loadVec(file, prefix + ".bias", d, b, error);
}

}  // namespace

WhisperConfig whisperConfigFromGguf(const gguf::GgufFile& file, std::string& error) {
  error.clear();
  WhisperConfig cfg;
  if (file.architecture() != "whisper") {
    error = "architecture '" + file.architecture() + "' is not a whisper model";
    return cfg;
  }
  cfg.name = file.getString("general.name").value_or("");

  auto u32 = [&](const char* key, std::uint32_t fallback) {
    if (auto v = file.getU64(key)) return static_cast<std::uint32_t>(*v);
    return fallback;
  };
  cfg.dModel = u32("whisper.embedding_length", 0);
  cfg.melBins = u32("whisper.mel_bins", cfg.melBins);
  cfg.vocabSize = u32("whisper.vocab_size", 0);
  cfg.encLayers = u32("whisper.encoder.block_count", 0);
  cfg.encHeads = u32("whisper.encoder.attention.head_count", 0);
  cfg.encFfn = u32("whisper.encoder.feed_forward_length", 0);
  cfg.encCtx = u32("whisper.encoder.context_length", cfg.encCtx);
  cfg.decLayers = u32("whisper.decoder.block_count", 0);
  cfg.decHeads = u32("whisper.decoder.attention.head_count", 0);
  cfg.decFfn = u32("whisper.decoder.feed_forward_length", 0);
  cfg.decCtx = u32("whisper.decoder.context_length", cfg.decCtx);
  if (auto v = file.getF64("whisper.attention.layer_norm_epsilon")) {
    cfg.normEpsilon = static_cast<float>(*v);
  }
  cfg.multilingual = file.getBool("whisper.is_multilingual").value_or(true);
  // The specials boundary comes from the file rather than from a constant here: it is 50257 on
  // every OpenAI release, but it is a property of the vocabulary, and a hardcoded 50257 against a
  // differently-sized vocabulary would silently either leak specials into the transcript or eat
  // real text off the end of it.
  cfg.textTokenEnd = static_cast<int>(u32("whisper.text_token_end", 0));

  auto readIds = [&](const char* key, std::vector<int>& dst) {
    const gguf::GgufValue* v = file.find(key);
    if (!v || !v->isArray()) return;
    dst.reserve(v->array().size());
    for (const auto& e : v->array()) {
      if (auto id = e.asI64()) dst.push_back(static_cast<int>(*id));
    }
  };
  readIds("whisper.suppress_tokens", cfg.suppressTokens);
  readIds("whisper.begin_suppress_tokens", cfg.beginSuppressTokens);

  if (!cfg.valid()) {
    error = "whisper metadata is missing or inconsistent (d_model " + std::to_string(cfg.dModel) +
            ", enc " + std::to_string(cfg.encLayers) + "x" + std::to_string(cfg.encHeads) +
            ", dec " + std::to_string(cfg.decLayers) + "x" + std::to_string(cfg.decHeads) +
            ", vocab " + std::to_string(cfg.vocabSize) + ", text_token_end " +
            std::to_string(cfg.textTokenEnd) + ")";
  }
  return cfg;
}

std::optional<WhisperWeights> loadWhisperWeights(const gguf::GgufFile& file,
                                                 const WhisperConfig& cfg, std::string& error) {
  error.clear();
  if (!cfg.valid()) {
    error = "invalid whisper config";
    return std::nullopt;
  }
  const int d = static_cast<int>(cfg.dModel);
  const int mels = static_cast<int>(cfg.melBins);

  WhisperWeights w;
  // The stem: [d, mels*3] and [d, d*3]. The converter wrote Conv1d's [out, in, k] bytes unchanged,
  // so the flattened column index is in*3 + k, which is what the stem builds per output frame.
  if (!loadMat(file, "enc.conv1.weight", d, mels * 3, w.conv1, error) ||
      !loadVec(file, "enc.conv1.bias", d, w.conv1B, error) ||
      !loadMat(file, "enc.conv2.weight", d, d * 3, w.conv2, error) ||
      !loadVec(file, "enc.conv2.bias", d, w.conv2B, error) ||
      !loadMat(file, "enc.position_embd.weight", static_cast<int>(cfg.encCtx), d, w.encPos, error)) {
    return std::nullopt;
  }

  w.enc.resize(cfg.encLayers);
  for (int i = 0; i < static_cast<int>(cfg.encLayers); ++i) {
    WhisperEncoderLayer& L = w.enc[static_cast<std::size_t>(i)];
    if (!loadAttn(file, encBlk(i, "attn"), d, L.attn, error) ||
        !loadNorm(file, encBlk(i, "attn_norm"), d, L.attnNormW, L.attnNormB, error) ||
        !loadMat(file, encBlk(i, "ffn_up.weight"), static_cast<int>(cfg.encFfn), d, L.ffnUp, error) ||
        !loadVec(file, encBlk(i, "ffn_up.bias"), static_cast<int>(cfg.encFfn), L.ffnUpB, error) ||
        !loadMat(file, encBlk(i, "ffn_down.weight"), d, static_cast<int>(cfg.encFfn), L.ffnDown,
                 error) ||
        !loadVec(file, encBlk(i, "ffn_down.bias"), d, L.ffnDownB, error) ||
        !loadNorm(file, encBlk(i, "ffn_norm"), d, L.ffnNormW, L.ffnNormB, error)) {
      return std::nullopt;
    }
  }
  if (!loadNorm(file, "enc.output_norm", d, w.encNormW, w.encNormB, error)) return std::nullopt;

  if (!loadMat(file, "dec.token_embd.weight", static_cast<int>(cfg.vocabSize), d, w.tokenEmbd,
               error) ||
      !loadMat(file, "dec.position_embd.weight", static_cast<int>(cfg.decCtx), d, w.decPos, error)) {
    return std::nullopt;
  }

  w.dec.resize(cfg.decLayers);
  for (int i = 0; i < static_cast<int>(cfg.decLayers); ++i) {
    WhisperDecoderLayer& L = w.dec[static_cast<std::size_t>(i)];
    if (!loadAttn(file, decBlk(i, "attn"), d, L.attn, error) ||
        !loadNorm(file, decBlk(i, "attn_norm"), d, L.attnNormW, L.attnNormB, error) ||
        !loadAttn(file, decBlk(i, "cross_attn"), d, L.cross, error) ||
        !loadNorm(file, decBlk(i, "cross_attn_norm"), d, L.crossNormW, L.crossNormB, error) ||
        !loadMat(file, decBlk(i, "ffn_up.weight"), static_cast<int>(cfg.decFfn), d, L.ffnUp, error) ||
        !loadVec(file, decBlk(i, "ffn_up.bias"), static_cast<int>(cfg.decFfn), L.ffnUpB, error) ||
        !loadMat(file, decBlk(i, "ffn_down.weight"), d, static_cast<int>(cfg.decFfn), L.ffnDown,
                 error) ||
        !loadVec(file, decBlk(i, "ffn_down.bias"), d, L.ffnDownB, error) ||
        !loadNorm(file, decBlk(i, "ffn_norm"), d, L.ffnNormW, L.ffnNormB, error)) {
      return std::nullopt;
    }
  }
  if (!loadNorm(file, "dec.output_norm", d, w.decNormW, w.decNormB, error)) return std::nullopt;

  return w;
}

}  // namespace qorvix::audio
