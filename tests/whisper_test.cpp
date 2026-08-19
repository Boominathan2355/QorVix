#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "qorvix/audio/whisper_model.hpp"
#include "qorvix/runtime/ops.hpp"
#include "qorvix/runtime/weights.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

using namespace qorvix::audio;
using qorvix::runtime::WeightMat;
using qorvix::tokenizer::SpecialTokens;
using qorvix::tokenizer::Tokenizer;
using qorvix::tokenizer::TokenizerModel;
namespace ops = qorvix::runtime::ops;

// Phase 11b-3b: the Whisper encoder/decoder. These pin what holds with NO model and no reference
// fixture — the convolutional stem's arithmetic, the protocol the decoder is prompted with, the two
// caches' lifetimes, and the refusals. `qorvix whisper-check` covers what only transformers can
// settle (that this is *Whisper* and not merely self-consistent).

namespace {

// Ten tokens is enough for the whole protocol: two text ids, then the specials in Whisper's own
// order, so that langFirst/langLast can be derived from the sot/translate anchors exactly as they
// are on a real vocabulary.
std::vector<std::string> multilingualVocab() {
  return {"a", "b", "<|endoftext|>", "<|startoftranscript|>", "<|en|>",
          "<|translate|>", "<|transcribe|>", "<|notimestamps|>", "<|0.00|>", "<|0.02|>"};
}

// The same, minus the language tokens: <|translate|> sits immediately after sot, which is how an
// English-only release looks and is what makes languageToken() unresolvable.
std::vector<std::string> englishOnlyVocab() {
  return {"a", "b", "<|endoftext|>", "<|startoftranscript|>", "<|translate|>", "<|transcribe|>",
          "<|notimestamps|>", "<|0.00|>"};
}

Tokenizer toyTok(std::vector<std::string> vocab) {
  SpecialTokens sp;
  sp.bos = 3;  // <|startoftranscript|>
  sp.eos = 2;  // <|endoftext|>
  sp.addBos = false;
  sp.addEos = false;
  return Tokenizer(TokenizerModel::Bpe, std::move(vocab), {}, {}, sp);
}

WhisperConfig toyConfig(int vocabSize, bool multilingual) {
  WhisperConfig c;
  c.name = "toy";
  c.dModel = 4;
  c.melBins = 2;
  c.vocabSize = static_cast<std::uint32_t>(vocabSize);
  c.encLayers = 1;
  c.encHeads = 2;
  c.encFfn = 4;
  c.encCtx = 4;  // so the mel window is 8 frames, and stride 2 has something to get wrong
  c.decLayers = 1;
  c.decHeads = 2;
  c.decFfn = 4;
  c.decCtx = 8;
  c.normEpsilon = 1e-5f;
  c.multilingual = multilingual;
  c.textTokenEnd = 2;  // <|endoftext|>
  return c;
}

std::vector<float> ramp(int rows, int cols, float scale, float shift = 3.0f) {
  std::vector<float> v(static_cast<std::size_t>(rows) * cols);
  for (std::size_t i = 0; i < v.size(); ++i) {
    v[i] = scale * (static_cast<float>(i % 7) - shift);
  }
  return v;
}

// `neutralBlocks` zeroes the two output projections, which makes every transformer block an exact
// identity (pre-norm plus a zero residual contribution) without making the stem trivial. That is
// what lets a test compare the stem against a hand-written reference: with the blocks in the way
// there is nothing to compare against short of reimplementing the whole encoder.
WhisperWeights toyWeights(const WhisperConfig& c, bool neutralBlocks, bool signedHead) {
  const int d = static_cast<int>(c.dModel);
  const int mels = static_cast<int>(c.melBins);
  const int encFfn = static_cast<int>(c.encFfn);
  const int decFfn = static_cast<int>(c.decFfn);
  const int vocab = static_cast<int>(c.vocabSize);
  const std::vector<float> zerosD(d, 0.0f);
  const std::vector<float> onesD(d, 1.0f);

  auto attn = [&](float base) {
    WhisperAttnWeights a;
    a.wq = WeightMat::f32(ramp(d, d, base), d, d);
    a.wk = WeightMat::f32(ramp(d, d, base + 0.01f), d, d);
    a.wv = WeightMat::f32(ramp(d, d, base + 0.02f), d, d);
    a.wo = neutralBlocks ? WeightMat::f32(std::vector<float>(static_cast<std::size_t>(d) * d, 0.0f),
                                          d, d)
                         : WeightMat::f32(ramp(d, d, base + 0.03f), d, d);
    a.bq.assign(d, 0.05f);
    a.bv.assign(d, -0.02f);
    a.bo.assign(d, 0.0f);
    return a;
  };

  WhisperWeights w;
  w.conv1 = WeightMat::f32(ramp(d, mels * 3, 0.11f), d, mels * 3);
  w.conv1B.assign(d, 0.07f);
  w.conv2 = WeightMat::f32(ramp(d, d * 3, 0.09f), d, d * 3);
  w.conv2B.assign(d, -0.04f);
  w.encPos = WeightMat::f32(ramp(static_cast<int>(c.encCtx), d, 0.13f),
                            static_cast<int>(c.encCtx), d);

  WhisperEncoderLayer enc;
  enc.attn = attn(0.10f);
  enc.attnNormW = onesD;
  enc.attnNormB = zerosD;
  enc.ffnUp = WeightMat::f32(ramp(encFfn, d, 0.08f), encFfn, d);
  enc.ffnUpB.assign(encFfn, 0.01f);
  enc.ffnDown = neutralBlocks
                    ? WeightMat::f32(std::vector<float>(static_cast<std::size_t>(d) * encFfn, 0.0f),
                                     d, encFfn)
                    : WeightMat::f32(ramp(d, encFfn, 0.06f), d, encFfn);
  enc.ffnDownB = zerosD;
  enc.ffnNormW = onesD;
  enc.ffnNormB = zerosD;
  w.enc = {enc};
  w.encNormW = onesD;
  w.encNormB = zerosD;

  // The LM head is the token embedding. `signedHead` makes row 0 and row 1 exact negations of each
  // other and row 2 (<|endoftext|>) all zeros, so max(logit0, logit1) = |dot| >= 0 = logit(eot):
  // greedy decoding then provably never picks <|endoftext|>, which is what lets a token-limit test
  // be deterministic instead of dependent on the toy weights.
  std::vector<float> emb(static_cast<std::size_t>(vocab) * d);
  for (std::size_t i = 0; i < emb.size(); ++i) emb[i] = 0.1f * static_cast<float>((i % 5) + 1);
  if (signedHead) {
    for (int j = 0; j < d; ++j) {
      const float v = 0.2f * static_cast<float>(j + 1);
      emb[static_cast<std::size_t>(j)] = v;                                  // token 0
      emb[static_cast<std::size_t>(d) + j] = -v;                             // token 1
      emb[static_cast<std::size_t>(2) * d + j] = 0.0f;                       // <|endoftext|>
    }
  }
  w.tokenEmbd = WeightMat::f32(std::move(emb), vocab, d);
  w.decPos = WeightMat::f32(ramp(static_cast<int>(c.decCtx), d, 0.05f),
                            static_cast<int>(c.decCtx), d);

  WhisperDecoderLayer dec;
  dec.attn = attn(0.12f);
  dec.attnNormW = onesD;
  dec.attnNormB = zerosD;
  dec.cross = attn(0.14f);
  dec.crossNormW = onesD;
  dec.crossNormB = zerosD;
  dec.ffnUp = WeightMat::f32(ramp(decFfn, d, 0.07f), decFfn, d);
  dec.ffnUpB.assign(decFfn, 0.02f);
  dec.ffnDown = neutralBlocks
                    ? WeightMat::f32(std::vector<float>(static_cast<std::size_t>(d) * decFfn, 0.0f),
                                     d, decFfn)
                    : WeightMat::f32(ramp(d, decFfn, 0.05f), d, decFfn);
  dec.ffnDownB = zerosD;
  dec.ffnNormW = onesD;
  dec.ffnNormB = zerosD;
  w.dec = {dec};
  w.decNormW = onesD;
  w.decNormB = zerosD;
  return w;
}

WhisperModel toyModel(bool multilingual = true, bool neutralBlocks = false,
                      bool signedHead = false) {
  const std::vector<std::string> vocab =
      multilingual ? multilingualVocab() : englishOnlyVocab();
  WhisperConfig cfg = toyConfig(static_cast<int>(vocab.size()), multilingual);
  WhisperWeights w = toyWeights(cfg, neutralBlocks, signedHead);
  return WhisperModel(cfg, std::move(w), toyTok(vocab));
}

// A [melBins x frames] window with distinct values per (bin, frame) — a constant one would make a
// padding or stride error invisible.
std::vector<float> toyMel(int mels, int frames, float seed = 0.0f) {
  std::vector<float> mel(static_cast<std::size_t>(mels) * frames);
  for (int c = 0; c < mels; ++c) {
    for (int t = 0; t < frames; ++t) {
      mel[static_cast<std::size_t>(c) * frames + t] =
          std::sin(0.7f * static_cast<float>(t) + 1.3f * static_cast<float>(c) + seed);
    }
  }
  return mel;
}

}  // namespace

TEST_CASE("the convolutional stem pads, strides and flattens as Whisper's does", "[whisper]") {
  WhisperModel model = toyModel(/*multilingual=*/true, /*neutralBlocks=*/true);
  const WhisperConfig& c = model.config();
  const int d = static_cast<int>(c.dModel);
  const int mels = static_cast<int>(c.melBins);
  const int n = static_cast<int>(c.encCtx);
  const int frames = n * 2;
  const std::vector<float> mel = toyMel(mels, frames);

  std::string err;
  REQUIRE(model.encode(mel, err));
  REQUIRE(err.empty());

  // The reference: two Conv1d layers (kernel 3, padding 1; the second with stride 2), each followed
  // by exact GELU, then the sinusoidal position rows, then the final LayerNorm. Written out here so
  // a padding, stride or column-order mistake in the model shows up as a numeric difference.
  const WhisperWeights& w = toyWeights(c, /*neutralBlocks=*/true, /*signedHead=*/false);
  auto at = [](const WeightMat& m, int row, int col) {
    return m.owned[static_cast<std::size_t>(row) * m.cols + col];
  };
  std::vector<float> c1(static_cast<std::size_t>(frames) * d, 0.0f);
  for (int t = 0; t < frames; ++t) {
    for (int o = 0; o < d; ++o) {
      float acc = w.conv1B[static_cast<std::size_t>(o)];
      for (int ch = 0; ch < mels; ++ch) {
        for (int k = 0; k < 3; ++k) {
          const int src = t + k - 1;
          if (src < 0 || src >= frames) continue;
          acc += at(w.conv1, o, ch * 3 + k) * mel[static_cast<std::size_t>(ch) * frames + src];
        }
      }
      c1[static_cast<std::size_t>(t) * d + o] = ops::gelu(acc);
    }
  }
  std::vector<float> expect(static_cast<std::size_t>(n) * d, 0.0f);
  for (int p = 0; p < n; ++p) {
    for (int o = 0; o < d; ++o) {
      float acc = w.conv2B[static_cast<std::size_t>(o)];
      for (int ch = 0; ch < d; ++ch) {
        for (int k = 0; k < 3; ++k) {
          const int src = 2 * p + k - 1;
          if (src < 0 || src >= frames) continue;
          acc += at(w.conv2, o, ch * 3 + k) * c1[static_cast<std::size_t>(src) * d + ch];
        }
      }
      expect[static_cast<std::size_t>(p) * d + o] =
          ops::gelu(acc) + at(w.encPos, p, o);
    }
  }
  for (int p = 0; p < n; ++p) {
    float* row = expect.data() + static_cast<std::size_t>(p) * d;
    ops::layernorm(row, row, w.encNormW.data(), w.encNormB.data(), d, c.normEpsilon);
  }

  const std::vector<float>& got = model.encoderStates();
  REQUIRE(got.size() == expect.size());
  for (std::size_t i = 0; i < expect.size(); ++i) {
    REQUIRE_THAT(got[i], Catch::Matchers::WithinAbs(expect[i], 1e-5));
  }
}

TEST_CASE("the forced prefix follows Whisper's protocol", "[whisper]") {
  WhisperModel model = toyModel();
  std::string err;
  REQUIRE(model.validateSpecials(err));
  const WhisperModel::SpecialIds& ids = model.ids();
  REQUIRE(ids.sot == 3);
  REQUIRE(ids.eot == 2);
  // Derived from the anchors, not counted: exactly one language sits between sot and <|translate|>.
  REQUIRE(ids.langFirst == 4);
  REQUIRE(ids.langLast == 4);

  WhisperModel::TranscribeOptions opt;
  opt.language = "en";
  REQUIRE(model.promptTokens(opt, err) == std::vector<int>{3, 4, 6, 7});

  opt.timestamps = true;
  REQUIRE(model.promptTokens(opt, err) == std::vector<int>{3, 4, 6});

  opt.timestamps = false;
  opt.translate = true;
  REQUIRE(model.promptTokens(opt, err) == std::vector<int>{3, 4, 5, 7});

  opt.translate = false;
  opt.language = "de";  // not in this vocabulary
  REQUIRE(model.promptTokens(opt, err).empty());
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("<|de|>"));
}

TEST_CASE("an English-only model omits the language and task and refuses to translate",
          "[whisper]") {
  WhisperModel model = toyModel(/*multilingual=*/false);
  std::string err;
  REQUIRE(model.validateSpecials(err));
  REQUIRE(model.ids().langFirst == -1);

  WhisperModel::TranscribeOptions opt;
  REQUIRE(model.promptTokens(opt, err) == std::vector<int>{3, 6});

  opt.translate = true;
  REQUIRE(model.promptTokens(opt, err).empty());
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("English-only"));
}

TEST_CASE("language detection is restricted to the language ids", "[whisper]") {
  WhisperModel model = toyModel();
  std::string err;
  std::string code;
  // Before encode there is nothing to detect from, and guessing would be worse than refusing.
  REQUIRE_FALSE(model.detectLanguage(code, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("encode()"));

  REQUIRE(model.encode(toyMel(2, 8), err));
  REQUIRE(model.detectLanguage(code, err));
  // Only <|en|> exists here, so the argmax over the language range can only be that — which is the
  // point: an unrestricted argmax would have returned a text token instead.
  REQUIRE(code == "en");
}

TEST_CASE("a decoder step without audio is refused rather than run against zeros", "[whisper]") {
  WhisperModel model = toyModel();
  std::string err;
  REQUIRE_FALSE(model.step(3, 0, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("encode()"));
}

TEST_CASE("decoder positions must be filled in order", "[whisper]") {
  WhisperModel model = toyModel();
  std::string err;
  REQUIRE(model.encode(toyMel(2, 8), err));

  REQUIRE_FALSE(model.step(3, 1, err));  // skipping position 0 would leave a hole in the KV cache
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("in order"));
  REQUIRE(model.step(3, 0, err));
  REQUIRE_FALSE(model.step(0, 0, err));  // rewriting a filled position
  REQUIRE(model.step(0, 1, err));

  REQUIRE_FALSE(model.step(0, static_cast<int>(model.config().decCtx), err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("position table"));
  REQUIRE_FALSE(model.step(999, 2, err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("vocabulary"));
}

TEST_CASE("the decoder listens to the audio, and only to the audio it was given", "[whisper]") {
  WhisperModel model = toyModel();
  std::string err;

  REQUIRE(model.encode(toyMel(2, 8, /*seed=*/0.0f), err));
  model.resetDecoder();
  REQUIRE(model.step(3, 0, err));
  const std::vector<float> first = model.logits();

  // Reset and repeat: the cross-attention cache is written once per clip, so the same step must
  // reproduce itself exactly. (A cache accidentally APPENDED to per step drifts here.)
  model.resetDecoder();
  REQUIRE(model.step(3, 0, err));
  REQUIRE(model.logits() == first);

  // Different audio, same token and position: the logits must move. If cross-attention silently
  // read zeros — the failure this whole seam is exposed to — they would not.
  REQUIRE(model.encode(toyMel(2, 8, /*seed=*/1.7f), err));
  model.resetDecoder();
  REQUIRE(model.step(3, 0, err));
  bool moved = false;
  for (std::size_t i = 0; i < first.size(); ++i) {
    if (std::abs(first[i] - model.logits()[i]) > 1e-6f) moved = true;
  }
  REQUIRE(moved);
}

TEST_CASE("self-attention grows one row per token while cross-attention stays fixed", "[whisper]") {
  WhisperModel model = toyModel();
  std::string err;
  REQUIRE(model.encode(toyMel(2, 8), err));

  // Position 1's logits depend on position 0 having been seen (causal self-attention), so the same
  // token decoded at position 1 differs between "after sot" and "after sot, a".
  model.resetDecoder();
  REQUIRE(model.step(3, 0, err));
  REQUIRE(model.step(0, 1, err));
  const std::vector<float> afterSotA = model.logits();

  model.resetDecoder();
  REQUIRE(model.step(0, 0, err));
  REQUIRE(model.step(0, 1, err));
  bool differs = false;
  for (std::size_t i = 0; i < afterSotA.size(); ++i) {
    if (std::abs(afterSotA[i] - model.logits()[i]) > 1e-6f) differs = true;
  }
  REQUIRE(differs);
}

TEST_CASE("greedy decoding never emits a protocol token and reports the token limit", "[whisper]") {
  WhisperModel model = toyModel(/*multilingual=*/true, /*neutralBlocks=*/false,
                                /*signedHead=*/true);
  std::string err;
  WhisperModel::TranscribeOptions opt;
  opt.language = "en";
  opt.maxTokens = 3;

  WhisperModel::TranscribeResult res;
  REQUIRE(model.transcribe(toyMel(2, 8), opt, res, err));
  // The head is rigged so <|endoftext|> can never win the argmax, so this run can only end on the
  // bound — and must say so.
  REQUIRE(res.hitTokenLimit);
  REQUIRE(res.tokens.size() == 4 + 3);  // the four prefix tokens plus the budget
  REQUIRE(res.language == "en");
  REQUIRE_FALSE(res.languageDetected);
  for (std::size_t i = 4; i < res.tokens.size(); ++i) {
    REQUIRE(res.tokens[i] < model.config().textTokenEnd);
  }
  // The transcript is the text tokens only: none of the prefix survives into it.
  REQUIRE(res.text.find('<') == std::string::npos);
}

TEST_CASE("decodeText drops every special token", "[whisper]") {
  WhisperModel model = toyModel();
  // sot, language, task, notimestamps, "a", "b", a timestamp, then eot.
  REQUIRE(model.decodeText({3, 4, 6, 7, 0, 1, 8, 2}) == model.decodeText({0, 1}));
  REQUIRE(model.decodeText({3, 4, 6, 7, 2}).empty());
}

TEST_CASE("a vocabulary without the protocol tokens is refused, not worked around", "[whisper]") {
  // Same dimensions, but no <|0.00|>: nothing in the file says where the timestamps begin, so
  // timestamped decoding has no defined meaning.
  std::vector<std::string> vocab = {"a", "b", "<|endoftext|>", "<|startoftranscript|>", "<|en|>",
                                    "<|translate|>", "<|transcribe|>", "<|notimestamps|>"};
  WhisperConfig cfg = toyConfig(static_cast<int>(vocab.size()), true);
  WhisperModel model(cfg, toyWeights(cfg, false, false), toyTok(std::move(vocab)));
  std::string err;
  REQUIRE_FALSE(model.validateSpecials(err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("<|0.00|>"));
}

TEST_CASE("a text_token_end that disagrees with the vocabulary is refused", "[whisper]") {
  std::vector<std::string> vocab = multilingualVocab();
  WhisperConfig cfg = toyConfig(static_cast<int>(vocab.size()), true);
  cfg.textTokenEnd = 5;  // the file claims the specials start somewhere else
  WhisperModel model(cfg, toyWeights(cfg, false, false), toyTok(std::move(vocab)));
  std::string err;
  REQUIRE_FALSE(model.validateSpecials(err));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("text_token_end"));
}

TEST_CASE("whisper.cpp's legacy container is named rather than reported as bad magic", "[whisper]") {
  // The files the Hub advertises as "whisper GGUF" are this: ggml's magic, 0x67676d6c, which spells
  // "lmgg" on disk. "bad magic" is not an actionable error for someone holding one.
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "qorvix_whisper_lmgg_probe.bin";
  {
    std::ofstream out(path, std::ios::binary);
    out << "lmgg";
    out.write("\x9a\xca\x00\x00", 4);
  }
  std::string err;
  REQUIRE_FALSE(WhisperModel::fromPath(path, err).has_value());
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("ggml"));
  REQUIRE_THAT(err, Catch::Matchers::ContainsSubstring("convert_whisper_to_gguf.py"));
  std::filesystem::remove(path);

  std::string missing;
  REQUIRE_FALSE(WhisperModel::fromPath(std::filesystem::temp_directory_path() /
                                           "qorvix_no_such_whisper.gguf",
                                       missing)
                    .has_value());
  REQUIRE_THAT(missing, Catch::Matchers::ContainsSubstring("cannot open"));
}

TEST_CASE("whisper config validation catches an inconsistent header", "[whisper]") {
  WhisperConfig good = toyConfig(10, true);
  REQUIRE(good.valid());
  REQUIRE(good.encHeadDim() == 2);
  REQUIRE(good.decHeadDim() == 2);

  WhisperConfig c = good;
  c.encHeads = 3;  // 4 is not divisible by 3
  REQUIRE_FALSE(c.valid());

  c = good;
  c.textTokenEnd = 0;  // no specials boundary means no way to strip them from the transcript
  REQUIRE_FALSE(c.valid());

  c = good;
  c.textTokenEnd = static_cast<int>(c.vocabSize) + 1;
  REQUIRE_FALSE(c.valid());

  c = good;
  c.decLayers = 0;
  REQUIRE_FALSE(c.valid());
}
