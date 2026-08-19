#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/audio/mel.hpp"
#include "qorvix/audio/whisper_weights.hpp"
#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/tokenizer/tokenizer.hpp"

namespace qorvix::audio {

// CPU Whisper — SPEC's "Audio → Mel Spectrogram → Encoder → Decoder → Text" stage, behind the
// log-mel front end that Phase 11b-3a gated.
//
// Two things here are new to this codebase and are called out where they occur:
//
//   * CROSS-ATTENTION. Every attention so far read its keys and values from the same stream as its
//     queries. A decoder block's middle sublayer reads them from the ENCODER's output instead,
//     which means two caches with opposite lifetimes: the cross-attention K/V are computed once
//     per audio clip and never grow, while the self-attention K/V grow one row per generated
//     token. Conflating them (one cache, one length counter) is the obvious bug and it does not
//     crash — it truncates what the decoder can hear.
//   * THE TOKENIZER LIVES INSIDE THE MODEL. For a chat LM the caller owns the tokenizer because it
//     also renders the chat template; Whisper's tokens are internal protocol (a task, a language,
//     a timestamp mode) and the caller's unit of work is "audio in, text out". Handing a caller
//     the vocabulary would only let it assemble that prefix wrongly.
//
// Not thread-safe: per-call scratch lives in members so it is allocated once. Callers sharing an
// instance must serialize it, as `serve` does for the other engines.
class WhisperModel {
 public:
  // Ids of the protocol tokens, resolved by NAME from the file's own vocabulary. Looking them up
  // rather than hardcoding 50258/50257/… means a differently-sized vocabulary cannot silently
  // shift the whole protocol by one; a missing name is an error at load, not a wrong token later.
  struct SpecialIds {
    int sot = -1;             // <|startoftranscript|>
    int eot = -1;             // <|endoftext|>
    int transcribe = -1;      // <|transcribe|>
    int translate = -1;       // <|translate|>
    int noTimestamps = -1;    // <|notimestamps|>
    int timestampBegin = -1;  // <|0.00|>; each subsequent id is +0.02 s
    int noSpeech = -1;        // <|nocaptions|> on some releases, <|nospeech|> on others
    // Languages occupy the ids strictly between sot and <|translate|>, so the range is derived
    // from those two anchors instead of from a hardcoded count of languages.
    int langFirst = -1;
    int langLast = -1;
    bool valid() const { return sot >= 0 && eot >= 0 && timestampBegin > 0; }
  };

  struct TranscribeOptions {
    std::string language;      // empty => detect from the audio (multilingual models only)
    bool translate = false;    // <|translate|> instead of <|transcribe|>
    bool timestamps = false;   // emit timestamp tokens and report segments
    int maxTokens = 0;         // 0 => bounded by the decoder's position table
  };

  struct Segment {
    float start = 0.0f;
    float end = 0.0f;
    std::string text;
  };

  struct TranscribeResult {
    std::string text;
    std::string language;           // the code actually used
    bool languageDetected = false;  // true when it came from the audio rather than the caller
    std::vector<int> tokens;        // the full stream, specials included
    std::vector<Segment> segments;  // populated only with timestamps enabled
    bool hitTokenLimit = false;     // stopped on the bound rather than on <|endoftext|>
  };

  static std::optional<WhisperModel> fromGguf(gguf::GgufFile file, std::string& error);
  // Same, but opens the path and — when it is not a GGUF at all — names the case. Files served as
  // "whisper GGUF" are usually whisper.cpp's legacy ggml container (magic "lmgg"), and "bad magic"
  // is not an actionable error for someone holding one.
  static std::optional<WhisperModel> fromPath(const std::filesystem::path& path,
                                              std::string& error);

  WhisperModel(WhisperConfig cfg, WhisperWeights weights, tokenizer::Tokenizer tok);
  WhisperModel(WhisperModel&&) noexcept = default;
  WhisperModel& operator=(WhisperModel&&) noexcept = default;

  // ---- encoder ----
  // `logMel` is [melBins x frames] row-major by mel bin, exactly as logMelSpectrogram() writes it.
  // Frames must be 2*encCtx: the stem's second convolution has stride 2, so a 3000-frame window
  // becomes 1500 encoder positions, and the position table has no rows beyond that.
  bool encode(const std::vector<float>& logMel, std::string& error);
  bool encoded() const { return encoded_; }
  // [encCtx * dModel], row-major by position. Also the source of the cross-attention K/V.
  const std::vector<float>& encoderStates() const { return encStates_; }

  // ---- decoder ----
  void resetDecoder();
  // One autoregressive step: `token` at absolute position `pos`, which must be the next unfilled
  // position. Fills logits(). Requires encode() first — the cross-attention has nothing to read
  // otherwise, and silently attending to zeros would produce fluent text unrelated to the audio.
  bool step(int token, int pos, std::string& error);
  const std::vector<float>& logits() const { return logits_; }

  // The forced prefix: sot, then language and task on a multilingual model, then <|notimestamps|>
  // unless timestamps were asked for.
  std::vector<int> promptTokens(const TranscribeOptions& opt, std::string& error) const;

  // Greedy decode over the model's own suppression lists. Requires encode() to have run.
  bool generate(const TranscribeOptions& opt, TranscribeResult& out, std::string& error);
  // encode() + generate(), which is what the CLI and the HTTP route call.
  bool transcribe(const std::vector<float>& logMel, const TranscribeOptions& opt,
                  TranscribeResult& out, std::string& error);

  // Runs one step from sot alone and takes the argmax over the language ids — Whisper's own
  // detection procedure, not a heuristic. Leaves the decoder reset for the real decode.
  bool detectLanguage(std::string& code, std::string& error);

  // The protocol tokens are resolved by the constructor, because they are a pure function of the
  // vocabulary; this reports whether the vocabulary actually carried them. fromGguf() refuses a
  // file that fails it, and a directly-constructed model (tests) can check the same thing.
  bool validateSpecials(std::string& error) const;

  const WhisperConfig& config() const { return cfg_; }
  const SpecialIds& ids() const { return ids_; }
  const tokenizer::Tokenizer& tokenizer() const { return tok_; }
  MelConfig melConfig() const;
  // Language code ("en") <-> token id, using the file's vocabulary. -1 / empty when unknown.
  int languageToken(const std::string& code) const;
  std::string languageCode(int token) const;
  // Text of a decoded token stream with every special token dropped.
  std::string decodeText(const std::vector<int>& tokens) const;

 private:
  void encoderAttention(const WhisperEncoderLayer& L, int n);
  void selfAttention(const WhisperDecoderLayer& L, int layer, int pos);
  void crossAttention(const WhisperDecoderLayer& L, int layer);
  void resolveSpecials();
  // -inf on everything this position may not emit, then argmax.
  int pickToken(bool firstGenerated, bool timestamps);

  WhisperConfig cfg_;
  WhisperWeights w_;
  tokenizer::Tokenizer tok_;
  std::unique_ptr<gguf::GgufFile> file_;  // keeps the mmap alive behind the borrowed weights
  SpecialIds ids_;

  bool encoded_ = false;
  int decLen_ = 0;  // filled self-attention cache rows

  // Encoder scratch.
  std::vector<float> encStates_, encNorm_, encQ_, encK_, encV_, encAttn_, encTmp_, encFfn_,
      encScores_, convCols_, conv1Out_;
  // Decoder scratch. crossK_/crossV_ are [layers][encCtx * d], written once per encode();
  // selfK_/selfV_ are [layers][decCtx * d], written one row per step.
  std::vector<std::vector<float>> crossK_, crossV_, selfK_, selfV_;
  std::vector<float> x_, dNorm_, dQ_, dK_, dV_, dAttn_, dTmp_, dFfn_, dScores_, logits_;
};

}  // namespace qorvix::audio
