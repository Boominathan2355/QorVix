#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/runtime/weights.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::audio {

// Hyperparameters of a Whisper model, read from a GGUF written by
// scripts/convert_whisper_to_gguf.py. There is no Whisper GGUF upstream (whisper.cpp never left
// its own container, and the Hub files calling themselves "whisper GGUF" are that container
// renamed — magic "lmgg", not "GGUF"), so this repo converts rather than adding a second reader.
//
// Encoder and decoder are sized SEPARATELY. They happen to match in every released Whisper — same
// width, same layer count, same head count — which is exactly why they are read as two sets of
// keys: a single `block_count` would work on every model that exists today and quietly halve or
// double the wrong stack on the first one that does not.
struct WhisperConfig {
  std::string name;
  std::uint32_t dModel = 0;
  std::uint32_t melBins = 80;
  std::uint32_t vocabSize = 0;

  std::uint32_t encLayers = 0, encHeads = 0, encFfn = 0;
  std::uint32_t encCtx = 1500;  // encoder positions == mel frames after the stride-2 conv
  std::uint32_t decLayers = 0, decHeads = 0, decFfn = 0;
  std::uint32_t decCtx = 448;  // learned position table length; the hard cap on transcript length

  float normEpsilon = 1e-5f;
  bool multilingual = true;
  // Every id from here up is a special token: <|endoftext|>, the languages, the task tokens, the
  // timestamps and the unused tail. One boundary is all the decoder needs to keep them out of the
  // transcript and out of the greedy argmax.
  int textTokenEnd = 0;

  // The model's own suppression lists, carried in the file. Whisper's greedy decode is not argmax
  // over the whole vocabulary: `suppressTokens` blocks ids the released models never emit as text
  // (punctuation-only and control tokens), and `beginSuppressTokens` additionally blocks a leading
  // space and an immediate <|endoftext|> at the first generated position. A runtime that ignores
  // them produces transcripts that differ from every other Whisper implementation by a token here
  // and there, which is exactly the kind of drift that reads as "close enough" and is not.
  std::vector<int> suppressTokens;
  std::vector<int> beginSuppressTokens;

  std::uint32_t encHeadDim() const { return encHeads ? dModel / encHeads : 0; }
  std::uint32_t decHeadDim() const { return decHeads ? dModel / decHeads : 0; }

  bool valid() const {
    return dModel && melBins && vocabSize && encLayers && encHeads && encFfn && decLayers &&
           decHeads && decFfn && encCtx && decCtx && dModel % encHeads == 0 &&
           dModel % decHeads == 0 && textTokenEnd > 0 &&
           textTokenEnd <= static_cast<int>(vocabSize);
  }
};

WhisperConfig whisperConfigFromGguf(const gguf::GgufFile& file, std::string& error);

// One attention block's projections, used for BOTH self- and cross-attention: the two differ in
// what they attend to, not in their parameter shapes.
//
// There is no `bk`. Whisper's k_proj has no bias — in self-attention and in cross-attention, in
// every size — while q, v and out all do. Modelling it as an optional-and-usually-empty vector
// would invite a loader that "helpfully" fills it with zeros; leaving the field out means the
// asymmetry is visible in the type.
struct WhisperAttnWeights {
  runtime::WeightMat wq, wk, wv, wo;  // [d, d] each
  std::vector<float> bq, bv, bo;      // [d] each
};

// Pre-norm, like the CLIP tower and unlike the BERT encoder: the LayerNorm feeds the sublayer and
// the residual is added to the unnormalized stream.
struct WhisperEncoderLayer {
  WhisperAttnWeights attn;
  std::vector<float> attnNormW, attnNormB;
  runtime::WeightMat ffnUp, ffnDown;  // [ffn, d] and [d, ffn] — the ordinary meaning of up/down
  std::vector<float> ffnUpB, ffnDownB;
  std::vector<float> ffnNormW, ffnNormB;
};

// A decoder block is an encoder block with a cross-attention sublayer wedged between the
// self-attention and the FFN, each with its own pre-norm.
struct WhisperDecoderLayer {
  WhisperAttnWeights attn;
  std::vector<float> attnNormW, attnNormB;
  WhisperAttnWeights cross;
  std::vector<float> crossNormW, crossNormB;
  runtime::WeightMat ffnUp, ffnDown;
  std::vector<float> ffnUpB, ffnDownB;
  std::vector<float> ffnNormW, ffnNormB;
};

struct WhisperWeights {
  // The convolutional stem, stored as matmul weights: a Conv1d whose kernel is flattened over
  // (in_channels x kernel_width) is one dot product per output channel per output frame. Shapes
  // are [d, melBins*3] and [d, d*3].
  runtime::WeightMat conv1, conv2;
  std::vector<float> conv1B, conv2B;

  runtime::WeightMat encPos;  // [encCtx, d] — sinusoidal in origin, but read from the file
  std::vector<WhisperEncoderLayer> enc;
  std::vector<float> encNormW, encNormB;

  runtime::WeightMat tokenEmbd;  // [vocab, d] — also the LM head, tied
  runtime::WeightMat decPos;     // [decCtx, d] — learned, unlike the encoder's
  std::vector<WhisperDecoderLayer> dec;
  std::vector<float> decNormW, decNormB;
};

// Borrows matmul weights straight from the file's mmap, so the file must outlive the result —
// WhisperModel guarantees that by owning it. Returns nullopt with `error` naming the tensor.
std::optional<WhisperWeights> loadWhisperWeights(const gguf::GgufFile& file,
                                                 const WhisperConfig& cfg, std::string& error);

}  // namespace qorvix::audio
