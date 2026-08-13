#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "qorvix/runtime/inference_engine.hpp"

namespace qorvix::tokenizer {
class Tokenizer;
}

// Assembly of a prompt that mixes text tokens with precomputed input embeddings (Phase 11b-2).
//
// This lives in `runtime`, not `vision`, on purpose: it manipulates InputToken sequences and knows
// nothing about images beyond "someone handed me d_model-wide vectors". That keeps the scheduler —
// which must prefill such a sequence — off any dependency on the vision tower, and lets a future
// audio encoder feed the same path without a second assembler.
namespace qorvix::runtime {

// The placeholder LLaVA-family prompts use to mark where image features belong. Callers may write
// it into the chat text; splitOnImageMarker() finds it.
inline constexpr std::string_view kImageMarker = "<image>";

// Splits `text` on every occurrence of kImageMarker. Always returns markers+1 segments (possibly
// empty ones), so segment i is the text that precedes image i and the last segment is the tail.
std::vector<std::string> splitOnImageMarker(std::string_view text);

// A prefill sequence of token ids interleaved with image-feature blocks.
//
// Each segment's text is tokenized SEPARATELY by the caller and handed over with addTokens. That
// is a real (small) difference from tokenizing the whole prompt at once — a BPE/SPM merge that
// would have spanned the marker cannot form — and it is unavoidable: the image occupies positions
// in the middle of the sequence, so there is no single string to tokenize. llama.cpp splits the
// same way.
class MultimodalPrompt {
 public:
  explicit MultimodalPrompt(int dModel) : dModel_(dModel) {}

  void addTokens(const std::vector<int>& ids);

  // `features` is [tokens, dim] row-major, already projected into the decoder's INPUT space (the
  // LLaVA mlp projector's output). Fails when dim != dModel — a mismatched projector would
  // otherwise read past the row and generate confident nonsense.
  bool addImage(const std::vector<float>& features, int tokens, int dim, std::string& error);

  // The prefill sequence, rebuilt on demand. The embedding pointers reference this object's own
  // storage, so the returned steps stay valid only while the prompt is alive AND unmodified —
  // build the whole prompt first, then call this.
  std::vector<InputToken> steps() const;

  // Token ids only, in sequence order. This is the sampler's repetition-penalty history: image
  // positions contribute nothing to it because they have no id that could be penalized.
  const std::vector<int>& textIds() const { return textIds_; }

  int size() const { return textTokens_ + imageTokens_; }
  int textTokens() const { return textTokens_; }
  int imageTokens() const { return imageTokens_; }
  int imageCount() const { return imageCount_; }
  int dModel() const { return dModel_; }
  bool empty() const { return size() == 0; }
  bool hasImages() const { return imageCount_ > 0; }

 private:
  struct Chunk {
    bool image = false;
    std::vector<int> tokens;  // text chunk
    std::size_t offset = 0;   // image chunk: start index into features_
    int count = 0;            // image chunk: number of patch tokens
  };

  int dModel_;
  std::vector<Chunk> chunks_;
  std::vector<float> features_;  // every image's features, concatenated
  std::vector<int> textIds_;
  int textTokens_ = 0, imageTokens_ = 0, imageCount_ = 0;
};

// One piece of a multimodal prompt: either text, or one image's projected features. Parts are
// consumed in order, so [text, image, text] puts the image between the two text spans — which is
// what the LLaVA prompt format wants.
struct PromptPart {
  std::string text;

  // [imageTokens, imageDim] row-major, already projected into the DECODER's input space by
  // ClipVisionModel::encodeProjected.
  std::vector<float> imageFeatures;
  int imageTokens = 0;
  int imageDim = 0;

  bool isImage() const { return imageTokens > 0; }

  static PromptPart fromText(std::string t) {
    PromptPart p;
    p.text = std::move(t);
    return p;
  }
  static PromptPart fromImage(std::vector<float> features, int tokens, int dim) {
    PromptPart p;
    p.imageFeatures = std::move(features);
    p.imageTokens = tokens;
    p.imageDim = dim;
    return p;
  }
};

// Interleaves `text` with `images` at each kImageMarker, producing the ordered part list.
//
// When the text contains no marker the images are PREPENDED, which is the LLaVA convention for a
// bare prompt (the image is context for everything that follows). A marker/image count mismatch is
// an error rather than a silent drop — losing an image quietly produces fluent answers about a
// picture the model never saw.
bool partsFromPrompt(std::string_view text, std::vector<PromptPart> images,
                     std::vector<PromptPart>& out, std::string& error);

// Assembles ordered parts into a prefill sequence. Text parts are tokenized here; BOS goes on the
// first text part only, since it belongs to the sequence rather than to each segment.
bool buildPrompt(const std::vector<PromptPart>& parts, const tokenizer::Tokenizer& tok, bool addBos,
                 MultimodalPrompt& out, std::string& error);

}  // namespace qorvix::runtime
