#include "qorvix/runtime/multimodal.hpp"

#include <utility>

#include "qorvix/tokenizer/tokenizer.hpp"

namespace qorvix::runtime {

std::vector<std::string> splitOnImageMarker(std::string_view text) {
  std::vector<std::string> out;
  std::size_t start = 0;
  for (;;) {
    const std::size_t hit = text.find(kImageMarker, start);
    if (hit == std::string_view::npos) break;
    out.emplace_back(text.substr(start, hit - start));
    start = hit + kImageMarker.size();
  }
  out.emplace_back(text.substr(start));
  return out;
}

void MultimodalPrompt::addTokens(const std::vector<int>& ids) {
  if (ids.empty()) return;
  Chunk c;
  c.tokens = ids;
  chunks_.push_back(std::move(c));
  textIds_.insert(textIds_.end(), ids.begin(), ids.end());
  textTokens_ += static_cast<int>(ids.size());
}

bool MultimodalPrompt::addImage(const std::vector<float>& features, int tokens, int dim,
                                std::string& error) {
  if (dim != dModel_) {
    error = "projected image features are " + std::to_string(dim) + "-d but the decoder expects " +
            std::to_string(dModel_) + " — the mmproj file does not match this language model";
    return false;
  }
  if (tokens <= 0) {
    error = "image contributed no patch tokens";
    return false;
  }
  const std::size_t need = static_cast<std::size_t>(tokens) * dim;
  if (features.size() < need) {
    error = "image feature buffer holds " + std::to_string(features.size()) + " floats, need " +
            std::to_string(need);
    return false;
  }

  Chunk c;
  c.image = true;
  c.offset = features_.size();
  c.count = tokens;
  features_.insert(features_.end(), features.begin(), features.begin() + need);
  chunks_.push_back(std::move(c));
  imageTokens_ += tokens;
  ++imageCount_;
  return true;
}

std::vector<InputToken> MultimodalPrompt::steps() const {
  std::vector<InputToken> out;
  out.reserve(static_cast<std::size_t>(size()));
  for (const auto& c : chunks_) {
    if (!c.image) {
      for (int id : c.tokens) out.push_back(InputToken{id, nullptr});
      continue;
    }
    const float* base = features_.data() + c.offset;
    for (int i = 0; i < c.count; ++i) {
      out.push_back(InputToken{0, base + static_cast<std::size_t>(i) * dModel_});
    }
  }
  return out;
}

bool partsFromPrompt(std::string_view text, std::vector<PromptPart> images,
                     std::vector<PromptPart>& out, std::string& error) {
  error.clear();
  out.clear();

  // With no images there is nothing to splice, so the marker is ordinary text. This matters: a
  // text-only chat message that happens to mention "<image>" must not be rewritten, and must not
  // be rejected for a marker/image mismatch it never intended.
  if (images.empty()) {
    if (!text.empty()) out.push_back(PromptPart::fromText(std::string(text)));
    return true;
  }

  const auto segments = splitOnImageMarker(text);
  const std::size_t markers = segments.size() - 1;

  if (markers == 0) {
    for (auto& img : images) out.push_back(std::move(img));
    if (!segments[0].empty()) out.push_back(PromptPart::fromText(segments[0]));
    return true;
  }
  if (markers != images.size()) {
    error = "prompt has " + std::to_string(markers) + " " + std::string(kImageMarker) +
            " marker(s) but " + std::to_string(images.size()) + " image(s) were supplied";
    return false;
  }
  for (std::size_t i = 0; i < segments.size(); ++i) {
    if (!segments[i].empty()) out.push_back(PromptPart::fromText(segments[i]));
    if (i < images.size()) out.push_back(std::move(images[i]));
  }
  return true;
}

bool buildPrompt(const std::vector<PromptPart>& parts, const tokenizer::Tokenizer& tok, bool addBos,
                 MultimodalPrompt& out, std::string& error) {
  error.clear();
  // BOS and EOS bracket the SEQUENCE, not each segment. Emitting BOS up front (rather than folding
  // it into the first text part) also keeps it at position 0 when the prompt LEADS with an image,
  // which is the common LLaVA shape. It goes in as a bare id rather than through encode(""),
  // because SentencePiece prefixes a ▁ to whatever it is handed — encoding the empty string would
  // smuggle a real token in behind BOS.
  const bool addEos = tok.special().addEos;
  if (addBos && tok.special().bos >= 0) out.addTokens({tok.special().bos});

  for (const auto& p : parts) {
    if (p.isImage()) {
      if (!out.addImage(p.imageFeatures, p.imageTokens, p.imageDim, error)) return false;
      continue;
    }
    if (p.text.empty()) continue;
    out.addTokens(tok.encode(p.text, /*addBos=*/false, /*addEos=*/false));
  }

  if (addEos && tok.special().eos >= 0) out.addTokens({tok.special().eos});
  return true;
}

}  // namespace qorvix::runtime
