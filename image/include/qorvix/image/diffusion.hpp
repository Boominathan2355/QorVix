#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "qorvix/image/clip_tokenizer.hpp"
#include "qorvix/image/nn.hpp"
#include "qorvix/image/scheduler.hpp"
#include "qorvix/image/sd_weights.hpp"
#include "qorvix/image/text_encoder.hpp"
#include "qorvix/image/unet.hpp"
#include "qorvix/image/vae.hpp"
#include "qorvix/vision/image.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::image {

// Text to image: SPEC's "Prompt -> Latent Diffusion -> Image" stage, end to end on the CPU.
//
// The loop is four lines of arithmetic around three networks. Encode the prompt once. Start from
// noise. At every step, ask the UNet what noise it sees and hand the answer to the scheduler.
// Decode the result. Everything difficult is inside those three networks, and everything easy to
// get subtly wrong is in how they are wired together — which is why the pieces are separately
// gated rather than judged by whether the picture looks plausible.
//
// CLASSIFIER-FREE GUIDANCE is the reason a step costs two UNet evaluations rather than one. The
// model is run against the prompt and against a second, usually empty, prompt, and the difference
// between them is amplified:
//
//     prediction = unconditional + scale * (conditional - unconditional)
//
// At scale <= 1 the unconditional pass contributes nothing, so it is SKIPPED rather than computed
// and multiplied by zero — that halves the work, and it is what the distilled few-step models
// expect.
//
// Not thread-safe: the three networks keep scratch in members. `serve` gives it its own mutex for
// the same reason the CLIP tower and Whisper have theirs.
class StableDiffusion {
 public:
  static std::optional<StableDiffusion> fromPath(const std::filesystem::path& path,
                                                 std::string& error);
  static std::optional<StableDiffusion> fromGguf(gguf::GgufFile file, std::string& error);

  StableDiffusion(StableDiffusion&&) noexcept;
  StableDiffusion& operator=(StableDiffusion&&) noexcept;
  ~StableDiffusion();

  struct Options {
    std::string prompt;
    std::string negativePrompt;
    int steps = 20;
    float guidance = 7.5f;
    // 0 means the size the checkpoint was trained at. Anything else must be a multiple of
    // `sizeMultiple()` — the VAE's stride times the UNet's own downsampling — or the up path's
    // skip connections meet feature maps of different extents.
    int width = 0;
    int height = 0;
    std::uint64_t seed = 0;
    SamplerKind sampler = SamplerKind::Euler;
    int clipSkip = 1;
  };

  struct Result {
    vision::Image image;
    double promptSeconds = 0.0;
    double unetSeconds = 0.0;
    double decodeSeconds = 0.0;
    int unetEvaluations = 0;
    bool promptTruncated = false;
    bool negativeTruncated = false;
  };

  // `onStep(completed, total)` is called after each denoising step. A sample is minutes long on
  // this hardware, so silence is indistinguishable from a hang.
  bool generate(const Options& opt, Result& out, std::string& error,
                const std::function<void(int, int)>& onStep = {});

  // The prompt half on its own: tokenize, pad to the context length, run the text encoder.
  // Exposed because the gate compares conditioning before it compares pictures.
  bool encodePrompt(const std::string& text, int clipSkip, std::vector<float>& out, bool& truncated,
                    std::string& error);

  const SdConfig& config() const { return cfg_; }
  const ClipTokenizer& tokenizer() const { return *tok_; }
  TextEncoder& textEncoder() { return *text_; }
  Unet& unet() { return *unet_; }
  VaeDecoder& vae() { return *vae_; }

  // Pixel sizes must be a multiple of this: the VAE downsamples by `vaeScale`, and the UNet
  // halves the latent once per block boundary on top of that.
  int sizeMultiple() const { return cfg_.vaeScale * (1 << (cfg_.unet.blocks() - 1)); }

 private:
  StableDiffusion(SdConfig cfg, SdWeights weights, ClipTokenizer tok,
                  std::unique_ptr<gguf::GgufFile> file);

  SdConfig cfg_;
  std::unique_ptr<SdWeights> w_;
  std::unique_ptr<ClipTokenizer> tok_;
  std::unique_ptr<TextEncoder> text_;
  std::unique_ptr<Unet> unet_;
  std::unique_ptr<VaeDecoder> vae_;
  // Keeps the mmap alive behind every borrowed weight.
  std::unique_ptr<gguf::GgufFile> file_;
};

}  // namespace qorvix::image
