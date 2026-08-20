#include "qorvix/image/diffusion.hpp"

#include <chrono>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/image/rng.hpp"

namespace qorvix::image {

StableDiffusion::StableDiffusion(SdConfig cfg, SdWeights weights, ClipTokenizer tok,
                                 std::unique_ptr<gguf::GgufFile> file)
    : cfg_(std::move(cfg)),
      w_(std::make_unique<SdWeights>(std::move(weights))),
      tok_(std::make_unique<ClipTokenizer>(std::move(tok))),
      file_(std::move(file)) {
  text_ = std::make_unique<TextEncoder>(cfg_.text, w_->text);
  unet_ = std::make_unique<Unet>(cfg_.unet, w_->unet);
  vae_ = std::make_unique<VaeDecoder>(cfg_, w_->vae);
}

StableDiffusion::StableDiffusion(StableDiffusion&&) noexcept = default;
StableDiffusion& StableDiffusion::operator=(StableDiffusion&&) noexcept = default;
StableDiffusion::~StableDiffusion() = default;

std::optional<StableDiffusion> StableDiffusion::fromGguf(gguf::GgufFile file, std::string& error) {
  const SdConfig cfg = sdConfigFromGguf(file, error);
  if (!error.empty()) return std::nullopt;
  auto tok = ClipTokenizer::fromGguf(file, error);
  if (!tok) return std::nullopt;
  if (tok->vocabSize() != cfg.text.vocab) {
    // The tokenizer and the embedding table have to be the same size or every id past the
    // boundary either reads garbage or is rejected — and a file carrying two halves of different
    // conversions is worth naming at load rather than at the first prompt that reaches into the
    // gap.
    error = "the file's vocabulary has " + std::to_string(tok->vocabSize()) +
            " tokens but its text encoder was built for " + std::to_string(cfg.text.vocab);
    return std::nullopt;
  }
  auto owned = std::make_unique<gguf::GgufFile>(std::move(file));
  auto weights = loadSdWeights(*owned, cfg, error);
  if (!weights) return std::nullopt;
  return StableDiffusion(cfg, std::move(*weights), std::move(*tok), std::move(owned));
}

std::optional<StableDiffusion> StableDiffusion::fromPath(const std::filesystem::path& path,
                                                         std::string& error) {
  error.clear();
  try {
    return fromGguf(gguf::GgufFile::open(path), error);
  } catch (const std::exception& e) {
    error = e.what();
    return std::nullopt;
  }
}

bool StableDiffusion::encodePrompt(const std::string& text, int clipSkip, std::vector<float>& out,
                                   bool& truncated, std::string& error) {
  const std::vector<int> ids = tok_->encodePadded(text, cfg_.text.contextLength, truncated);
  if (static_cast<int>(ids.size()) != cfg_.text.contextLength) {
    error = "the text encoder's context length is " + std::to_string(cfg_.text.contextLength) +
            ", which leaves no room for a prompt";
    return false;
  }
  return text_->encode(ids, clipSkip, out, error);
}

bool StableDiffusion::generate(const Options& opt, Result& out, std::string& error,
                               const std::function<void(int, int)>& onStep) {
  using clock = std::chrono::steady_clock;
  error.clear();
  out = Result{};

  const int mult = sizeMultiple();
  int width = opt.width > 0 ? opt.width : cfg_.defaultPixels();
  int height = opt.height > 0 ? opt.height : cfg_.defaultPixels();
  if (width % mult != 0 || height % mult != 0 || width <= 0 || height <= 0) {
    error = "size must be a positive multiple of " + std::to_string(mult) + " (got " +
            std::to_string(width) + "x" + std::to_string(height) + ")";
    return false;
  }

  auto scheduler = Scheduler::make(cfg_.scheduler, opt.sampler, opt.steps, error);
  if (!scheduler) return false;

  const bool guided = opt.guidance > 1.0f;
  const auto tPrompt0 = clock::now();
  std::vector<float> cond, uncond;
  if (!encodePrompt(opt.prompt, opt.clipSkip, cond, out.promptTruncated, error)) return false;
  if (guided && !encodePrompt(opt.negativePrompt, opt.clipSkip, uncond, out.negativeTruncated, error)) {
    return false;
  }
  out.promptSeconds = std::chrono::duration<double>(clock::now() - tPrompt0).count();

  GaussianRng rng(opt.seed);
  FeatureMap latent;
  latent.resize(cfg_.latentChannels, height / cfg_.vaeScale, width / cfg_.vaeScale);
  const float sigma0 = scheduler->initNoiseSigma();
  for (float& v : latent.data) v = rng.normal() * sigma0;

  FeatureMap scaled, predCond, predUncond;
  const auto tUnet0 = clock::now();
  for (int i = 0; i < scheduler->steps(); ++i) {
    const int t = scheduler->timesteps()[static_cast<std::size_t>(i)];
    scaled = latent;
    scheduler->scaleModelInput(scaled, i);

    if (!unet_->forward(scaled, t, cond, cfg_.text.contextLength, predCond, error)) return false;
    ++out.unetEvaluations;
    if (guided) {
      if (!unet_->forward(scaled, t, uncond, cfg_.text.contextLength, predUncond, error)) {
        return false;
      }
      ++out.unetEvaluations;
      for (std::size_t k = 0; k < predCond.size(); ++k) {
        predCond.data[k] = predUncond.data[k] + opt.guidance * (predCond.data[k] - predUncond.data[k]);
      }
    }

    if (!scheduler->step(latent, predCond, i, rng, error)) return false;
    if (onStep) onStep(i + 1, scheduler->steps());
  }
  out.unetSeconds = std::chrono::duration<double>(clock::now() - tUnet0).count();

  const auto tDecode0 = clock::now();
  if (!vae_->decodeToImage(latent, out.image, error)) return false;
  out.decodeSeconds = std::chrono::duration<double>(clock::now() - tDecode0).count();
  return true;
}

}  // namespace qorvix::image
