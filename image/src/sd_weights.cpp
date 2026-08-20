#include "qorvix/image/sd_weights.hpp"

#include <algorithm>

#include "qorvix/gguf/gguf_file.hpp"
#include "qorvix/runtime/tensor_load.hpp"

namespace qorvix::image {

namespace rt = qorvix::runtime;
using rt::detail::loadMat;
using rt::detail::loadVec;

namespace {

// A convolution is [out, k*k*in] after the converter's flattening; a linear layer is [out, in],
// which is the same call with k = 1.
bool loadConv(const gguf::GgufFile& f, const std::string& name, int outC, int inC, int kernel,
              MatWeights& out, std::string& error) {
  return loadMat(f, name + ".weight", outC, kernel * kernel * inC, out.w, error) &&
         loadVec(f, name + ".bias", outC, out.b, error);
}

bool loadLinear(const gguf::GgufFile& f, const std::string& name, int outC, int inC,
                MatWeights& out, std::string& error) {
  return loadConv(f, name, outC, inC, 1, out, error);
}

// q/k/v in the UNet's attention blocks: weight only. There is deliberately no bias field to fill,
// so a checkpoint that grew one would fail the converter's `unused` check rather than have it
// silently ignored here.
bool loadProj(const gguf::GgufFile& f, const std::string& name, int outC, int inC,
              rt::WeightMat& out, std::string& error) {
  return loadMat(f, name + ".weight", outC, inC, out, error);
}

bool loadNorm(const gguf::GgufFile& f, const std::string& name, int n, NormWeights& out,
              std::string& error) {
  return loadVec(f, name + ".weight", n, out.w, error) &&
         loadVec(f, name + ".bias", n, out.b, error);
}

bool loadResnet(const gguf::GgufFile& f, const std::string& name, int inC, int outC,
                int timeEmbedDim, ResnetWeights& out, std::string& error) {
  if (!loadNorm(f, name + ".norm1", inC, out.norm1, error) ||
      !loadConv(f, name + ".conv1", outC, inC, 3, out.conv1, error)) {
    return false;
  }
  // timeEmbedDim == 0 marks the VAE's resnets, which have nothing to condition on.
  if (timeEmbedDim > 0 && !loadLinear(f, name + ".time_emb", outC, timeEmbedDim, out.timeEmb, error)) {
    return false;
  }
  if (!loadNorm(f, name + ".norm2", outC, out.norm2, error) ||
      !loadConv(f, name + ".conv2", outC, outC, 3, out.conv2, error)) {
    return false;
  }
  // The shortcut exists exactly when the channel count changes. Deriving its presence from the
  // channel counts rather than from the file's tensor list means a checkpoint that disagrees is a
  // missing-tensor error naming the block, not a silently skipped projection.
  if (inC != outC && !loadConv(f, name + ".skip", outC, inC, 1, out.skip, error)) return false;
  return true;
}

bool loadSpatialTransformer(const gguf::GgufFile& f, const std::string& name, int channels,
                            int heads, int depth, int crossDim, SpatialTransformerWeights& out,
                            std::string& error) {
  if (!loadNorm(f, name + ".norm", channels, out.norm, error) ||
      !loadLinear(f, name + ".proj_in", channels, channels, out.projIn, error)) {
    return false;
  }
  out.blocks.resize(static_cast<std::size_t>(depth));
  for (int d = 0; d < depth; ++d) {
    auto& b = out.blocks[static_cast<std::size_t>(d)];
    const std::string p = name + ".tf." + std::to_string(d) + ".";
    // diffusers' FeedForward multiplies the model dimension by 4; it is not a UNet config knob,
    // so it is written here rather than read. A checkpoint that disagreed would fail loadMat with
    // an element-count error naming this exact tensor.
    const int inner = channels * 4;
    if (!loadNorm(f, p + "ln1", channels, b.ln1, error) ||
        !loadProj(f, p + "attn1_q", channels, channels, b.q1, error) ||
        !loadProj(f, p + "attn1_k", channels, channels, b.k1, error) ||
        !loadProj(f, p + "attn1_v", channels, channels, b.v1, error) ||
        !loadLinear(f, p + "attn1_out", channels, channels, b.out1, error) ||
        !loadNorm(f, p + "ln2", channels, b.ln2, error) ||
        !loadProj(f, p + "attn2_q", channels, channels, b.q2, error) ||
        !loadProj(f, p + "attn2_k", channels, crossDim, b.k2, error) ||
        !loadProj(f, p + "attn2_v", channels, crossDim, b.v2, error) ||
        !loadLinear(f, p + "attn2_out", channels, channels, b.out2, error) ||
        !loadNorm(f, p + "ln3", channels, b.ln3, error) ||
        !loadLinear(f, p + "ffn_geglu", 2 * inner, channels, b.geglu, error) ||
        !loadLinear(f, p + "ffn_out", channels, inner, b.ffnOut, error)) {
      return false;
    }
  }
  (void)heads;
  return loadLinear(f, name + ".proj_out", channels, channels, out.projOut, error);
}

bool loadUnet(const gguf::GgufFile& f, const SdConfig& cfg, UnetWeights& out, std::string& error) {
  const SdUnetConfig& u = cfg.unet;
  const int n = u.blocks();
  const int ch0 = u.channels.front();

  if (!loadConv(f, "unet.conv_in", ch0, u.inChannels, 3, out.convIn, error) ||
      !loadLinear(f, "unet.time_mlp.0", u.timeEmbedDim, ch0, out.timeMlp0, error) ||
      !loadLinear(f, "unet.time_mlp.1", u.timeEmbedDim, u.timeEmbedDim, out.timeMlp1, error)) {
    return false;
  }

  out.down.resize(static_cast<std::size_t>(n));
  int prev = ch0;
  for (int i = 0; i < n; ++i) {
    auto& blk = out.down[static_cast<std::size_t>(i)];
    const int outC = u.channels[static_cast<std::size_t>(i)];
    const int depth = u.transformerDepth[static_cast<std::size_t>(i)];
    blk.resnets.resize(static_cast<std::size_t>(u.layersPerBlock));
    if (depth > 0) blk.attns.resize(static_cast<std::size_t>(u.layersPerBlock));
    for (int j = 0; j < u.layersPerBlock; ++j) {
      const int inC = j == 0 ? prev : outC;
      const std::string p = "unet.down." + std::to_string(i) + ".";
      if (!loadResnet(f, p + "resnet." + std::to_string(j), inC, outC, u.timeEmbedDim,
                      blk.resnets[static_cast<std::size_t>(j)], error)) {
        return false;
      }
      if (depth > 0 &&
          !loadSpatialTransformer(f, p + "attn." + std::to_string(j), outC,
                                  u.headCounts[static_cast<std::size_t>(i)], depth, u.crossDim,
                                  blk.attns[static_cast<std::size_t>(j)], error)) {
        return false;
      }
    }
    if (i < n - 1 &&
        !loadConv(f, "unet.down." + std::to_string(i) + ".downsample", outC, outC, 3,
                  blk.downsample, error)) {
      return false;
    }
    prev = outC;
  }

  const int mid = u.channels.back();
  if (!loadResnet(f, "unet.mid.resnet.0", mid, mid, u.timeEmbedDim, out.midResnet0, error) ||
      !loadSpatialTransformer(f, "unet.mid.attn", mid, u.headCounts.back(), u.midTransformerDepth,
                              u.crossDim, out.midAttn, error) ||
      !loadResnet(f, "unet.mid.resnet.1", mid, mid, u.timeEmbedDim, out.midResnet1, error)) {
    return false;
  }

  // Up blocks walk the channel list backwards, and each resnet's input is the previous activation
  // concatenated with one saved skip. The skip's width is the down-block output it came from —
  // which is `outC` for all but the LAST resnet of a block, where it is the coarser block's
  // input. Getting that one entry wrong shifts every skip by one and still runs.
  std::vector<int> rev(u.channels.rbegin(), u.channels.rend());
  out.up.resize(static_cast<std::size_t>(n));
  int prevOut = rev.front();
  for (int i = 0; i < n; ++i) {
    auto& blk = out.up[static_cast<std::size_t>(i)];
    const int outC = rev[static_cast<std::size_t>(i)];
    const int inC = rev[static_cast<std::size_t>(std::min(i + 1, n - 1))];
    const int depth = u.transformerDepth[static_cast<std::size_t>(n - 1 - i)];
    const int layers = u.layersPerBlock + 1;
    blk.resnets.resize(static_cast<std::size_t>(layers));
    if (depth > 0) blk.attns.resize(static_cast<std::size_t>(layers));
    for (int j = 0; j < layers; ++j) {
      const int skipC = (j == layers - 1) ? inC : outC;
      const int resIn = (j == 0 ? prevOut : outC) + skipC;
      const std::string p = "unet.up." + std::to_string(i) + ".";
      if (!loadResnet(f, p + "resnet." + std::to_string(j), resIn, outC, u.timeEmbedDim,
                      blk.resnets[static_cast<std::size_t>(j)], error)) {
        return false;
      }
      if (depth > 0 &&
          !loadSpatialTransformer(f, p + "attn." + std::to_string(j), outC,
                                  u.headCounts[static_cast<std::size_t>(n - 1 - i)], depth,
                                  u.crossDim, blk.attns[static_cast<std::size_t>(j)], error)) {
        return false;
      }
    }
    if (i < n - 1 && !loadConv(f, "unet.up." + std::to_string(i) + ".upsample", outC, outC, 3,
                               blk.upsample, error)) {
      return false;
    }
    prevOut = outC;
  }

  return loadNorm(f, "unet.norm_out", ch0, out.normOut, error) &&
         loadConv(f, "unet.conv_out", u.outChannels, ch0, 3, out.convOut, error);
}

bool loadVae(const gguf::GgufFile& f, const SdConfig& cfg, VaeDecoderWeights& out,
             std::string& error) {
  const SdVaeConfig& v = cfg.vae;
  const int n = v.blocks();
  const int deepest = v.channels.back();

  if (f.tensor("vae.post_quant_conv.weight") &&
      !loadConv(f, "vae.post_quant_conv", cfg.latentChannels, cfg.latentChannels, 1,
                out.postQuantConv, error)) {
    return false;
  }
  if (!loadConv(f, "vae.dec.conv_in", deepest, cfg.latentChannels, 3, out.convIn, error) ||
      !loadResnet(f, "vae.dec.mid.resnet.0", deepest, deepest, 0, out.midResnet0, error) ||
      !loadNorm(f, "vae.dec.mid.attn.norm", deepest, out.midAttn.norm, error) ||
      !loadLinear(f, "vae.dec.mid.attn.q", deepest, deepest, out.midAttn.q, error) ||
      !loadLinear(f, "vae.dec.mid.attn.k", deepest, deepest, out.midAttn.k, error) ||
      !loadLinear(f, "vae.dec.mid.attn.v", deepest, deepest, out.midAttn.v, error) ||
      !loadLinear(f, "vae.dec.mid.attn.out", deepest, deepest, out.midAttn.out, error) ||
      !loadResnet(f, "vae.dec.mid.resnet.1", deepest, deepest, 0, out.midResnet1, error)) {
    return false;
  }

  std::vector<int> rev(v.channels.rbegin(), v.channels.rend());
  out.up.resize(static_cast<std::size_t>(n));
  int prev = rev.front();
  for (int i = 0; i < n; ++i) {
    auto& blk = out.up[static_cast<std::size_t>(i)];
    const int outC = rev[static_cast<std::size_t>(i)];
    // The decoder's up blocks have one more resnet than the encoder's down blocks, and NO skip
    // connections — the VAE is not a UNet, it just looks like half of one.
    const int layers = v.layersPerBlock + 1;
    blk.resnets.resize(static_cast<std::size_t>(layers));
    for (int j = 0; j < layers; ++j) {
      const int inC = j == 0 ? prev : outC;
      if (!loadResnet(f, "vae.dec.up." + std::to_string(i) + ".resnet." + std::to_string(j), inC,
                      outC, 0, blk.resnets[static_cast<std::size_t>(j)], error)) {
        return false;
      }
    }
    if (i < n - 1 && !loadConv(f, "vae.dec.up." + std::to_string(i) + ".upsample", outC, outC, 3,
                               blk.upsample, error)) {
      return false;
    }
    prev = outC;
  }

  return loadNorm(f, "vae.dec.norm_out", v.channels.front(), out.normOut, error) &&
         loadConv(f, "vae.dec.conv_out", 3, v.channels.front(), 3, out.convOut, error);
}

bool loadText(const gguf::GgufFile& f, const SdConfig& cfg, TextEncoderWeights& out,
              std::string& error) {
  const SdTextConfig& t = cfg.text;
  if (!loadMat(f, "te.token_embd.weight", t.vocab, t.dModel, out.tokenEmbd, error) ||
      !loadVec(f, "te.position_embd.weight", t.contextLength * t.dModel, out.positionEmbd, error)) {
    return false;
  }
  out.layers.resize(static_cast<std::size_t>(t.layers));
  for (int i = 0; i < t.layers; ++i) {
    auto& L = out.layers[static_cast<std::size_t>(i)];
    const std::string p = "te.blk." + std::to_string(i) + ".";
    if (!loadNorm(f, p + "ln1", t.dModel, L.ln1, error) ||
        !loadLinear(f, p + "attn_q", t.dModel, t.dModel, L.q, error) ||
        !loadLinear(f, p + "attn_k", t.dModel, t.dModel, L.k, error) ||
        !loadLinear(f, p + "attn_v", t.dModel, t.dModel, L.v, error) ||
        !loadLinear(f, p + "attn_out", t.dModel, t.dModel, L.out, error) ||
        !loadNorm(f, p + "ln2", t.dModel, L.ln2, error) ||
        !loadLinear(f, p + "ffn_up", t.ffn, t.dModel, L.ffnUp, error) ||
        !loadLinear(f, p + "ffn_down", t.dModel, t.ffn, L.ffnDown, error)) {
      return false;
    }
  }
  return loadNorm(f, "te.ln_final", t.dModel, out.lnFinal, error);
}

}  // namespace

bool SdConfig::valid() const {
  if (unet.channels.empty() || vae.channels.empty()) return false;
  if (unet.headCounts.size() != unet.channels.size()) return false;
  if (unet.transformerDepth.size() != unet.channels.size()) return false;
  if (text.dModel <= 0 || text.layers <= 0 || text.heads <= 0 || text.vocab <= 0) return false;
  if (text.dModel % text.heads != 0) return false;
  if (unet.crossDim != text.dModel) return false;  // the pairing this one file exists to enforce
  for (std::size_t i = 0; i < unet.channels.size(); ++i) {
    const int h = unet.headCounts[i];
    if (unet.transformerDepth[i] > 0 && (h <= 0 || unet.channels[i] % h != 0)) return false;
  }
  if (unet.normGroups <= 0 || vae.normGroups <= 0) return false;
  if (latentChannels <= 0 || vaeScale <= 0) return false;
  // vaeScale must be what the VAE's own block count implies, or the latent grid and the pixel
  // grid disagree and every size the CLI accepts is off by a factor.
  const int implied = 1 << (vae.blocks() - 1);
  return vaeScale == implied;
}

SdConfig sdConfigFromGguf(const gguf::GgufFile& file, std::string& error) {
  error.clear();
  SdConfig cfg;
  if (file.architecture() != "sd") {
    error = "architecture '" + file.architecture() +
            "' is not a stable-diffusion model (convert one with scripts/convert_sd_to_gguf.py)";
    return cfg;
  }
  cfg.name = file.getString("general.name").value_or("");

  auto u32 = [&](const char* key, int fallback) {
    if (auto v = file.getU64(key)) return static_cast<int>(*v);
    return fallback;
  };
  auto f32 = [&](const char* key, float fallback) {
    if (auto v = file.getF64(key)) return static_cast<float>(*v);
    return fallback;
  };
  auto str = [&](const char* key, const char* fallback) {
    return file.getString(key).value_or(fallback);
  };
  auto ints = [&](const char* key, std::vector<int>& dst) {
    const gguf::GgufValue* v = file.find(key);
    if (!v || !v->isArray()) return;
    dst.reserve(v->array().size());
    for (const auto& e : v->array()) {
      if (auto n = e.asI64()) dst.push_back(static_cast<int>(*n));
    }
  };

  cfg.scaleFactor = f32("sd.scale_factor", cfg.scaleFactor);
  cfg.latentChannels = u32("sd.latent_channels", cfg.latentChannels);
  cfg.vaeScale = u32("sd.vae_scale", cfg.vaeScale);
  cfg.sampleSize = u32("sd.sample_size", cfg.sampleSize);

  cfg.scheduler.predictionType = str("sd.prediction_type", "epsilon");
  cfg.scheduler.trainTimesteps = u32("sd.scheduler.train_timesteps", 1000);
  cfg.scheduler.betaStart = f32("sd.scheduler.beta_start", cfg.scheduler.betaStart);
  cfg.scheduler.betaEnd = f32("sd.scheduler.beta_end", cfg.scheduler.betaEnd);
  cfg.scheduler.betaSchedule = str("sd.scheduler.beta_schedule", "scaled_linear");
  cfg.scheduler.timestepSpacing = str("sd.scheduler.timestep_spacing", "leading");
  cfg.scheduler.stepsOffset = u32("sd.scheduler.steps_offset", 1);
  cfg.scheduler.setAlphaToOne = file.getBool("sd.scheduler.set_alpha_to_one").value_or(false);

  cfg.text.dModel = u32("sd.text.embedding_length", 0);
  cfg.text.layers = u32("sd.text.block_count", 0);
  cfg.text.heads = u32("sd.text.attention.head_count", 0);
  cfg.text.ffn = u32("sd.text.feed_forward_length", 0);
  cfg.text.contextLength = u32("sd.text.context_length", 77);
  cfg.text.vocab = u32("sd.text.vocab_size", 0);
  cfg.text.normEpsilon = f32("sd.text.layer_norm_epsilon", 1e-5f);
  cfg.text.quickGelu = str("sd.text.activation", "quick_gelu") == "quick_gelu";

  cfg.unet.inChannels = u32("sd.unet.in_channels", 4);
  cfg.unet.outChannels = u32("sd.unet.out_channels", 4);
  ints("sd.unet.channels", cfg.unet.channels);
  ints("sd.unet.head_counts", cfg.unet.headCounts);
  ints("sd.unet.transformer_depth", cfg.unet.transformerDepth);
  cfg.unet.midTransformerDepth = u32("sd.unet.mid_transformer_depth", 1);
  cfg.unet.layersPerBlock = u32("sd.unet.layers_per_block", 2);
  cfg.unet.crossDim = u32("sd.unet.cross_attention_dim", 768);
  cfg.unet.normGroups = u32("sd.unet.norm_groups", 32);
  cfg.unet.normEpsilon = f32("sd.unet.norm_epsilon", 1e-5f);
  cfg.unet.transformerNormEpsilon = f32("sd.unet.transformer_norm_epsilon", 1e-6f);
  cfg.unet.freqShift = f32("sd.unet.freq_shift", 0.0f);
  cfg.unet.flipSinToCos = file.getBool("sd.unet.flip_sin_to_cos").value_or(true);
  cfg.unet.timeEmbedDim =
      u32("sd.unet.time_embed_dim", cfg.unet.channels.empty() ? 1280 : cfg.unet.channels[0] * 4);

  ints("sd.vae.channels", cfg.vae.channels);
  cfg.vae.layersPerBlock = u32("sd.vae.layers_per_block", 2);
  cfg.vae.normGroups = u32("sd.vae.norm_groups", 32);
  cfg.vae.normEpsilon = f32("sd.vae.norm_epsilon", 1e-6f);

  if (!cfg.valid()) {
    error = "sd metadata is missing or inconsistent (text d=" + std::to_string(cfg.text.dModel) +
            " layers=" + std::to_string(cfg.text.layers) +
            " heads=" + std::to_string(cfg.text.heads) +
            ", unet blocks=" + std::to_string(cfg.unet.blocks()) +
            " cross=" + std::to_string(cfg.unet.crossDim) +
            ", vae blocks=" + std::to_string(cfg.vae.blocks()) +
            " scale=" + std::to_string(cfg.vaeScale) + ")";
  }
  return cfg;
}

std::optional<SdWeights> loadSdWeights(const gguf::GgufFile& file, const SdConfig& cfg,
                                       std::string& error) {
  error.clear();
  if (!cfg.valid()) {
    error = "invalid sd config";
    return std::nullopt;
  }
  SdWeights w;
  if (!loadText(file, cfg, w.text, error)) return std::nullopt;
  if (!loadUnet(file, cfg, w.unet, error)) return std::nullopt;
  if (!loadVae(file, cfg, w.vae, error)) return std::nullopt;
  return w;
}

}  // namespace qorvix::image
