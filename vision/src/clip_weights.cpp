#include "qorvix/vision/clip_weights.hpp"

#include "qorvix/gguf/gguf_file.hpp"

// Reuses the runtime's tensor-load helpers rather than duplicating the mmap borrowing,
// element-count checking and quantized-type gating.
#include "qorvix/runtime/tensor_load.hpp"

namespace qorvix::vision {

namespace rt = qorvix::runtime;
using rt::detail::loadMat;
using rt::detail::loadMatOpt;
using rt::detail::loadVec;
using rt::detail::loadVecOpt;

namespace {

std::string vblk(int i, const char* suffix) {
  return "v.blk." + std::to_string(i) + "." + suffix;
}

}  // namespace

ClipConfig clipConfigFromGguf(const gguf::GgufFile& file, std::string& error) {
  error.clear();
  ClipConfig cfg;
  if (file.architecture() != "clip") {
    error = "architecture '" + file.architecture() + "' is not a clip vision tower";
    return cfg;
  }
  cfg.name = file.getString("general.name").value_or("");

  auto u32 = [&](const char* key, std::uint32_t fallback) {
    if (auto v = file.getU64(key)) return static_cast<std::uint32_t>(*v);
    return fallback;
  };
  cfg.imageSize = u32("clip.vision.image_size", cfg.imageSize);
  cfg.patchSize = u32("clip.vision.patch_size", cfg.patchSize);
  cfg.embeddingLength = u32("clip.vision.embedding_length", cfg.embeddingLength);
  cfg.feedForwardLength = u32("clip.vision.feed_forward_length", cfg.feedForwardLength);
  cfg.headCount = u32("clip.vision.attention.head_count", cfg.headCount);
  cfg.blockCount = u32("clip.vision.block_count", cfg.blockCount);
  cfg.projectionDim = u32("clip.vision.projection_dim", cfg.projectionDim);
  if (auto v = file.getF64("clip.vision.attention.layer_norm_epsilon")) {
    cfg.normEpsilon = static_cast<float>(*v);
  }
  cfg.useGelu = file.getBool("clip.use_gelu").value_or(false);
  cfg.hasVisionEncoder = file.getBool("clip.has_vision_encoder").value_or(true);
  cfg.hasLlavaProjector = file.getBool("clip.has_llava_projector").value_or(false);
  cfg.projectorType = file.getString("clip.projector_type").value_or("mlp");

  // The tower carries its own normalization statistics. Read them rather than assuming OpenAI's,
  // so a differently-trained tower is normalized correctly instead of plausibly.
  auto readTriple = [&](const char* key, std::array<float, 3>& dst) {
    const gguf::GgufValue* v = file.find(key);
    if (!v || !v->isArray() || v->array().size() < 3) return;
    for (int i = 0; i < 3; ++i) {
      dst[static_cast<std::size_t>(i)] = static_cast<float>(v->array()[i].asF64().value_or(dst[i]));
    }
  };
  readTriple("clip.vision.image_mean", cfg.imageMean);
  readTriple("clip.vision.image_std", cfg.imageStd);

  if (!cfg.hasVisionEncoder) {
    error = "this clip file has no vision encoder";
    return cfg;
  }
  if (!cfg.valid()) {
    error = "clip metadata is missing required vision hyperparameters";
  }
  return cfg;
}

std::optional<ClipWeights> loadClipWeights(const gguf::GgufFile& file, const ClipConfig& cfg,
                                           std::string& error) {
  error.clear();
  if (!cfg.valid()) {
    error = "invalid clip config";
    return std::nullopt;
  }

  const int d = static_cast<int>(cfg.embeddingLength);
  const int ffn = static_cast<int>(cfg.feedForwardLength);
  const int patchElems = static_cast<int>(cfg.patchSize) * static_cast<int>(cfg.patchSize) * 3;
  const int tokens = static_cast<int>(cfg.tokenCount());

  ClipWeights w;
  // The patch embedding is stored as a conv2d kernel [patch, patch, 3, d]; a stride-equals-kernel
  // convolution IS a matmul over flattened patches, so it loads directly as [d, patch*patch*3]
  // with no transpose.
  if (!loadMat(file, "v.patch_embd.weight", d, patchElems, w.patchEmbd, error)) return std::nullopt;
  if (!loadVec(file, "v.class_embd", d, w.classEmbd, error)) return std::nullopt;
  if (!loadMat(file, "v.position_embd.weight", tokens, d, w.positionEmbd, error)) {
    return std::nullopt;
  }
  if (!loadVecOpt(file, "v.pre_ln.weight", d, w.preLnW, error)) return std::nullopt;
  if (!loadVecOpt(file, "v.pre_ln.bias", d, w.preLnB, error)) return std::nullopt;
  // LLaVA towers drop the final block and the post-layernorm with it, because the projector
  // consumes the second-to-last hidden state. Optional, not missing.
  if (!loadVecOpt(file, "v.post_ln.weight", d, w.postLnW, error)) return std::nullopt;
  if (!loadVecOpt(file, "v.post_ln.bias", d, w.postLnB, error)) return std::nullopt;

  w.layers.resize(cfg.blockCount);
  for (int i = 0; i < static_cast<int>(cfg.blockCount); ++i) {
    ClipLayerWeights& L = w.layers[i];
    const bool ok =
        loadMat(file, vblk(i, "attn_q.weight"), d, d, L.wq, error) &&
        loadMat(file, vblk(i, "attn_k.weight"), d, d, L.wk, error) &&
        loadMat(file, vblk(i, "attn_v.weight"), d, d, L.wv, error) &&
        loadMat(file, vblk(i, "attn_out.weight"), d, d, L.wo, error) &&
        loadVec(file, vblk(i, "attn_q.bias"), d, L.bq, error) &&
        loadVec(file, vblk(i, "attn_k.bias"), d, L.bk, error) &&
        loadVec(file, vblk(i, "attn_v.bias"), d, L.bv, error) &&
        loadVec(file, vblk(i, "attn_out.bias"), d, L.bo, error) &&
        loadVec(file, vblk(i, "ln1.weight"), d, L.ln1W, error) &&
        loadVec(file, vblk(i, "ln1.bias"), d, L.ln1B, error) &&
        loadVec(file, vblk(i, "ln2.weight"), d, L.ln2W, error) &&
        loadVec(file, vblk(i, "ln2.bias"), d, L.ln2B, error) &&
        // See the header: ffn_down EXPANDS and ffn_up CONTRACTS in this format. The shapes are
        // the proof — loadMat checks the element count, so a swap fails loudly here rather than
        // producing a wrongly-shaped forward pass.
        loadMat(file, vblk(i, "ffn_down.weight"), ffn, d, L.ffnExpand, error) &&
        loadVec(file, vblk(i, "ffn_down.bias"), ffn, L.ffnExpandB, error) &&
        loadMat(file, vblk(i, "ffn_up.weight"), d, ffn, L.ffnContract, error) &&
        loadVec(file, vblk(i, "ffn_up.bias"), d, L.ffnContractB, error);
    if (!ok) return std::nullopt;
  }

  if (cfg.hasLlavaProjector) {
    if (cfg.projectorType != "mlp") {
      error = "projector type '" + cfg.projectorType + "' is not supported (only 'mlp')";
      return std::nullopt;
    }
    // The projector's output width is the language model's d_model, which is not in this file's
    // metadata — infer it from the tensor itself.
    const gguf::GgufTensor* mm0 = file.tensor("mm.0.weight");
    if (!mm0 || mm0->nElements % static_cast<std::uint64_t>(d) != 0) {
      error = "missing or malformed mm.0.weight";
      return std::nullopt;
    }
    const int llmDim = static_cast<int>(mm0->nElements / static_cast<std::uint64_t>(d));
    const bool ok = loadMat(file, "mm.0.weight", llmDim, d, w.mm0, error) &&
                    loadVec(file, "mm.0.bias", llmDim, w.mm0B, error) &&
                    loadMat(file, "mm.2.weight", llmDim, llmDim, w.mm2, error) &&
                    loadVec(file, "mm.2.bias", llmDim, w.mm2B, error);
    if (!ok) return std::nullopt;
  }

  return w;
}

}  // namespace qorvix::vision
