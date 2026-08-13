#include "qorvix/vision/clip_model.hpp"

#include <algorithm>
#include <cmath>

#include "qorvix/runtime/ops.hpp"

namespace qorvix::vision {

namespace rt = qorvix::runtime;

ClipVisionModel::ClipVisionModel(ClipConfig cfg, ClipWeights weights)
    : cfg_(std::move(cfg)), w_(std::move(weights)) {
  const std::size_t d = cfg_.embeddingLength;
  const std::size_t ffn = cfg_.feedForwardLength;
  const std::size_t n = cfg_.tokenCount();

  states_.assign(n * d, 0.0f);
  norm_.assign(n * d, 0.0f);
  q_.assign(n * d, 0.0f);
  k_.assign(n * d, 0.0f);
  v_.assign(n * d, 0.0f);
  attn_.assign(n * d, 0.0f);
  tmp_.assign(n * std::max(d, ffn), 0.0f);
  ffn_.assign(n * ffn, 0.0f);
  scores_.assign(static_cast<std::size_t>(cfg_.headCount) * n, 0.0f);
}

std::optional<ClipVisionModel> ClipVisionModel::fromGguf(gguf::GgufFile file, std::string& error) {
  ClipConfig cfg = clipConfigFromGguf(file, error);
  if (!cfg.valid() || !error.empty()) {
    if (error.empty()) error = "invalid clip config";
    return std::nullopt;
  }
  auto weights = loadClipWeights(file, cfg, error);
  if (!weights) return std::nullopt;

  ClipVisionModel model(std::move(cfg), std::move(*weights));
  model.file_ = std::make_unique<gguf::GgufFile>(std::move(file));
  return model;
}

PreprocessConfig ClipVisionModel::preprocessConfig() const {
  PreprocessConfig p;
  p.size = static_cast<int>(cfg_.imageSize);
  p.mean = cfg_.imageMean;
  p.std = cfg_.imageStd;
  return p;
}

void ClipVisionModel::attention(const ClipLayerWeights& L, int n) {
  const int d = static_cast<int>(cfg_.embeddingLength);
  const int nHeads = static_cast<int>(cfg_.headCount);
  const int headDim = static_cast<int>(cfg_.headDim());
  const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));

  const float* xs = norm_.data();
  rt::wmatmulNBias(q_.data(), L.wq, xs, n, L.bq);
  rt::wmatmulNBias(k_.data(), L.wk, xs, n, L.bk);
  rt::wmatmulNBias(v_.data(), L.wv, xs, n, L.bv);

  // Fully bidirectional: a vision transformer has no causal structure at all — every patch
  // attends to every patch, and to the class token.
  for (int h = 0; h < nHeads; ++h) {
    const int off = h * headDim;
    float* scores = scores_.data() + static_cast<std::size_t>(h) * cfg_.tokenCount();
    for (int i = 0; i < n; ++i) {
      const float* qi = q_.data() + static_cast<std::size_t>(i) * d + off;
      for (int t = 0; t < n; ++t) {
        const float* kt = k_.data() + static_cast<std::size_t>(t) * d + off;
        float dot = 0.0f;
        for (int e = 0; e < headDim; ++e) dot += qi[e] * kt[e];
        scores[t] = dot * scale;
      }
      rt::ops::softmax(scores, n);

      float* outRow = attn_.data() + static_cast<std::size_t>(i) * d + off;
      for (int e = 0; e < headDim; ++e) outRow[e] = 0.0f;
      for (int t = 0; t < n; ++t) {
        const float wgt = scores[t];
        const float* vt = v_.data() + static_cast<std::size_t>(t) * d + off;
        for (int e = 0; e < headDim; ++e) outRow[e] += wgt * vt[e];
      }
    }
  }
}

bool ClipVisionModel::encodePixels(const std::vector<float>& chw, std::vector<float>& out,
                                   std::string& error) {
  error.clear();
  const int d = static_cast<int>(cfg_.embeddingLength);
  const int ffn = static_cast<int>(cfg_.feedForwardLength);
  const int side = static_cast<int>(cfg_.patchesPerSide());
  const int patch = static_cast<int>(cfg_.patchSize);
  const int imgSize = static_cast<int>(cfg_.imageSize);
  const int n = static_cast<int>(cfg_.tokenCount());
  const float eps = cfg_.normEpsilon;

  const std::size_t expect = static_cast<std::size_t>(imgSize) * imgSize * 3;
  if (chw.size() != expect) {
    error = "expected " + std::to_string(expect) + " preprocessed values (3x" +
            std::to_string(imgSize) + "x" + std::to_string(imgSize) + "), got " +
            std::to_string(chw.size());
    return false;
  }

  // ---- patch embedding ----
  // A conv2d whose stride equals its kernel is exactly a matmul over non-overlapping flattened
  // patches. The flattening order must match ggml's kernel layout [KW, KH, IC, OC], where KW is
  // fastest: column index = channel*(patch*patch) + ky*patch + kx.
  const int patchElems = patch * patch * 3;
  std::vector<float> flat(static_cast<std::size_t>(side) * side * patchElems);
  const std::size_t plane = static_cast<std::size_t>(imgSize) * imgSize;
  for (int py = 0; py < side; ++py) {
    for (int px = 0; px < side; ++px) {
      float* dst = flat.data() + (static_cast<std::size_t>(py) * side + px) * patchElems;
      for (int c = 0; c < 3; ++c) {
        for (int ky = 0; ky < patch; ++ky) {
          for (int kx = 0; kx < patch; ++kx) {
            const int sy = py * patch + ky;
            const int sx = px * patch + kx;
            dst[static_cast<std::size_t>(c) * patch * patch + static_cast<std::size_t>(ky) * patch +
                kx] = chw[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(sy) * imgSize + sx];
          }
        }
      }
    }
  }

  // Token 0 is the class embedding; tokens 1..N are the patches.
  std::copy(w_.classEmbd.begin(), w_.classEmbd.end(), states_.begin());
  rt::wmatmulNBias(states_.data() + static_cast<std::size_t>(d), w_.patchEmbd, flat.data(),
                   side * side, {});

  // Learned absolute position embeddings, one row per token including the class token.
  for (int t = 0; t < n; ++t) {
    rt::embeddingRow(w_.positionEmbd, t, tmp_.data());
    rt::ops::add(states_.data() + static_cast<std::size_t>(t) * d, tmp_.data(), d);
  }

  // CLIP normalizes BEFORE the blocks as well as inside them.
  if (!w_.preLnW.empty()) {
    for (int t = 0; t < n; ++t) {
      float* x = states_.data() + static_cast<std::size_t>(t) * d;
      rt::ops::layernorm(x, x, w_.preLnW.data(), w_.preLnB.empty() ? nullptr : w_.preLnB.data(), d,
                         eps);
    }
  }

  // ---- pre-norm transformer blocks ----
  for (const auto& L : w_.layers) {
    // PRE-norm, unlike the BERT encoder: the LayerNorm feeds the sublayer, and the residual is
    // added to the UNNORMALIZED stream. Swapping the two still trains and still runs; it just
    // produces different numbers, which is why the gate compares against a reference.
    for (int t = 0; t < n; ++t) {
      const float* x = states_.data() + static_cast<std::size_t>(t) * d;
      rt::ops::layernorm(norm_.data() + static_cast<std::size_t>(t) * d, x, L.ln1W.data(),
                         L.ln1B.empty() ? nullptr : L.ln1B.data(), d, eps);
    }
    attention(L, n);
    rt::wmatmulNBias(tmp_.data(), L.wo, attn_.data(), n, L.bo);
    for (int t = 0; t < n; ++t) {
      rt::ops::add(states_.data() + static_cast<std::size_t>(t) * d,
                   tmp_.data() + static_cast<std::size_t>(t) * d, d);
    }

    for (int t = 0; t < n; ++t) {
      const float* x = states_.data() + static_cast<std::size_t>(t) * d;
      rt::ops::layernorm(norm_.data() + static_cast<std::size_t>(t) * d, x, L.ln2W.data(),
                         L.ln2B.empty() ? nullptr : L.ln2B.data(), d, eps);
    }
    // ffnExpand/ffnContract, NOT ffn_up/ffn_down: this format inverts those names (see
    // clip_weights.hpp). The fields are named for what they do so the order here reads correctly.
    rt::wmatmulNBias(ffn_.data(), L.ffnExpand, norm_.data(), n, L.ffnExpandB);
    if (cfg_.useGelu) {
      rt::ops::geluInPlace(ffn_.data(), n * ffn);
    } else {
      rt::ops::geluQuickInPlace(ffn_.data(), n * ffn);
    }
    rt::wmatmulNBias(tmp_.data(), L.ffnContract, ffn_.data(), n, L.ffnContractB);
    for (int t = 0; t < n; ++t) {
      rt::ops::add(states_.data() + static_cast<std::size_t>(t) * d,
                   tmp_.data() + static_cast<std::size_t>(t) * d, d);
    }
  }

  if (!w_.postLnW.empty()) {
    for (int t = 0; t < n; ++t) {
      float* x = states_.data() + static_cast<std::size_t>(t) * d;
      rt::ops::layernorm(x, x, w_.postLnW.data(), w_.postLnB.empty() ? nullptr : w_.postLnB.data(),
                         d, eps);
    }
  }

  // Drop the class token: LLaVA feeds the decoder the patch grid, not the pooled summary.
  out.assign(states_.begin() + static_cast<std::size_t>(d), states_.begin() + static_cast<std::size_t>(n) * d);
  return true;
}

bool ClipVisionModel::encodeImage(const Image& img, std::vector<float>& out, std::string& error) {
  std::vector<float> chw;
  if (!preprocessClip(img, preprocessConfig(), chw, error)) return false;
  return encodePixels(chw, out, error);
}

bool ClipVisionModel::project(const std::vector<float>& hidden, std::vector<float>& out,
                              std::string& error) {
  error.clear();
  if (!w_.hasProjector()) {
    error = "this clip file has no llava projector";
    return false;
  }
  const int d = static_cast<int>(cfg_.embeddingLength);
  const int tokens = static_cast<int>(patchTokens());
  if (hidden.size() != static_cast<std::size_t>(tokens) * d) {
    error = "projector input has the wrong shape";
    return false;
  }
  const int llm = w_.projectedDim();

  // LLaVA's "mlp" projector: Linear -> GELU -> Linear. Exact GELU here, not quick — the projector
  // is trained as part of LLaVA rather than inherited from CLIP, and uses the standard activation.
  std::vector<float> mid(static_cast<std::size_t>(tokens) * llm);
  rt::wmatmulNBias(mid.data(), w_.mm0, hidden.data(), tokens, w_.mm0B);
  rt::ops::geluInPlace(mid.data(), tokens * llm);

  out.assign(static_cast<std::size_t>(tokens) * llm, 0.0f);
  rt::wmatmulNBias(out.data(), w_.mm2, mid.data(), tokens, w_.mm2B);
  return true;
}

}  // namespace qorvix::vision
