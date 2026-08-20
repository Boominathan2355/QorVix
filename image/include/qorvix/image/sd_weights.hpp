#pragma once

#include <optional>
#include <string>
#include <vector>

#include "qorvix/runtime/weights.hpp"

namespace qorvix::gguf {
class GgufFile;
}

namespace qorvix::image {

// ---- configuration ---------------------------------------------------------------------------
// Every number here is read from the GGUF rather than compiled in. That is not uniformity for its
// own sake: Stable Diffusion 1.x and 2.x differ only in these values (head counts, activation,
// linear-vs-conv projections, the prediction type), so a hardcoded constant is a model this
// runtime silently gets wrong rather than one it refuses.

// The forward diffusion process the sampler inverts. Pure arithmetic over these six constants.
struct SdSchedulerConfig {
  int trainTimesteps = 1000;
  float betaStart = 0.00085f;
  float betaEnd = 0.012f;
  std::string betaSchedule = "scaled_linear";   // or "linear"
  std::string timestepSpacing = "leading";      // or "trailing" / "linspace"
  int stepsOffset = 1;
  bool setAlphaToOne = false;
  // "epsilon" (SD 1.x, SD 2.0-base) or "v_prediction" (SD 2.1, and every distilled model since).
  // Getting this wrong does not error — it produces a smooth grey field, because the model's
  // output is being interpreted as the wrong quantity from the first step.
  std::string predictionType = "epsilon";
};

struct SdTextConfig {
  int dModel = 0;
  int layers = 0;
  int heads = 0;
  int ffn = 0;
  int contextLength = 77;
  int vocab = 0;
  float normEpsilon = 1e-5f;
  bool quickGelu = true;  // hidden_act: quick_gelu on CLIP-L (SD 1.x), gelu on OpenCLIP (SD 2.x)

  int headDim() const { return heads ? dModel / heads : 0; }
};

struct SdUnetConfig {
  int inChannels = 4;
  int outChannels = 4;
  std::vector<int> channels;           // block_out_channels, coarsest last
  std::vector<int> headCounts;         // attention heads per block
  std::vector<int> transformerDepth;   // BasicTransformerBlocks per attention; 0 = no attention
  int midTransformerDepth = 1;
  int layersPerBlock = 2;
  int crossDim = 768;
  int normGroups = 32;
  int timeEmbedDim = 1280;
  float normEpsilon = 1e-5f;
  // The spatial transformer's own GroupNorm runs at 1e-6 while the resnets around it run at
  // 1e-5. Two epsilons in one network, and only one of them is in the diffusers config.
  float transformerNormEpsilon = 1e-6f;
  float freqShift = 0.0f;
  bool flipSinToCos = true;

  int blocks() const { return static_cast<int>(channels.size()); }
};

struct SdVaeConfig {
  std::vector<int> channels;
  int layersPerBlock = 2;
  int normGroups = 32;
  float normEpsilon = 1e-6f;

  int blocks() const { return static_cast<int>(channels.size()); }
};

struct SdConfig {
  std::string name;
  float scaleFactor = 0.18215f;
  int latentChannels = 4;
  int vaeScale = 8;      // pixels per latent cell
  int sampleSize = 64;   // the latent edge the model was trained at
  SdSchedulerConfig scheduler;
  SdTextConfig text;
  SdUnetConfig unet;
  SdVaeConfig vae;

  int defaultPixels() const { return sampleSize * vaeScale; }
  bool valid() const;
};

SdConfig sdConfigFromGguf(const gguf::GgufFile& file, std::string& error);

// ---- weights ---------------------------------------------------------------------------------
// Convolutions and linear layers are the same struct because, after the converter's flattening,
// they are the same thing: an [out, in] matrix and an [out] bias. What distinguishes them is the
// kernel extent the caller convolves with, which is structural rather than stored.
struct MatWeights {
  runtime::WeightMat w;
  std::vector<float> b;
  bool valid() const { return w.valid(); }
};

struct NormWeights {
  std::vector<float> w, b;
  bool valid() const { return !w.empty(); }
};

// A residual block: two 3x3 convolutions with the timestep vector added between them, plus a 1x1
// shortcut when the channel count changes.
struct ResnetWeights {
  NormWeights norm1;
  MatWeights conv1;
  MatWeights timeEmb;   // absent in the VAE, which has no timestep to condition on
  NormWeights norm2;
  MatWeights conv2;
  MatWeights skip;      // absent when inChannels == outChannels
};

// One BasicTransformerBlock: self-attention, cross-attention, GEGLU feed-forward, each pre-normed
// with a residual. Note there are no q/k/v biases — only the output projections have them.
struct TransformerBlockWeights {
  NormWeights ln1, ln2, ln3;
  runtime::WeightMat q1, k1, v1;
  MatWeights out1;
  runtime::WeightMat q2, k2, v2;
  MatWeights out2;
  MatWeights geglu;   // [2 * inner, dim] — value half first, gate half second
  MatWeights ffnOut;  // [dim, inner]
};

// Transformer2DModel: GroupNorm, project in, N transformer blocks, project out, residual.
struct SpatialTransformerWeights {
  NormWeights norm;
  MatWeights projIn, projOut;
  std::vector<TransformerBlockWeights> blocks;
};

struct UnetDownBlockWeights {
  std::vector<ResnetWeights> resnets;
  std::vector<SpatialTransformerWeights> attns;  // empty when this block has no cross-attention
  MatWeights downsample;                         // absent on the last block
};

struct UnetUpBlockWeights {
  std::vector<ResnetWeights> resnets;
  std::vector<SpatialTransformerWeights> attns;
  MatWeights upsample;  // absent on the last block
};

struct UnetWeights {
  MatWeights convIn;
  MatWeights timeMlp0, timeMlp1;
  std::vector<UnetDownBlockWeights> down;
  ResnetWeights midResnet0, midResnet1;
  SpatialTransformerWeights midAttn;
  std::vector<UnetUpBlockWeights> up;
  NormWeights normOut;
  MatWeights convOut;
};

// The VAE's mid-block attention: ONE head over every spatial position, and unlike the UNet's
// cross-attention its q/k/v all carry biases.
struct VaeAttnWeights {
  NormWeights norm;
  MatWeights q, k, v, out;
};

struct VaeUpBlockWeights {
  std::vector<ResnetWeights> resnets;
  MatWeights upsample;
};

struct VaeDecoderWeights {
  MatWeights postQuantConv;  // optional 1x1; absent on VAEs configured without one
  MatWeights convIn;
  ResnetWeights midResnet0, midResnet1;
  VaeAttnWeights midAttn;
  std::vector<VaeUpBlockWeights> up;
  NormWeights normOut;
  MatWeights convOut;
};

struct TextLayerWeights {
  NormWeights ln1, ln2;
  MatWeights q, k, v, out;
  MatWeights ffnUp, ffnDown;
};

struct TextEncoderWeights {
  runtime::WeightMat tokenEmbd;      // [vocab, d]
  std::vector<float> positionEmbd;   // [ctx * d], F32 and small
  std::vector<TextLayerWeights> layers;
  NormWeights lnFinal;
};

struct SdWeights {
  TextEncoderWeights text;
  UnetWeights unet;
  VaeDecoderWeights vae;
};

// Loads all three networks, keeping matmul weights borrowed from the file's mmap (which must
// outlive the result — StableDiffusion guarantees it) and copying the small F32 norms.
std::optional<SdWeights> loadSdWeights(const gguf::GgufFile& file, const SdConfig& cfg,
                                       std::string& error);

}  // namespace qorvix::image
