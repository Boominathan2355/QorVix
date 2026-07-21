#pragma once

#include <cstddef>

// CPU reference implementations of the transformer primitives (SPEC "Text Runtime"). Correctness
// first, no SIMD/threading yet — these define the ground-truth math that the CUDA kernels in
// Phase 6/8 must reproduce. All tensors are row-major float arrays; sizes are explicit.
namespace qorvix::runtime::ops {

// RMSNorm: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i].
void rmsnorm(float* out, const float* x, const float* weight, int n, float eps = 1e-5f);

// LayerNorm with optional bias: out[i] = (x[i] - mean) / sqrt(var + eps) * weight[i] + bias[i].
// `bias` may be null.
void layernorm(float* out, const float* x, const float* weight, const float* bias, int n,
               float eps = 1e-5f);

// Matrix-vector: out[r] = dot(W[r, :], x), where W is row-major [rows, cols] and x is [cols].
// This is the layout GGUF stores weights in (out_features x in_features).
void matmul(float* out, const float* weight, const float* x, int rows, int cols);

// Numerically stable softmax over the first n elements, in place.
void softmax(float* x, int n);

// SiLU (a.k.a. swish): silu(z) = z * sigmoid(z).
float silu(float z);

// SwiGLU FFN activation: out[i] = silu(gate[i]) * up[i].
void swiglu(float* out, const float* gate, const float* up, int n);

// Elementwise residual add: out[i] += x[i].
void add(float* out, const float* x, int n);

// Rotary position embedding, applied in place to a [n_heads * head_dim] vector.
//   Interleaved (a.k.a. GGML "NORM"): rotates adjacent pairs (2i, 2i+1).
//   Neox: rotates split-half pairs (i, i + head_dim/2).
// theta_i = pos * freqBase^(-2i / head_dim). Real models pick a mode per architecture; both are
// provided so the model loader can select the correct one.
enum class RopeMode { Interleaved, Neox };
void rope(float* vec, int n_heads, int head_dim, int pos, float freqBase, RopeMode mode);

// Index of the maximum element (greedy argmax).
int argmax(const float* x, int n);

}  // namespace qorvix::runtime::ops
