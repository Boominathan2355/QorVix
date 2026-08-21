# Phase 11c — GPU Embedding + Vision Backends

## Goal
Implement CUDA and Vulkan backends for:
1. **BERT-family embedding engine** (`IEmbeddingEngine`)
2. **CLIP vision tower** (`ClipVisionModel`)
3. **`forwardEmbedding` on device backends** (needed for multimodal chat with GPU)

---

## Architecture Notes (from ROADMAP)

### Embedding Engine GPU Backends
- `createEmbeddingEngine` currently only builds CPU (`BertModel`)
- Must add CUDA + Vulkan implementations behind the same `IEmbeddingEngine` seam
- **Honest reporting**: factory returns `nullopt` + error when GPU not available rather than silently falling back to CPU

### CLIP Vision Tower GPU Backends
- `ClipVisionModel` is CPU-only today
- Same pattern: CUDA + Vulkan implementations behind the same interface
- `ClipVisionModel` reuses BERT machinery (bidirectional attention, LayerNorm, GEMV)

### `forwardEmbedding` on Device Backends
- Phase 11b-2 added `forwardEmbedding(session, const float*, pos)` to `IInferenceEngine`
- GPU backends (`GpuModel`, `VulkanModel`) hold embedding table in VRAM and look up rows on-device
- Need: **upload path** + **kernel entry point** that skips the lookup
- Until implemented: `acceptsInputEmbeddings()` returns `false` on GPU backends; multimodal surfaces refuse

---

## Implementation Plan

### 1. CUDA Embedding Engine (`cuda/embedding_model.cu/.hpp`)

**Files to create:**
- `cuda/include/qorvix/cuda/embedding_model.hpp` — public header
- `cuda/src/embedding_model.cu` — implementation
- Update `cuda/include/qorvix/cuda/backend.hpp` — add factory declaration
- Update `cuda/src/cuda_backend.cpp` — add factory implementation

**Kernel requirements (mirror `BertModel::encode`):**
- Token embedding lookup (from VRAM-resident table)
- Position + token-type embeddings (if used by model)
- For each encoder layer:
  - LayerNorm (with bias)
  - QKV projection (quantized GEMV: Q4_K/Q6_K/Q8_0/F32)
  - Bidirectional attention (all-to-all, no causal mask)
  - O-projection
  - Residual add
  - LayerNorm (with bias)
  - FFN: up-proj → GELU → down-proj
  - Residual add
- Pooling (CLS/mean/last) + optional L2 normalize

**Key differences from `GpuModel`:**
- No KV cache — encoder has no incremental state
- All tokens processed at once (batch = 1 sequence, not autoregressive)
- No LM head
- Bidirectional attention (not causal GQA)
- GELU not SwiGLU (for BERT)
- LayerNorm **with bias** (not RMSNorm)

**Reusable pieces from existing CUDA code:**
- `qmatmulQ8_0Kernel`, `qmatmulQ4_KKernel`, `qmatmulQ6_KKernel` — quantized GEMV
- `rmsNormKernel` → adapt to `layerNormWithBiasKernel`
- `geluKernel` (already exists for FFN)
- `ropeKernel` — NOT needed (BERT uses learned position embeddings)
- `attentionKernel` — adapt for bidirectional (no causal mask, full Q×K^T)
- `toGpuWeight` / `GpuWeight` structs — weight upload

**Memory management:**
- Upload all weights to VRAM at load time (quantized, never dequantized)
- Embedding table stays in VRAM
- Scratch buffers: hidden states `[maxSeq, d]`, Q/K/V, attention scores, FFN intermediates
- Allocate once at construction, reuse per call (like CPU `BertModel`)

---

### 2. Vulkan Embedding Engine (`vulkan/embedding_model.cpp/.hpp`)

**Files to create:**
- `vulkan/include/qorvix/vulkan/embedding_model.hpp`
- `vulkan/src/embedding_model.cpp`
- Update `vulkan/include/qorvix/vulkan/backend.hpp` — add factory declaration
- Update `vulkan/src/vulkan_backend.cpp` — add factory implementation

**Compute shaders needed (mirror CUDA kernels):**
- `embedding_lookup.comp` — token → embedding row
- `layer_norm_bias.comp` — LayerNorm with bias
- `qmatmul_q8_0.comp`, `qmatmul_q4_k.comp`, `qmatmul_q6_k.comp` — quantized GEMV
- `gelu.comp` — GELU activation
- `attention_bidir.comp` — bidirectional attention (full QK^T)
- `residual_add.comp`
- `ffn_up.comp`, `ffn_down.comp` — FFN projections
- `pooling.comp` — CLS/mean/last + normalize

**Vulkan-specific:**
- SPIR-V shaders compiled at build time, embedded in binary
- Buffer management via `VkBuffer` + `VkDeviceMemory`
- Descriptor sets for weight/hidden state bindings
- Command buffer recording per forward pass
- Synchronization: fences/semaphores for host-device sync

**Reusable from existing Vulkan code:**
- `qmatmulQ8_0SelfTest` etc. — quantized matmul shaders exist
- `opsSelfTest` shaders — RMSNorm, RoPE, SwiGLU, residual add
- Adapt `layerNormBias` from RMSNorm shader
- Adapt `attentionBidir` from `attentionSelfTest` (remove causal mask)

---

### 3. CUDA CLIP Vision Tower (`cuda/clip_model.cu/.hpp`)

**Files to create:**
- `cuda/include/qorvix/cuda/clip_model.hpp`
- `cuda/src/clip_model.cu`
- Update `cuda/include/qorvix/cuda/backend.hpp` — add factory
- Update `cuda/src/cuda_backend.cpp` — add factory

**Key differences from BERT encoder:**
- Patch embedding: Conv2D (kernel=patch_size, stride=patch_size) → flatten → linear projection
- Prepended class token (learned)
- Learned position embeddings (not RoPE)
- **Pre-norm** (LayerNorm before attention, not after)
- **Quick-GELU** (`x * sigmoid(1.702 * x)`) not GELU
- No token-type embeddings
- Output: per-patch hidden states (class token dropped)
- Optional: MLP projector (2-layer: up → GELU → down)

**Reusable kernels:**
- All BERT encoder kernels apply (LayerNorm with bias, bidirectional attention, GELU/Quick-GELU, FFN)
- New: `patch_embedding_kernel` (conv2d + flatten + project)
- New: `quick_gelu_kernel`
- New: `mlp_projector_kernel` (if projector weights present)

---

### 4. Vulkan CLIP Vision Tower (`vulkan/clip_model.cpp/.hpp`)

**Files to create:**
- `vulkan/include/qorvix/vulkan/clip_model.hpp`
- `vulkan/src/clip_model.cpp`
- Update `vulkan/include/qorvix/vulkan/backend.hpp` — add factory
- Update `vulkan/src/vulkan_backend.cpp` — add factory

**Shaders needed (add to embedding shaders):**
- `patch_embedding.comp` — conv2d + project
- `quick_gelu.comp`
- `mlp_projector.comp`

---

### 5. `forwardEmbedding` on CUDA Backend (`cuda/gpu_model.cu`)

**Modify existing:**
- `cuda/include/qorvix/cuda/gpu_model.hpp` — add `forwardEmbedding` declaration
- `cuda/src/gpu_model.cu` — implement `forwardEmbedding`

**Implementation:**
```cpp
// Upload host [d_model] vector to device buffer
// Launch kernel that:
//   1. Copies input embedding to hidden states buffer at position `pos`
//   2. Runs RMSNorm → QKV → RoPE → GQA attention → O-proj → residual
//   3. Runs RMSNorm → SwiGLU FFN → residual
//   4. Repeats for all layers
//   5. Runs final norm → LM head → logits
// Returns logits to host (or keeps on device for next step)
```

**Key point:** Skip embedding table lookup — input is already the embedded vector.

---

### 6. `forwardEmbedding` on Vulkan Backend (`vulkan/vulkan_model.cpp`)

**Modify existing:**
- `vulkan/include/qorvix/vulkan/vulkan_model.hpp` — add `forwardEmbedding`
- `vulkan/src/vulkan_model.cpp` — implement

**Similar to CUDA but with Vulkan command buffers:**
- Upload input embedding to device buffer
- Dispatch compute shaders for each layer (RMSNorm, QKV, RoPE, Attention, O-proj, FFN, etc.)
- Read back logits or keep for next step

---

### 7. Factory Integration

**`cuda/backend.hpp` additions:**
```cpp
namespace qorvix::cuda {
std::unique_ptr<embeddings::IEmbeddingEngine> createEmbeddingEngine(
    gguf::GgufFile file, std::string& error, std::uint32_t maxSeq = 0);
std::unique_ptr<vision::ClipVisionModel> createClipVisionModel(
    gguf::GgufFile file, std::string& error);
}
```

**`vulkan/backend.hpp` additions:**
```cpp
namespace qorvix::vulkan {
std::unique_ptr<embeddings::IEmbeddingEngine> createEmbeddingEngine(
    gguf::GgufFile file, std::string& error, std::uint32_t maxSeq = 0);
std::unique_ptr<vision::ClipVisionModel> createClipVisionModel(
    gguf::GgufFile file, std::string& error);
}
```

**Main factory (`embeddings/backend.hpp`, `vision/backend.hpp`):**
- Try CUDA first (if `builtWithCuda()` and `deviceCount() > 0`)
- Fall back to Vulkan (if `builtWithVulkan()` and `deviceCount() > 0`)
- Fall back to CPU
- Return error if GPU requested but unavailable (honest reporting)

---

### 8. CLI Commands

**New commands:**
- `qorvix embed-check --cuda <model.gguf> --ref <fixture>` — GPU embedding parity
- `qorvix embed-check --vulkan <model.gguf> --ref <fixture>` — Vulkan embedding parity
- `qorvix vision-check --cuda <mmproj.gguf> --ref <fixture>` — GPU CLIP parity
- `qorvix vision-check --vulkan <mmproj.gguf> --ref <fixture>` — Vulkan CLIP parity
- `qorvix vlm-check --cuda <model.gguf> --mmproj <clip.gguf>` — end-to-end GPU multimodal

**Update existing:**
- `qorvix generate --gpu --mmproj ...` — enable when `forwardEmbedding` implemented
- `qorvix serve --gpu --mmproj ...` — enable when `forwardEmbedding` implemented

---

### 9. Tests

**Unit tests (Catch2):**
- `tests/bert_cuda_test.cpp` — CUDA embedding engine vs CPU reference
- `tests/bert_vulkan_test.cpp` — Vulkan embedding engine vs CPU reference
- `tests/clip_cuda_test.cpp` — CUDA CLIP vs CPU reference
- `tests/clip_vulkan_test.cpp` — Vulkan CLIP vs CPU reference
- `tests/forward_embedding_cuda_test.cpp` — `forwardEmbedding` vs `forward` identity
- `tests/forward_embedding_vulkan_test.cpp` — same

**Self-tests (in `cuda::SelfTestResult` / `vulkan::SelfTestResult`):**
- `embeddingSelfTest()` — small synthetic BERT, compare to CPU
- `clipSelfTest()` — small synthetic CLIP, compare to CPU
- `forwardEmbeddingSelfTest()` — token id vs embedded vector parity

---

### 10. Gates (Parity Requirements)

| Component | Gate | Threshold |
|-----------|------|-----------|
| CUDA Embedding | `embed-check --cuda` | cosine ≥ 0.999 vs CPU (F16), ≥ 0.97 vs CPU (Q4_K) |
| Vulkan Embedding | `embed-check --vulkan` | same |
| CUDA CLIP | `vision-check --cuda` | preprocess ≤ 1e-7, patch row 0 cos 1.0, all patches cos ≥ 0.999999 |
| Vulkan CLIP | `vision-check --vulkan` | same |
| CUDA forwardEmbedding | `vlm-check --cuda` tier 1/2 | bit-identical logits |
| Vulkan forwardEmbedding | `vlm-check --vulkan` tier 1/2 | bit-identical logits |

---

## Dependencies / Prerequisites

✅ **Already done:**
- CUDA backend facade + memory tier + quantized kernels (Phases 4, 6, 8)
- Vulkan backend facade + quantized matmul shaders (Phase 8 Vulkan bring-up)
- CPU `BertModel` + `ClipVisionModel` (Phases 11a, 11b-1)
- `IEmbeddingEngine` seam + factory (Phase 11a)
- `forwardEmbedding` interface + CPU implementation (Phase 11b-2)
- Quantized GEMV kernels for Q4_K/Q6_K/Q8_0 on both CUDA + Vulkan

⏳ **Need to verify before starting:**
- CUDA `qmatmulQ4_K/Q6_K` kernels are numerically correct (Phase 8b ✅)
- Vulkan `qmatmulQ4_K/Q6_K` shaders are numerically correct (Phase 8 Vulkan)
- `opsSelfTest` passes on both backends (RMSNorm, RoPE, SwiGLU, residual)

---

## Estimated Effort

| Task | Effort |
|------|--------|
| CUDA Embedding Engine | 3-4 days |
| Vulkan Embedding Engine | 3-4 days |
| CUDA CLIP Vision Tower | 2-3 days |
| Vulkan CLIP Vision Tower | 2-3 days |
| CUDA forwardEmbedding | 1-2 days |
| Vulkan forwardEmbedding | 1-2 days |
| Factory integration + CLI | 1 day |
| Tests + gates | 2 days |
| **Total** | **~15-20 days** |

---

## Risk Mitigation

1. **Numerical drift**: Every kernel must pass `SelfTestResult` vs CPU reference before integration
2. **Memory pressure**: Encoder processes all tokens at once — maxSeq=512 for BERT, 576 patches for CLIP. Pre-allocate scratch at max size.
3. **Quick-GELU variant**: Verify `x * sigmoid(1.702 * x)` matches CPU reference exactly
4. **Pre-norm vs post-norm**: CLIP uses pre-norm; ensure kernel ordering matches CPU `ClipVisionModel::attention`
5. **Projector weights**: May not exist in all mmproj files — `hasProjector()` check required

---

## Acceptance Criteria

- [x] CUDA BERT embedding engine (`cuda/src/embedding_model.cu`, `cuda/include/qorvix/cuda/embedding_model.hpp`)
- [x] CUDA CLIP vision tower & LLaVA MLP projector (`cuda/src/clip_model.cu`, `cuda/include/qorvix/cuda/clip_model.hpp`)
- [x] `forwardEmbedding` on device backends (`cuda::GpuModel` eager & TP sharded, `vulkan::VulkanModel`)
- [x] Unified factory integration in `core/include/qorvix/backend.hpp` (`createEmbeddingEngine`, `createClipVisionModel`)
- [x] CLI commands integration in `core/src/main.cpp` (`embed-check`, `vision-check`, `vlm-check`, `image-embed`, `generate`, `serve`)
- [x] Vulkan stubs and honest error reporting when GPU backend unavailable
- [x] Catch2 unit tests in `tests/cuda_test.cpp` for GPU embedding/vision model creation and graceful handling
- [x] Non-regression on CPU execution path and existing unit test suite