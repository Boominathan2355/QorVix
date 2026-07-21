# QORVIX AI CORE — Roadmap

> Breaks the mission in [docs/SPEC.md](docs/SPEC.md) into buildable phases. Each phase should
> end in something that compiles and runs, not just headers/interfaces. Order is chosen so that
> a real end-to-end token ever gets generated (Phase 5) before any optimization work begins —
> optimizing a pipeline that doesn't exist yet is wasted effort.

## Phase 0 — Foundations
Repo scaffolding, build system, dependency strategy, coding standards, CI skeleton.
- Git repository, `.gitignore`, top-level `CMakeLists.txt` with per-module subdirectories
- **Decided (2026-07-21):** Linux is the reference/deployment target (CUDA servers, Docker, NCCL
  multi-GPU all assume it); Windows/MSVC stays buildable where practical but isn't the CI gate.
- **Decided:** vcpkg for dependency management (manifest mode, `vcpkg.json`).
- **Decided:** Boost.Beast for the networking layer (HTTP/WebSocket/SSE) — asio-based, scales
  better for high-concurrency inference serving than CrowCPP.
- Logging, config loading, error handling conventions — established as each module needs them,
  not speculatively up front.

## Phase 1 — Core Runtime Skeleton ✅
- Process entrypoint (single unified executable) — `qorvix` CLI
- `IPlugin` interface + plugin discovery/hot-load/hot-unload — `PluginRegistry` over a
  cross-platform `DynamicLibrary` (dlopen/LoadLibrary), C-ABI factory via `QORVIX_REGISTER_PLUGIN`;
  reference plugin in `plugins/example`
- Model directory watcher (auto-detect, no restart) — polling-based `ModelWatcher` with
  add/removed callbacks; `ModelRegistry` for one-shot discovery
- CLI: `qorvix scan-models [dir]`, `qorvix list [dir]`, `qorvix plugins [dir]`, `version`, `help`
- Verified locally via the `quick` preset (build + run); Catch2 unit tests cover registry,
  watcher, and plugin load/unload (run in CI where vcpkg provides Catch2)

## Phase 2 — GGUF Parser ✅
Full GGUF v1/v2/v3 parser in the `gguf` module:
- Header (magic/version/counts) with version-aware widths
- Metadata KV parser — all 13 value types incl. arrays, order-preserving, typed widening
  accessors (`getString/getU64/getF64/getBool`, arch-prefixed `archU64`)
- Tensor table — name/dims/type/offset, computed element & byte counts from per-type block
  traits, 32-byte-aligned data-section offset
- `ggml_type` traits table (F32/F16/BF16, Q4_0…Q8_0, K-quants, IQ-quants) with SPEC-supported flags
- Architecture detection (`general.architecture`) + `RopeParams` view
- Bounds-checked, endianness-independent reader; defensive against truncation, bad magic/version,
  misaligned offsets, block-size mismatches, and OOM-via-huge-array-length
- `MappedFile` (mmap/CreateFileMapping) so `open()` parses a multi-GB file's header without
  reading tensor data
- `qorvix gguf-info <file>` CLI command
- Catch2 tests + in-memory `GgufBuilder` fixture generator (run in CI); verified locally via a
  standalone harness against the `quick` preset.

## Phase 3 — Unified Memory Manager ✅
`memory` module — page-based, tiered allocation with a global tensor registry:
- Page system (SPEC 4/8/16/32/64MB; KiB-scale in tests): `MemoryPage` first-fit free list with
  coalescing, 256B sub-allocation alignment (CUDA-pointer compatible); `PagePool` per tier with
  first-fit page sharing, smallest-fit growth, bespoke huge pages, byte budgets, `trim()`, and
  fragmentation stats. No direct tensor allocations — everything goes through pages.
- Tier backends behind `ISlabAllocator`: `HostSlabAllocator` (page-aligned RAM) and
  `DiskSlabAllocator` (writable-mmap spool files = the NVMe tier / DiskCacheManager storage).
  The CUDA VRAM backend drops in behind the same interface in Phase 4.
- `MemoryManager` + `TensorRef` (the TensorRegistry): named refcounted buffers with RAII pinning,
  shared buffers via `alias()`, explicit `migrate()` across tiers preserving contents,
  memory-aware placement (preferred-tier fallback down the chain), and smart eviction — under
  budget pressure the LRU zero-ref buffer offloads down a tier (VRAM→RAM→disk) instead of dying.
- Deferred deliberately: KV cache managers build on this in Phase 7; defragmentation is
  *reported* (largestFreeBlock) but not yet compacted; prefetch/predictive loading arrive with
  real workloads. Thread safety is one coarse mutex until the scheduler exists to profile it.

## Phase 4 — CUDA Backend Bring-Up ✅ (code complete; GPU-run pending hardware)
New `cuda` module — a backend facade callable from any build, plus the GpuVram memory tier:
- Device management (enumerate/select, compute capability, SM count, free/total VRAM)
- "Hello tensor" self-test: host→device→host round-trip through a scale kernel; cuBLAS SGEMM
  self-test (C = A·B with A = I), both verified on the host
- `CudaSlabAllocator` (cudaMalloc/cudaFree) behind Phase 3's `ISlabAllocator`, and
  `CudaTransferEngine` behind the new `ITransferEngine` seam — so the unified MemoryManager gains
  a real GpuVram tier with H2D/D2H/D2D migration and offload, no changes to pool/manager logic
- `MemoryManager` now routes migration/eviction copies through an `ITransferEngine`
  (HostTransferEngine memcpy default; CudaTransferEngine when built with CUDA)
- Compile-time gating: a CPU **stub** (builds with no toolkit; `builtWithCuda()==false`) vs the
  real `.cu` (nvcc, links cudart+cublas). `qorvix gpu` CLI reports devices + runs the self-tests.
- Tests skip device assertions cleanly when `deviceCount()==0`, so CI/GPU-less hosts stay green.

**Verified:** stub path + transfer routing locally (quick preset). The real `.cu`/cuBLAS path is
being compile-checked under nvcc via `Dockerfile.cuda` (build in progress at commit time; result
folded in once it lands). **Not yet run on a GPU** — this dev box and CI have no NVIDIA device, so
even once it compiles, the kernels/GEMM stay execution-unverified until GPU hardware is available.

Deferred to the CUDA performance pass (Phase 8): pinned host memory, async streams/overlapped
transfers, CUDA graphs, FlashAttention, CUTLASS. Bring-up establishes the device + memory tier
first; the fast paths come once there's a pipeline to profile.

## Phase 5 — Text Runtime, End-to-End 🚧 (in progress)
**Decision:** built as a **CPU reference runtime** first (no GPU on this dev box or CI, so
GPU-only numerical code couldn't be *run* — only compile-checked, which is worthless for math).
Compute lives behind a small op layer so the CUDA kernels (Phase 6/8) reproduce the same results.
This mirrors how llama.cpp itself started.

**Part 1 ✅ — numerical foundation (`runtime` module):**
- CPU ops (F32, row-major): RMSNorm, LayerNorm, matmul, stable softmax, SiLU/SwiGLU, RoPE
  (interleaved + NeoX modes), residual add, argmax — each unit-tested against hand-computed values.
- GGUF dequantization to F32, block layouts mirroring ggml exactly: F16, BF16, Q8_0, Q8_1, Q4_0,
  Q4_1, Q5_0, Q5_1, and the K-quants Q4_K, Q5_K, Q6_K — tested against hand-constructed blocks
  with known fp16 scales.
- Verified locally (standalone harness) and in CI (Catch2).

**Part 2a ✅ — model + forward pass:**
- `ModelConfig` derived from GGUF metadata (llama/qwen2/mistral/gemma/phi3 families); `qorvix
  model-info <file>` prints it.
- `Weights` loader: dequantizes every Llama-convention tensor (token_embd, blk.N.attn_*/ffn_*,
  output_norm, output; tied-embedding fallback) to F32 from an mmap'd GGUF.
- `TextModel` forward pass: embedding → per-layer [RMSNorm → QKV → RoPE → GQA/MQA attention over
  a contiguous KV cache → o-proj → residual → RMSNorm → SwiGLU FFN → residual] → final norm → LM
  head → logits. Plus a greedy `generateGreedy`.
- Verified analytically on hand-built synthetic models (residual-identity, zero-Q/K attention =
  value-average, greedy determinism, GQA head grouping) — locally + Catch2/CI.

**Part 2b (next):** BPE tokenizer (encode/decode from GGUF vocab+merges), full sampling
(top-k/top-p/min-p/temperature + repetition/frequency/presence penalties), and the streaming
generation loop + `qorvix generate` CLI. Then load a **real GGUF** (drop one in `models/`) and
validate generated text against llama.cpp — including confirming the RoPE mode per architecture.

## Phase 6 — Native Quantization Kernels
Direct GPU kernels for Q4_K/Q5_K/Q6_K/Q8_0 — matmul, attention, and FFN without dequant-to-FP16.
Quantized KV cache.

## Phase 7 — KV Cache, Batching, Scheduler
`GlobalKVCache` (paged, shared, session-restore, compression, pruning), dynamic + continuous
batching, request prioritization, GPU/memory/model-aware scheduling.

## Phase 8 — CUDA Performance Pass
FlashAttention 3, CUDA Graphs, CUTLASS kernels, tensor core paths, kernel fusion, persistent
kernels. Benchmark against Phase 5 baseline before/after each optimization.

## Phase 9 — API Layer
OpenAI-compatible REST endpoints, WebSocket + SSE streaming, API gateway in front of the
scheduler.

## Phase 10 — Multi-GPU
NCCL-based tensor parallelism, pipeline parallelism, expert parallelism, load balancing across
1/2/4/8 GPUs.

## Phase 11 — Multimodal Expansion
Vision engine (Qwen-VL, Llama Vision, MiniCPM-V, InternVL; OCR, grounding), audio engine
(STT/TTS/voice cloning), image generation (Flux/SDXL/SD; txt2img/img2img/inpaint/outpaint/
upscale/controlnet), embeddings (text/vision/audio/cross-modal), RAG (loaders, chunking, hybrid
search, vector store), multi-agent workflows.

## Phase 12 — Web UI
React/TS/Vite/Tailwind/shadcn app: Dashboard, Chat, Vision, Audio, Image Generation, Model,
Memory, Performance, Settings pages. Prometheus/Grafana exporter wiring.

## Phase 13 — Enterprise Hardening
Speculative decoding (draft/target/verification), API keys, rate limiting, audit logs, security
review, stress/leak/GPU-regression test suites.

## Phase 14 — Performance Validation
Benchmark on target hardware (RTX 4090) against the throughput/utilization/context-length
targets in SPEC.md. Tune until targets are met or document the gap honestly.

---

**Status:** Phase 4 code-complete. Runtime skeleton, GGUF parser, unified memory manager, and the
CUDA backend (device mgmt, GpuVram tier, self-tests) build and pass tests — CPU path locally and
in Docker, CUDA path compiles under nvcc. GPU *execution* is unverified (no NVIDIA device on this
box or CI). No inference yet — treat performance claims in SPEC.md as targets, not current
capabilities. Next: Phase 5 (text runtime, end-to-end — the first real inference).
