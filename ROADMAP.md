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

## Phase 1 — Core Runtime Skeleton
- Process entrypoint (single unified executable)
- `IPlugin` interface + plugin discovery/hot-load/hot-unload
- Model directory watcher (auto-detect, no restart)
- Minimal CLI (`qorvix scan-models`, `qorvix list`)

## Phase 2 — GGUF Parser
Header, metadata, tensor table, tokenizer metadata, RoPE metadata, quant metadata, tensor
offsets, architecture detection. Unit tests against real GGUF files (F32/F16/BF16/Q4_0/Q4_K/
Q5_0/Q5_K/Q6_K/Q8_0).

## Phase 3 — Unified Memory Manager
`MemoryManager`, `TensorRegistry`, `MemoryPool`, `TensorPool`, `KVCacheManager`,
`DiskCacheManager`. Page system (4/8/16/32/64MB pages) — no direct tensor allocations. Tiered
storage (VRAM → RAM → mmap'd NVMe), refcounting, shared buffers.

## Phase 4 — CUDA Backend Bring-Up
Device management, cuBLAS integration, pinned memory, async streams/transfers. Goal: a "hello
tensor" test that moves data host↔device and runs one GEMM correctly.

## Phase 5 — Text Runtime, End-to-End (FP16 first)
RMSNorm, LayerNorm, RoPE, GQA/MQA attention, SwiGLU FFN, residual connections, sampling
(greedy/top-k/top-p/min-p/temperature/penalties), streaming generation. Target: one reference
architecture (Llama-style) producing correct tokens end-to-end, unoptimized. This is the
critical milestone — first real inference.

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

**Status:** Phase 0 in progress. Nothing beyond directory scaffolding and this roadmap exists
yet — treat any performance claims in SPEC.md as targets, not current capabilities.
