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

**Verified:** stub path + transfer routing locally (quick preset). The real `.cu`/cuBLAS path
**compiles under nvcc** — confirmed by building `Dockerfile.cuda` (nvcc 12.6 builds
`cuda_backend.cu` → `libqorvix_cuda.a` → the `qorvix:cuda` image; 70/70 tests pass in-container).
**Not yet run on a GPU** — this dev box and CI have no NVIDIA device, so the kernels/GEMM stay
execution-unverified until GPU hardware is available (the CUDA facade degrades gracefully with no
device, which is what the container tests exercise).

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

**Part 2b ✅ — generation engine (text out from a GGUF file):**
- `tokenizer` module: SPM (score-based merges, ▁ marker) and byte-level BPE (merge-rank, GPT-2
  byte↔unicode alphabet) — encode + byte-aware decode, byte-token fallback, loaded from
  GGUF `tokenizer.ggml.*` metadata.
- `Sampler`: temperature, top-k, top-p (nucleus), min-p, repetition/frequency/presence penalties;
  greedy when temperature ≤ 0; deterministic per seed.
- `Generator`: tokenize → prefill → streaming sample/decode loop; stops on EOS / maxNewTokens /
  max seq len. `qorvix generate <file> --prompt "..." [--max --temp --top-k --top-p --seed]`.
- Verified: tokenizer (SPM+BPE), sampler, and full generation on synthetic models (standalone
  harness + Catch2), **and end-to-end from a real GGUF on disk** — `generate` loads a built
  toy model (dequantized F32 weights + BPE vocab) and streams the expected tokens.

**Part 2c ✅ — real-model validation:** validated on **TinyLlama 1.1B Chat Q4_K_M** (real llama
arch, 22 layers, GQA 32/4, SPM tokenizer). `model-info` reads the config exactly; greedy
`generate` on "The capital of France is" produces "the city of Paris, which is a" — coherent,
grammatical, factually correct. This exercises the whole path on real weights: GGUF parse → SPM
tokenizer (32k vocab) → Q4_K/Q6_K dequant → 22-layer forward pass → greedy decode. The RoPE mode
is confirmed empirically (NeoX is correct for this GGUF llama model; wrong rope would scramble
positions and produce garbage). Exact bit-level logit parity with llama.cpp is not yet measured
(llama.cpp isn't installed here), but generation correctness is established.

**Phase 5 is functionally complete** — Qorvix generates correct text from a real GGUF model on
CPU. A first optimization pass (Release `-O3` + OpenMP-parallel matmul & dequant + `-march=native`
AVX2, via the `quick-release` preset) took TinyLlama from ~185s to ~43s for the same work (~4.3×,
byte-identical output) — load ~16s, forward ~0.8 tok/s on 4 cores. It's still a *reference*, not
fast: the forward pass is memory-bandwidth-bound because it streams all ~4.4 GB of dequantized-F32
weights every token. **Phase 6** (native quantized kernels — read 4-bit weights directly, ~8× less
bandwidth) and **Phase 8** (GPU) are where real speed comes from — same validated path, accelerated.
Other known gaps, deferred by design: Qwen2/Gemma need their attention-bias / logit-softcap quirks;
partial-rope (rope_dim < head_dim) untested.

## Phase 6 — Native Quantization Kernels 🚧
**Part a ✅ — quantized matmul kernels:** `qmatmul` computes the GEMV directly against GGUF blocks
(dequantize one block into registers, fold into the dot product — never materialize the F32
weight); `dequantRow` for embedding lookup. Verified bit-for-bit against dequant+F32-matmul.

**Part b ✅ — wired into the model:** `WeightMat` holds each matmul weight as either owned-F32
(tests) or **borrowed quantized bytes aliasing the mmap** (real models); `TextModel` owns the
GgufFile to keep that mapping alive. The forward pass runs `qmatmul` on 4-bit weights directly.
Measured on TinyLlama 1.1B (vs the F32-preload path, identical output):
- **Peak RAM ~5 GB → ~0.8 GB** (~6×) — weights stay 4-bit.
- **Load 16 s → 0.2 s** (~80×) — no upfront dequant.
- **Forward ~0.72 tok/s** — on par with F32 (the on-the-fly dequant compute balances the ~8× lower
  bandwidth on 4 cores). Fixed a nasty nested-OpenMP bug (dequant-inside-parallel-qmatmul) that
  had made it 30× slower.

**Remaining:** the on-the-fly dequant is still scalar — SIMD `vec_dot` kernels per quant type
would push CPU speed up; quantized KV cache; and the GPU forms of these kernels (Phase 8). Real
speed at scale is the GPU path.

### (original scope)
Direct GPU kernels for Q4_K/Q5_K/Q6_K/Q8_0 — matmul, attention, and FFN without dequant-to-FP16.
Quantized KV cache.

## Phase 7 — KV Cache, Batching, Scheduler 🚧
**Part 1 ✅ — `GlobalKvCache`:** paged, multi-session KV store (vLLM-style page tables) in the
`memory` module. A page holds `tokensPerPage` tokens of K+V for one layer; per-session page tables
map (session, layer, token) → a page from one shared pool, so sequences are isolated, their KV need
not be contiguous, and closing a session returns pages to the pool. `appendToken` grows pages on
boundaries and fails cleanly when the pool is exhausted; `reset` frees pages but keeps the session.
Wired into `TextModel` (one session sized to maxSeq): TinyLlama output is byte-identical to the old
contiguous cache, and the in-memory F32 forward tests still pass.

**Part 2 ✅ — scheduler + continuous batching:** `TextModel` now supports many concurrent sessions
(`openSession`/`forward(session,…)`; KV pool sized for `maxSessions`). The `scheduler` module adds a
priority request queue, admission up to `maxConcurrent` (each request gets a session and is
prefilled), and a `step()` loop that decodes one token per active request per round — finished ones
free their session so waiting requests are admitted mid-flight (continuous batching: the running set
isn't drained between requests). Streaming per-token callbacks, per-request sampling/params.
Verified on synthetic models: isolated correct output across concurrent requests, batching 5
requests through 2 slots, priority ordering, streaming. TinyLlama single-sequence output unchanged.

**Remaining:** each round still runs per-session forwards sequentially — fusing them into one
batched matmul is a GPU optimization (Phase 8). Plus GPU/memory-aware admission, and KV
compression / cross-session prefix reuse / sliding-window pruning on the cache. The scheduler is a
library; exposing it over HTTP is Phase 9.

## Phase 8 — CUDA Performance Pass / GPU inference 🚧
Moving the validated CPU forward pass onto the GPU, one kernel at a time, each verified on the T4
(via Colab) against the CPU reference — the discipline that got the CPU runtime correct.

**Part a ✅ — native quantized matmul kernel:** `qmatmulQ8_0Kernel` — a block-per-row GEMV that
dequantizes Q8_0 blocks (fp16 scale × int8) straight into the dot product, keeping the weight
quantized in VRAM (the GPU form of the CPU `qmatmul`). A `qmatmulSelfTest` checks it against a host
reference and times a 4096×4096 GEMV (reports GFLOP/s + GB/s), surfaced via `qorvix gpu`.
Compile-verified under nvcc in Docker (96/96 tests pass in-container); execution-verified on a T4
via the Colab notebook.

**Part b ✅ — real-model GPU inference.** Device kernels for the whole forward pass (RMSNorm, RoPE,
GQA attention over VRAM-resident KV, SwiGLU, residual add) plus native quantized matmuls (Q4_K,
Q6_K, Q8_0, F32), each verified against the CPU reference on a T4. `GpuModel` uploads a real GGUF's
weights to VRAM (layer weights quantized, embedding dequantized to F32, F32 norms) and runs the
forward pass dispatching per quant type; `qorvix gpu-check` gates correctness against the CPU
`TextModel`. **Verified on a T4:** gpu-check on real TinyLlama Q4_K_M matches the CPU runtime
(rel err 3.7e-06, argmax agrees everywhere), and **`qorvix generate --gpu` produces correct text
at ~66 tok/s vs the CPU's ~0.7 tok/s (~90×)**. The Colab notebook builds, self-tests, gpu-checks,
and benchmarks all of this on real hardware.

**Remaining (tuning — now directly raises tok/s):** coalesced/vectorized K-quant loads (or dp4a),
FlashAttention-style attention, CUDA graphs / kernel fusion, multi-token prefill batching, and
wiring the GPU path into the scheduler + HTTP server. Benchmark each against this ~66 tok/s floor.

## Phase 9 — API Layer 🚧
**Part a ✅ — OpenAI protocol layer (`api` module, zero external deps → builds in every preset):**
- A small in-tree JSON library (standard-conforming parser + serializer; order-preserving objects,
  string escapes incl. `\uXXXX` and surrogate pairs) — no vcpkg/third-party JSON needed.
- OpenAI schema mapping: parse `/v1/chat/completions` and `/v1/completions` requests (model,
  messages/prompt, stream, and sampling: max_tokens/temperature/top_p/top_k/min_p/penalties/seed/
  stop); build `/v1/models`, chat/text completion objects, streaming chunks, error objects, and SSE
  framing (`data: …\n\n`, `[DONE]`).
- Verified locally (standalone harness + Catch2): parsing, escaping, malformed-input rejection,
  request→struct, response/chunk shapes, SSE.

**Part b ✅ — HTTP transport + `serve`:** a from-scratch cross-platform HTTP/1.1 server
(winsock/POSIX, no vcpkg) in the `api` module, and `qorvix serve <file> [--port --ctx
--max-concurrent]` wiring it to the scheduler. Routes `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/completions` (streaming SSE + non-streaming), plus `/health` and CORS. **Verified with
curl against TinyLlama:** `/v1/models` lists the model; non-streaming `/v1/completions` returns a
proper OpenAI object ("The capital of France is" → "the city of Paris, which is the", with token
usage); streaming `/v1/chat/completions` emits the correct chunk sequence (role delta → content
deltas → `finish_reason` → `[DONE]`).

**Port allocation (decided 2026-07-24):** Qorvix reserves **2005–2010**, one contiguous block so a
full deployment never collides with itself and operators can firewall one range. Defined once in
`core/include/qorvix/ports.hpp` and pinned by tests — `kRuntime` is shipped, so renumbering it
breaks deployed client configs and the test fails loudly to force a deliberate decision.

| Port | Service | Status |
|------|---------|--------|
| 2005 | Qorvix Runtime (`qorvix serve`) | ✅ shipped (was 8080) |
| 2006 | Qorvix Gateway (auth, rate limit, model routing) | reserved — Phase 13 |
| 2007 | Qorvix Dashboard (web UI) | reserved — Phase 12 |
| 2008 | Qorvix Admin API (load/unload, introspection) | reserved |
| 2009 | Qorvix Metrics (Prometheus `/metrics`) | reserved — Phase 12 |
| 2010 | Qorvix gRPC | reserved — not yet scheduled |

Admin and Metrics are deliberately separate ports from the inference endpoint so metrics can be
scraped, and control operations firewalled, without exposing either through the other.

**Known follow-ups:** chat quality needs the model's own chat template (GGUF
`tokenizer.chat_template`) — the current generic `user:/assistant:` template mismatches
instruction-tuned models, so use `/v1/completions` for faithful output meanwhile. The server is
single-connection (one request at a time) — concurrent HTTP + the scheduler's batching needs a
threaded accept loop. WebSocket streaming and the multimodal endpoints
(embeddings/audio/images) follow their backends in Phases 11+. Boost.Beast stays an option for
production hardening.

## Phase 10 — Multi-GPU 🚧
NCCL-based tensor parallelism, pipeline parallelism, expert parallelism, load balancing across
1/2/4/8 GPUs.

**Part a ✅ — tensor-parallel sharding foundation (verified without multi-GPU hardware).**
Tensor parallelism is two separable problems: *which* slice of each weight a rank owns and where
the partial sums must be summed (pure integer math, and where essentially all the real bugs live),
and *moving bytes between devices* (NCCL). Splitting them means the first is verifiable on one GPU
— or none — by simulating the ranks; only the transport needs real multi-GPU.
- `cuda/multi_gpu.hpp`: device topology (P2P reachability matrix, NVLink-vs-PCIe via
  `cudaDevP2PAttrNativeAtomicSupported`, per-device free VRAM), the `ICollective` seam
  (`allReduceSum` — the *only* collective a TP decode step needs, twice per layer), and the
  `TensorParallelPlan`.
- `tensor_parallel.cpp` is **CUDA-free and compiles into both the real and stub builds**, so the
  split logic is unit-testable on a machine with no GPU.
- Sharding: column-parallel (`wq/wk/wv/ffnGate/ffnUp` split by *rows* = output dim, zero-copy since
  a row range is contiguous) and row-parallel (`wo/ffnDown` split by *cols* = input dim, a strided
  block gather). Weights stay quantized — a shard is a byte slice, never a dequant.
- Two constraints found and enforced rather than assumed: (1) **GQA caps TP at the KV-head count** —
  each rank must own whole KV heads or it would refetch half a head every step, so TinyLlama
  (nKv=4) tops out at TP=4 regardless of GPU count; (2) **row-parallel column splits must land on
  quantization-block boundaries** — TinyLlama's ffn=5632 is 22 super-blocks and 22 % 4 ≠ 0, so an
  even *element* split would cut at 1408 (mid-block) and slice a shared fp16 scale away from the
  quants it scales. Splitting the *block count* instead yields 6/6/5/5 and stays decodable, so TP=4
  works where a divisibility requirement would have rejected it.
- **Verified:** 12 Catch2 cases (2463 assertions) covering plan tiling/contiguity, the
  wo↔wq and ffnDown↔ffnGate pairing invariant, uneven splits, byte-exact shard gathering, and the
  rejection paths — passing in a CPU-only build, no GPU needed. Full suite 108 cases / 3851
  assertions green. `tensorParallelSelfTest()` (surfaced in `qorvix gpu`) additionally simulates a
  2- and 4-rank all-reduce on one device against the unsharded GPU GEMV.
- **Note:** TP *reassociates* the dot product, so multi-rank output is equal to single-GPU output
  only to within float rounding (~1e-7 relative here), not bit-identical. Column-parallel row
  splits, being a pure partition, remain bit-exact.

**Remaining:** Part b — NCCL transport behind `ICollective` (compile-gated like the CUDA stub/real
split; needs ≥2 GPUs to execution-verify) and a sharded `GpuModel` that runs the plan. Part c —
pipeline parallelism, expert parallelism, load balancing.

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

**Status:** Phases 0–9 substantially complete; **Qorvix runs correct GPU inference on real models.**
CPU path: correct text generation from real GGUF models with native quantized weights (~0.8 GB RAM),
a paged multi-session KV cache, a continuous-batching scheduler, and an OpenAI-compatible HTTP
server (`qorvix serve`). GPU path (Phase 8): every forward-pass kernel (Q4_K/Q6_K/Q8_0/F32 matmul,
RMSNorm, RoPE, GQA attention, SwiGLU) verified on a Tesla T4, and **`generate --gpu` produces
correct text on real TinyLlama at ~66 tok/s — ~90× the CPU** (gpu-check: rel err 3.7e-06 vs the CPU
runtime). The Colab notebook builds/self-tests/benchmarks it all on real hardware. Remaining work is
optimization (tuned kernels, FlashAttention, CUDA graphs, prefill batching), multi-GPU (Phase 10),
and the multimodal engines (Phase 11+) — the un-tuned GPU kernels mean ~66 tok/s is a floor.
