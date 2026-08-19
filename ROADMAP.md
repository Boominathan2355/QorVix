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

## Phase 4 — CUDA Backend Bring-Up ✅ (execution-verified on a Tesla T4 in Phase 8)
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

## Phase 5 — Text Runtime, End-to-End ✅
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

**Part c 🚧 — performance campaign (target ≥250 tok/s decode on the T4; currently ~87–89).**
Every change is gated on `gpu-check` argmax staying identical to the CPU, or it does not ship.
- ✅ Q6_K kernel optimization: 39 → 81–96 GB/s (~2×), generate 86.8 tok/s.
- ✅ CUDA Graphs (`1b6a015`): the decode forward is captured once and replayed per token, with
  token/pos routed through a device param buffer. Bit-identical, but only **+2.5%** (→ 89 tok/s) —
  launch overhead was not the bottleneck.
- ↩️ Vectorized Q4_K uint32 loads (`a341822`, reverted in `f9a02eb`): correct (argmax held) but
  **0% gain**. Load width is not the limiter.
- **Diagnosis:** decode is weight-bandwidth bound — 640 MB/token ÷ 11.2 ms ≈ **57 GB/s effective,
  only ~18% of the T4's ~320 GB/s peak**. Reaching ~160 GB/s would hit the 250 tok/s target.
- **Open blocker:** *why* the GEMV kernels sit so far below peak. Registers are not the limiter
  (theoretical occupancy 100%). It is likely not a pure DRAM ceiling either: Q6_K is the heavier
  format (210 B/super-block vs Q4_K's 144 B) yet reaches *higher* GB/s (75–92 vs 41–58) — if both
  were at the memory wall they would be equal. That points at latency / insufficient memory-level
  parallelism. Earlier ncu runs profiled the tiny cache-resident correctness matrix (DRAM ~1%) and
  were meaningless; `QORVIX_NO_GRAPH=1` + `scripts/colab_ncu_profile.sh` (`cd2119b`) now target the
  real generate workload. **Awaiting that T4 run** — the warp-stall breakdown picks the fix:
  *Long Scoreboard* → multi-row-per-warp GEMV (more independent load streams); *Barrier/MIO* → the
  per-super-block `__shfl` header broadcast that stalls all 32 lanes on lane 0.

**Remaining beyond the campaign:** FlashAttention-style attention, multi-token prefill batching,
and **wiring the GPU path into the scheduler + HTTP server** — today `serve` reaches only the CPU
`TextModel`, so the fast path is unreachable over HTTP (see the architecture note below).

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

**Part b 🚧 — the transport, and a sharded `GpuModel` that actually runs the plan.**

*b-1, collectives.* Part a shipped an `ICollective` seam whose `makeSimulatedCollective()`
`allReduceSum` was **a no-op** — it called itself the verification vehicle and summed nothing (the
Part-a self-test added the partials inline, so the seam itself was never once exercised). Part b
replaced it with one that has three real implementations and no test doubles.
- **The seam is group-scoped now** (`ICollectiveGroup::allReduceSum(float* const* bufs, n)`), not
  per-rank. Qorvix is single-process by design, so the driver is ONE host thread launching each
  rank’s kernels on that rank’s stream — the launches are asynchronous, so the ranks still execute
  concurrently and the all-reduce is the only place they meet. A per-rank `allReduceSum(buf)` would
  make rank 0 block for a rank the thread has not driven yet; NCCL’s own answer to that is
  `ncclGroupStart/ncclGroupEnd`, i.e. reducing at GROUP granularity. Taking every rank’s buffer at
  once puts that fact in the type instead of a comment, and makes the deadlock unrepresentable.
- **Three transports behind one selection point** (`makeCollectiveGroup`): the world=1 no-op;
  **host-staged** (D2H each rank, sum on the CPU, H2D the total back); and **NCCL**
  (`ncclCommInitAll`, single process, one rank per device), compile-gated by `QORVIX_ENABLE_NCCL`
  → `QORVIX_WITH_NCCL` exactly like the CUDA stub/real split. Host-staged is not a mock: it is the
  correct path when `cudaDeviceCanAccessPeer` reports no P2P route between two devices.
- And it is the answer to *how do you verify multi-GPU without multi-GPU*: host-staged does not care
  whether two ranks name the SAME device (NCCL rejects that outright), so **a rank is a
  (device, shard) pair and TP=4 runs on a single T4** — real shards, real per-rank KV caches, real
  all-reduces. Only bandwidth is missing, not coverage.
- `collectiveSelfTest()` (surfaced in `qorvix gpu`) checks the transport against a closed-form
  expected sum and inspects **every** rank’s buffer, because “the total lands on all of them” is the
  half of the contract a model would otherwise discover as garbage on ranks 1..N-1.

*b-2, the sharded model.* `createShardedGpuModel()` returns a plain `GpuModel`, so `GpuEngine`, the
scheduler, `serve` and the entire HTTP layer drive tensor parallelism with **zero** changes above the
seam. There is no “multi-GPU code path”; there is one model interface with two implementations.
- Per rank: wq/wk/wv and ffnGate/ffnUp sharded by rows, wo/ffnDown by columns, its own KV cache
  sized to its KV heads, scratch at its local dimensions. Norms are replicated — recomputing an
  RMSNorm on every rank is far cheaper than the collective a broadcast would cost.
- **Two all-reduces per layer** (after the o-projection, after the FFN down-projection), plus one
  for the embedding.
- The **embedding table is column-sharded, not replicated**: rank r fills only its columns of a
  zeroed x, and an all-reduce over disjoint contributions IS the concatenation (bit-exact, and it
  reuses the one collective already there). Replicating it would have cost 262 MB of F32 per rank on
  TinyLlama — more than a Q4_K_M copy of the entire model — so tensor parallelism would have saved
  no memory at all.
- The **LM head is column-parallel and needs no collective**: the logits are host-bound anyway, so
  each rank’s device-to-host copy lands at its own offset and the gather is free. `allReduceSum`
  therefore remains the ONLY collective in the whole model, which is what keeps the transport seam
  small enough to have three honest implementations.
- RoPE, the KV store and attention are entirely local: a rank never reads a peer’s KV. RoPE’s angle
  depends only on the position and the offset *within* a head, so rotating a rank’s own head range
  needs no global head index.

*b-3, the gate.* `qorvix tp-check <gguf> --tp N` compares the sharded logits against the
**unsharded GPU** logits on the same real model. The Vulkan bring-up established that
self-consistent self-tests miss real bugs and that only an argmax-vs-reference comparison on a real
model catches them, so that is the shape this takes. It gates on argmax parity plus relative error
< 1e-3 — tighter than `gpu-check`’s 5e-2, because both sides run the identical kernels here and the
only difference is the reassociation of the sums, so a larger gap is a bug rather than float noise.
The two models are built one at a time (holding both would need exactly the VRAM the sharding
exists to avoid). `--tp N` / `--devices a,b,c` also reach `generate`, `serve` and `bench`, and
announce when ranks share a device so a verification run cannot be misread as a benchmark.
`scripts/colab_tp_check.sh` runs the gate at TP=2 and TP=4 on one T4, including the expected
**refusal** at TP=8 (TinyLlama’s 4 KV heads cap it).

**Status:** both configurations build clean and the suite is green in each — CUDA 12.6 + NCCL 2.22
(242 cases / 5745 assertions) and the CPU-only stub (242 / 5752); the `[tp]` tag is 17 cases /
~2520 assertions, up from Part a’s 12. The collective contract is pinned in the CPU-only build
because the host-memory group is toolkit-free, exactly like the sharding math.

**Not yet executed on real hardware.** `scripts/colab_tp_check.sh` is the T4 gate and has not been
run, so the sharded forward pass is compile-verified only — the same state Part a was in before
`tensorParallelSelfTest` ran on a device. NCCL’s own transport and NVLink/PCIe P2P behaviour need
≥2 GPUs and stay untested until then.

**Remaining:** Part c — pipeline parallelism, expert parallelism, load balancing.

## Phase 11 — Multimodal Expansion

The original scope bundled six engines: vision (Qwen-VL, Llama Vision, MiniCPM-V, InternVL; OCR,
grounding), audio (STT/TTS/voice cloning), image generation (Flux/SDXL/SD), embeddings
(text/vision/audio/cross-modal), RAG (loaders, chunking, hybrid search, vector store), and
multi-agent workflows. Split so each part can end in something that compiles, runs, and is gated.

### Phase 11a — Text embeddings + RAG ✅

**Encoder engine (`embeddings` module).** BERT-family encoders from GGUF: `IEmbeddingEngine` is a
**second seam** beside `IInferenceEngine`, built by `createEmbeddingEngine` in the same
`backend.hpp`. Not a subclass — that seam's sessions *are* KV-cache allocations, its `forward` is
one autoregressive step at a position, and its output is vocab logits; an encoder has no KV cache,
consumes all N tokens at once, and has no LM head at all. Subclassing would have meant three stub
methods and a `forward` that doesn't return logits, which is the fake seam Phase 8.5 removed. One
seam per task, one factory per seam, still zero parallel paths.

Bidirectional attention, post-norm LayerNorm with bias, GELU FFN, learned position table, token-type
embeddings, and cls/mean/last pooling read from `<arch>.pooling_type`. `ModelConfig` gained an
`ArchFamily` and seven encoder fields whose defaults leave every decoder path byte-identical;
`createEngine` and `TextModel::fromGguf` both reject encoders explicitly, so widening the allowlist
could not silently route a `bert` file into the generation path.

**WordPiece tokenizer.** `TokenizerModel::WordPiece` with BERT's BasicTokenizer normalization
(control deletion, punctuation isolation, CJK per-character splitting, case-folding + accent
stripping over Latin-1/Latin-Extended-A) and explicit `[CLS]`/`[SEP]` wrapping. Supports **both**
vocabulary conventions: HuggingFace's `##` continuations and llama.cpp's SentencePiece-shaped
`▁word` markers, detected from the vocabulary — real bert GGUFs use the latter, and guessing wrong
is silent (every word still resolves to *something*, usually `[UNK]`, and the vector stays finite
and unit-norm while meaning nothing).

**The gate: `qorvix embed-check`.** Every prior phase diffed against an existing implementation;
there is no second embedding implementation, so ground truth is imported from
**sentence-transformers** (independent of both this codebase and llama.cpp, so it validates the GGUF
conversion too) by `scripts/capture_embed_reference.py`, committed under `tests/data/`. Tiered so a
failure is diagnosable: tokenizer parity (exact ids), vector parity (cosine), then invariants and
triplet ordering that need no fixture and always run.

| model | tokenizer parity | vector parity | gate |
|---|---|---|---|
| bge-small-en-v1.5 **F16** | 7/7 exact | **min cos 1.00000** | PASS (`--min-cos 0.999`) |
| all-MiniLM-L6-v2 **Q4_K_M** | 7/7 exact | min cos 0.97598 | PASS (`--min-cos 0.97`) |

Cosine 1.00000 against fp32 settles the GELU variant empirically: exact erf, not the tanh
approximation. On the quantized model, real text lands at 0.986–0.994 and the only outlier is the
empty string (2 tokens, where mean pooling gives quantization noise nothing to average against) —
which is why the gate prints a per-probe breakdown on failure.

**Serving.** `POST /v1/embeddings` (all four OpenAI `input` shapes, base64 encoding — the default
its Python SDK requests — `dimensions` truncation, `--max-batch` bound), and
`serve <chat.gguf> --embed-model <encoder.gguf>`: **two engines in one process**, per SPEC, each
with its own tokenizer and its own mutex. `/v1/models` lists both. Without the flag the route
returns 501, not 404 — the route exists, this process just has no encoder loaded.

**RAG (`rag` module).** Token-aware chunking using the embedding model's own tokenizer (a
character-count chunker cannot honour a token budget, and over-long chunks are silently truncated at
embed time), `.txt`/`.md`/`.csv`/`.tsv` loaders, a flat vector store with exact cosine top-k, BM25
over whole-word terms (deliberately not WordPiece — `##ation` is not a term anyone searches for),
and Reciprocal Rank Fusion (rank-based, so the incomparable cosine and BM25 scales never need a
calibration that does not exist). Native `.qvx` format shaped like GGUF: magic, version, validated
header, contiguous float matrix, then chunk records and the lexical index in the same file.
Verified self-hosting on the repo's own `docs/`.

**Performance.** Two CPU wins, both bit-identical and both invisible on the decode axis:
`qmatmul` was dequantizing one *block* per call, which for F16 (`blockSize == 1`) meant one call per
*element* (5.51 s → 0.73 s on a 4-token embed); then `qmatmulN` batches the GEMV across tokens
(**27.72 → 225.65 embed tok/s**, and RAG indexing 562 s → 20 s). See BENCHMARKS.md.

**Explicitly not done in 11a:** vision, audio, image generation, **vision/audio/cross-modal
embeddings** (SPEC lists four; this ships text only), multi-agent workflows, reranker models, GPU
embedding backends, PDF/DOCX loaders (gated on a from-scratch DEFLATE decoder — doing them badly
would silently poison every embedding derived from them), SQLite/Postgres vector stores, ANN
indexing, `nomic-bert` (rope-based encoders are refused rather than approximated), and multi-model
name routing in `serve`.

### Phase 11b-1 — Vision encoder ✅

**CLIP ViT tower + LLaVA projector (`vision` module).** SPEC's "Image → Vision Encoder →
Projected Embeddings" stage. A ViT is architecturally a BERT with patch embeddings, so this reuses
Phase 11a wholesale — bidirectional attention, LayerNorm with bias, the batched quantized GEMV.
The differences are called out where they occur: **pre-norm** rather than post-norm, **quick-GELU**
rather than GELU, a prepended class token, and a learned position table over patches.

`qorvix image-embed <mmproj.gguf> --image f.png [--project]` produces 576 patch tokens, either as
vision hidden states (1024-d) or projected into the language model's space (4096-d).

**Image loading, from scratch.** A DEFLATE decoder (RFC 1951 — fixed and dynamic Huffman, stored
blocks, the 32 KiB window) and a PNG reader on it, plus BMP and PPM. Verified **byte-exact against
PIL**: 562,500 bytes, zero differences. The same inflate is the honest prerequisite for the PDF
(FlateDecode) and DOCX (ZIP) loaders `rag/loaders.hpp` refuses — one decoder, three consumers.

**Three things read off the real file rather than assumed**, each of which would have been silent:
`ffn_down` is the EXPANSION and `ffn_up` the CONTRACTION (llama.cpp's CLIP converter inverts the
usual naming); `clip.use_gelu = false` selects **quick-GELU**, a third variant not interchangeable
with the two already in `ops`; and the normalization mean/std live in the file.

**The gate: `qorvix vision-check`**, tiered like `embed-check` and diffed against transformers'
`CLIPVisionModel` (fp32) via `scripts/capture_vision_reference.py`:

| tier | result |
|---|---|
| preprocessing | max \|diff\| **1.20e-07** (5 pixel probes + mean/abs-mean) |
| patch token row 0 | **cos 1.0000000**, max \|diff\| 6.93e-04 |
| all 576 patch rows | **cos 0.9999996** |

The residual 6.9e-04 is F16-vs-fp32 weight quantization, which is what it should be.

The tiering earned itself immediately: **both bugs found were in preprocessing, not the
transformer.** Pillow computes its resample window as `(int)(center ± support + 0.5)` *truncated*
(floor/ceil pulls in an extra source pixel — invisible on gradients, 1.8e-1 wrong on a
checkerboard), and Pillow clips to uint8 *between* the horizontal and vertical passes (a
float-throughout pipeline drifts ~half a level per pass, 6e-3 on every pixel). With one aggregate
verdict both would have read as "the transformer is slightly wrong".

**Deferred, with the reason stated:** JPEG (named explicitly in the error, so a user knows it is a
missing feature rather than a corrupt file), Adam7 interlacing, and wiring the projected features
into the decoder for actual image chat — that needs `TextModel` to accept input embeddings rather
than token ids.

### Phase 11b-2 — Vision-language chat ✅

**The seam, widened.** `IInferenceEngine` took a token id per step and an image patch has no id.
Rather than mint a fake one (it would collide with a real token and poison the sampler's
repetition history), the interface gained a second entry point: `forwardEmbedding(session, const
float*, pos)` runs the stack from a ready-made `[d_model]` vector, guarded by
`acceptsInputEmbeddings()`. `TextModel::forward` was split at the embedding lookup — everything
after it is now shared, so the two entry points cannot drift. `forwardInput(InputToken)` dispatches
between them, and that is what the prefill loops call.

Device backends keep the embedding table in VRAM and look rows up on-device, so accepting a host
vector is a new upload path and kernel entry point, not a wrapper. Until that exists **CPU is the
multimodal backend**, and `serve --mmproj --gpu` is refused at startup rather than silently
degraded — the same rule `createEmbeddingEngine` follows for GPU encoders.

**Prompt assembly (`runtime/multimodal.hpp`).** `MultimodalPrompt` interleaves token ids with
image-feature blocks; `partsFromPrompt` splits the rendered chat prompt at `<image>` markers and
drops each image into its slot (no marker → images are prepended, the LLaVA convention). BOS and
EOS bracket the **sequence**, not each segment — which required `Tokenizer::encode` to gain
explicit EOS control, since it appends EOS on every call and a split prompt would otherwise bury
one mid-sequence, where every model reads it as "the conversation ended here". Text-only prompts
are byte-identical to before, and the scheduler now runs **one** prefill path for both.

Two facts the sampler depends on: image positions contribute nothing to the repetition-penalty
history (no id to penalize), and `prompt_tokens` counts prefill **positions**, so 576 patches are
included — reporting only the text would understate the context consumed.

**Surfaces.**
- `qorvix generate <model.gguf> --mmproj <clip.gguf> --image f.png --prompt "USER: <image>\n..."` —
  `--image` is repeatable, and order pairs images with markers.
- `serve --mmproj <clip.gguf>` — multimodal `/v1/chat/completions` with `image_url` content parts,
  behind its own mutex (CLIP scratch lives in members; the encode is the long pole, so it does not
  queue behind generation). Projector/decoder width is checked **at startup**, not per request.
- Images arrive as `data:` URIs. A remote `http(s)://` URL is **refused with an explanation, not
  fetched**: server-side retrieval of a client-supplied URL is an SSRF primitive, so the capability
  is declined outright rather than added with a blocklist. Every OpenAI SDK already inlines images.
- The HTTP reader gained a **32 MB request-body cap answering 413**. It had none: bodies used to be
  small JSON, so an attacker-controlled `Content-Length` sized a buffer nobody had reason to send.
  Images make multi-MB bodies routine, which turned a latent hole into a reachable one. Verified:
  a 40 MB body is refused before a byte of it is read, a 3 MB one reaches the handler.
- **Max 8 images per request.** Each costs a full CLIP encode (tens of seconds, under one mutex)
  and 576 context positions, so an unbounded count lets one caller hold the server for an hour.
  The body cap does not constrain this by itself — a hundred small PNGs fit inside it easily.

**The gate: `qorvix vlm-check <model.gguf>`.** Tiers 1 and 2 need no vision model and no captured
reference, because the property is self-checking: feeding a token's **own** embedding row through
the new seam must reproduce, **bit for bit**, what feeding its id through the old path produced.

| tier | asserts | measured (TinyLlama 1.1B Q4_K_M) |
|---|---|---|
| 1 | single-step splice identity | max \|diff\| **0.00e+00**, 0 argmax mismatches — **PASS** |
| 2 | full-prefill identity, so a splice that only corrupts KV cannot pass tier 1 | max \|diff\| **0.00e+00**, argmax identical — **PASS** |
| 3 | projector width vs `d_model`, spliced prefill runs, logits finite | needs a matched mmproj (below) |

Tier 3 is labelled a **smoke test in its own output**: no LLaVA reference is captured in this
repo, so it asserts shape and well-formedness, not output fidelity. It is also **the one thing not
run end-to-end here** — the mmproj on disk is LLaVA-7B's (4096-d) and the only decoder on disk is
TinyLlama (2048-d), so the pair is refused rather than exercised. `models/README.md` gives the
exact command for the matching decoder. Capturing a real LLaVA reference is the next step.

Ordering is deliberate throughout: opening a GGUF only maps its header, so `d_model` and the
projector width are both known before either the CLIP encode or the decoder's weight load. A
mismatched pair therefore costs **1 second** rather than a full encode of an image whose features
were always going to be rejected — the same principle in `generate`, `serve` and `vlm-check`.

**Deferred, with the reason stated:** a captured LLaVA reference (tier 3 is a smoke test until
then); GPU/Vulkan `forwardEmbedding`; multi-image *layout* variants (LLaVA-1.6 tiling); and image
features in the paged prefix cache — `sharePrefix` keys on token ids, which image positions do not
have.

### Phase 11b-3 — Audio / image generation ⬜
Audio engine (Whisper: FFT + mel + encoder-decoder with cross-attention) and image generation
(SDXL/Flux — not feasible on this CPU-only box; gated on GPU hardware). `audio/`, `image/` and
`agents/` are still 0 files.

### Phase 11c — GPU embedding + vision backends ⬜🖥️
CUDA and Vulkan implementations of `IEmbeddingEngine` and the CLIP tower. `createEmbeddingEngine` reports honestly for
now rather than silently falling back to CPU. Note `buildGpuModel`/`buildVulkanModel` are *not*
reusable (they unconditionally read `L.ffnGate`); `detail::toGpuWeight`/`toVkWeight` are.

Phase 11b-2 adds a third item here: **`forwardEmbedding` on the device backends**. They hold the
embedding table in VRAM and look rows up on-device, so taking a host `[d_model]` vector needs an
upload path and a kernel entry point that skips the lookup — not a wrapper. Until it exists,
`acceptsInputEmbeddings()` returns false there and the multimodal surfaces refuse rather than
silently degrade.

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

**Status (2026-08-13):** Phases 0–9 complete, Phase 8.5 (cross-vendor + unified engine) complete,
Phase 10 started, **Phase 11a (text embeddings + RAG) complete**. **Qorvix runs correct inference on
real models across three backends behind one seam** — CPU, CUDA (**114.82 decode tok/s measured on a
Tesla T4**, argmax parity, see BENCHMARKS.md), and Vulkan (cross-vendor, argmax parity verified
GPU-free on lavapipe, rel err 2.6e-06) — **and correct text embeddings behind a second seam**
(bge-small F16 matches sentence-transformers at cosine 1.00000). Test suite green; the CPU-only,
CUDA, and Vulkan builds all compile and link.

**Five correctness gates, one per numerical path**, all CLI rather than CTest because each needs a
GGUF the test image does not have: `gpu-check` (CUDA vs CPU logits), `vulkan-check` (Vulkan vs CPU
logits), `tp-check` (tensor-parallel vs unsharded GPU logits), `embed-check` (embeddings vs a
sentence-transformers reference), and `vision-check` (CLIP features vs a transformers reference).

- **Unified backend ✅** — one `IInferenceEngine` seam, three implementations (CPU/CUDA/Vulkan), one
  `createEngine` factory, one generation loop. `generate` and `serve` reach any backend via
  `--gpu` / `--vulkan` / `--auto`; `qorvix backends` lists them.
- **CPU path ✅** — correct generation from real GGUF with native quantized weights (~0.8 GB RAM),
  paged multi-session KV cache, continuous-batching scheduler, OpenAI-compatible HTTP server.
- **CUDA path ✅ correct, 🚧 fast** — every kernel verified on a T4; CUDA Graphs shipped; decode is
  **memory-instruction-issue bound, not bandwidth bound** (T4: 28% DRAM but 84% L1TEX and 46% of
  warp stalls on a full MIO queue — earlier notes here called it bandwidth-bound, which the `ncu`
  data contradicts). Phase 8c works the instruction count, not the byte count.
- **Vulkan path ✅ correct, 🚧 fast** — full forward verified on lavapipe; single-session and
  correctness-first (throughput pass — device-local buffers, command-buffer reuse, subgroups — is
  future work).
- **Multi-GPU 🚧** — sharding math verified without hardware (10a); the collective transport
  (host-staged + NCCL, compile-gated) and a sharded `GpuModel` behind the same `GpuModel` interface
  (10b) compile but have not yet run on a device. Because a rank is a (device, shard) pair, ranks may
  share one GPU — so `qorvix tp-check --tp 4` executes the real TP path on a single T4.
- **Plugins ✅** — `IPlugin` + `PluginRegistry` (hot-load/unload), example plugin, `qorvix plugins`
  (Phase 1, tested). `plugins/` is real, not a placeholder.
- **Measurement ✅** — `qorvix bench` (backend-agnostic, median-of-runs, JSON), BENCHMARKS.md as the
  single source of truth, and performance-regression tests in the suite.
- **Embeddings + RAG ✅** — BERT/WordPiece encoders from GGUF behind a second seam
  (`IEmbeddingEngine`), `qorvix embed`, `POST /v1/embeddings`, `serve --embed-model` hosting a chat
  and an embedding model in one process, and a RAG layer (chunking, `.qvx` vector store, BM25,
  RRF hybrid search). Gated by `embed-check` against a sentence-transformers reference:
  bge-small F16 min cos **1.00000**, all-MiniLM Q4_K_M **0.97598**, both with exact tokenizer parity.
- **Vision ✅** — CLIP ViT-L/14-336 tower + LLaVA projector from a `clip` mmproj GGUF, with a
  from-scratch DEFLATE/PNG decoder and CLIP preprocessing. Gated by `vision-check` against
  transformers at **cos 1.0000000** (preprocessing exact to 1.2e-07).
- **Vision-language chat ✅ (CPU only)** — `IInferenceEngine::forwardEmbedding` splices projected
  image patches into the decoder's input embeddings; `generate --mmproj --image` and
  `serve --mmproj` (multimodal `/v1/chat/completions`). Gated by `vlm-check`: the splice is
  **bit-identical** to the id path when fed a token's own embedding row. Device backends refuse
  images rather than degrading — the GPU embedding upload path is Phase 11c.
- **Not started (empty dirs, 0 files each)** — `image/`, `monitoring/`, `agents/`,
  `audio/`, `ui/`, `cli/` — Phase 11b–12 placeholders, scaffolded only when their phase begins
  (see the status-annotated backlog above).

## Phase 8.5 — Cross-vendor Vulkan backend + unified engine ✅ (retrofit)

Added a second GPU backend so Qorvix runs on **any vendor** (NVIDIA / AMD / Apple via MoltenVK /
Intel), not NVIDIA-only, and unified all backends behind one seam — closing the architectural gap
that was noted here previously.

**Vulkan compute backend (`vulkan` module).** GLSL compute shaders → SPIR-V, embedded at build time
via glslang. Q8_0/Q4_K/Q6_K GEMV (tree-reduction, one workgroup/row), rmsnorm/rope/swiglu/add/GQA
attention, plus embed/kv-store/matmul-f32, chained into a full per-token forward in one command
buffer with the KV cache resident. `VulkanModel`/`createVulkanModel` mirror the CUDA `GpuModel`.
Verified **GPU-free** on Mesa's software device (lavapipe/llvmpipe) in Docker — the same argmax-vs-CPU
gate CUDA gets on a T4, but on CPU: `vulkan-check` on real TinyLlama 1.1B Q4_K_M matches the CPU
runtime (rel err 2.6e-06, argmax identical everywhere), and `generate --vulkan` produces identical
text. Two bugs the lavapipe loop caught that the self-tests missed: a RoPE dispatch under-launch
(64-lane workgroup sized with a /256 group count) and sequential-fp32 GEMV accumulation drift
(~16%, argmax-preserving) — both fixed. Throughput on the software device is a correctness figure,
not a perf one; real-hardware tuning is future work.

**Unified backend (one seam, one factory, one path).** `runtime::IInferenceEngine` is the single
seam; CPU (`TextModel`), CUDA (`GpuEngine`), and Vulkan (`VulkanEngine`) are its three
implementations. `core/include/qorvix/backend.hpp` is the one construction point —
`createEngine(Backend, GgufFile, maxSeq, maxSessions)` returns a `unique_ptr<IInferenceEngine>`
(CPU keeps the mmap; device backends upload + release it), with `backendAvailable` /
`selectBestBackend` (auto = CUDA > Vulkan > CPU). `runGenerate()` is one backend-agnostic loop;
`generate` and `serve` both go through the factory — no per-backend branch or parallel loop remains
above the seam, and `serve` now reaches Vulkan too. New `qorvix backends` command; `--gpu` /
`--vulkan` / `--auto` flags. **This resolves the former gap** below: there is now a real
`IExecutionEngine`-style seam, a second backend is an interface implementation (not a parallel
namespace), and every backend is reachable over HTTP.

**Still open from the old gap:** the GPU/Vulkan paths use flat device-resident KV, not the
vLLM-style paged `GlobalKvCache` (that remains CPU-only); and real batching across requests
(`forwardBatch`) is still sequential. *(The Vulkan engine is now multi-session — per-session KV
slices like `GpuModel`, verified by `multiSessionSelfTest` — so `serve --vulkan --max-concurrent N`
works.)*

### Cross-cutting backlog (status-annotated)

Consolidated from a feature audit. **Legend:** ✅ done · 🟡 partial · 🖥️ needs GPU hardware to build
or verify (not possible in this CPU-only dev/CI environment) · ⬜ not started.

- **Performance:** Vulkan kernel opt ⬜ · CUDA throughput 🟡 (Phase 8c, T4) · command-buffer reuse /
  graph replay (Vulkan) ⬜ · device-local buffers + staging ⬜ · subgroup/warp opt ⬜ · kernel
  fusion ⬜ · per-GPU autotune ⬜. *(All 🖥️ for real numbers — lavapipe perf is meaningless.)*
- **Backends:** CPU ✅ (all CPUs) · CUDA ✅ (NVIDIA) · Vulkan ✅ (cross-vendor: AMD/Apple/Intel/NVIDIA)
  · native HIP/ROCm ⬜🖥️ (AMD) · native Metal ⬜🖥️ (Apple) · native SYCL/oneAPI ⬜🖥️ (Intel).
  *(Vulkan already covers AMD/Apple/Intel functionally; the three native backends are a perf option,
  each gated on its own hardware.)*
- **Inference:** continuous batching ✅ (Phase 7) · paged KV ✅ (CPU) · prefix cache ✅ (`sharePrefix`)
  · scheduler ✅ · sliding-window attn ⬜ · speculative decoding ⬜ · multi-GPU 🟡 (TP math done,
  NCCL 🖥️).
- **Models:** GGUF F16/BF16/Q4_0..Q8_0/K-quants ✅ · more quant formats (IQ-quants) ⬜ · MoE ⬜ ·
  vision ✅ (CLIP tower + VLM chat, CPU only) · text embeddings ✅ (BERT/WPM; bge-small + all-MiniLM verified) ·
  vision/audio/cross-modal embeddings ⬜ · reranker ⬜.
- **Serving:** OpenAI API ✅ (`/v1/models`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`) · two models in one process ✅ (`serve --embed-model`) · SSE streaming ✅ ·
  multimodal `image_url` content parts ✅ (`serve --mmproj`, data: URIs only — remote URLs are
  refused, not fetched) ·
  WebSocket ⬜ · auth ⬜ (Phase 13, port 2006) ·
  Prometheus `/metrics` ⬜ (Phase 12, port 2009) · rate limiting ⬜.
- **Memory:** paged KV + eviction/offload ✅ · GPU memory pooling 🟡 · zero-copy 🟡 · prefetch ⬜ ·
  graceful OOM ⬜.
- **Testing:** unit suite ✅ (106 cases / 1289 checks incl. embeddings + RAG) · cross-vendor bench ⬜🖥️ · perf regression ⬜🖥️ ·
  multi-GPU CI ⬜🖥️ · long-context validation ⬜.
- **Dev-ex:** `backends`/self-tests ✅ · backend auto-select ✅ · benchmark tool ✅ (`qorvix bench`
  + BENCHMARKS.md + regression tests) · profiler integration 🟡 (`colab_ncu_profile.sh`) · runtime
  config ⬜ · docs 🟡.
- **Extensibility:** plugin system ✅ (`plugins/` — `IPlugin`, `PluginRegistry`, hot-load/unload,
  `qorvix plugins`, example plugin + test; Phase 1).
- **CPU generalization (one backend, best ISA at runtime — the CPU analogue of one Vulkan backend
  for all GPUs):** runtime CPU feature detection + SIMD dispatch ✅ (`runtime/cpu_features.cpp`,
  `qorvix cpuinfo`) — a portable default build auto-uses AVX2 on capable x86 with NO -march=native
  (was gated behind QORVIX_NATIVE), scalar fallback everywhere; verified + tests green with AVX2
  active. NEON/SVE (ARM/Apple Silicon) 🟡 (kernel written, aarch64-compile-gated, hardware-unverified)
  · AVX-512 kernel ⬜ · RISC-V vector ⬜🖥️ · SIMD dequant (the actual CPU-decode bottleneck — dot is
  already SIMD) ⬜ · NUMA-aware KV/weight placement ⬜🖥️ · thread affinity / pinning ⬜🖥️.
  *(x86 verifiable here; ARM/RISC-V/NUMA need their own hardware, like HIP/Metal/SYCL on the GPU side.)*
- **Platform:** Linux ✅ · Windows/macOS/AMD/Intel GPU validation ⬜🖥️.
- **Production:** model hot-swap/cache ⬜ · graceful OOM ⬜ · fault recovery ⬜ · telemetry ⬜.
- **Empty module dirs (not started; scaffolded only when their phase begins, to keep the build free
  of content-less libraries):** `image/` ⬜ (Phase 11b-3 — image generation) · `monitoring/` ⬜
  (Phase 12 — Prometheus/metrics exporter, port 2009) · `audio/` · `agents/` · `ui/` (Phase 12) ·
  `cli/` — all 0 files today. (`embeddings/` and `rag/` are real as of Phase 11a; `vision/` as of
  Phase 11b-1.)

### Current priorities (2026-07-29)

Measurement before features — everything below is validated against `qorvix bench` / BENCHMARKS.md.

1. **CUDA optimization** 🟡 — *highest priority.* Phase 8c. The Q4_K GEMV re-blocking **shipped**:
   86.65 → **114.82 tok/s** on the T4 (+32.5%), argmax parity held. Next, per the decision rule:
   the same re-blocking for Q6_K (75% L1TEX, still a `__shfl` broadcast for `d`, still a per-element
   scale decode), then staging `x` in shared per block. Each behind the same measure-first gate.
2. **Vulkan optimization** 🟡 — device-local buffers **done** (d8cb8c5: weights were in host RAM,
   streamed over PCIe every token — invisible on lavapipe, which has one memory type flagged both
   DEVICE_LOCAL and HOST_VISIBLE). Unmeasured: the T4 run at cfd1fa9 predates it and timed out at
   900 s. Still open: command-buffer reuse (~377 dispatches/token, each re-allocating a descriptor
   set), device-local KV, subgroups.
3. **Real-hardware benchmarks** 🟡🖥️ — T4 CUDA row is filled. Still open: T4 Vulkan, and
   RTX / AMD / Intel via `scripts/colab_bench.sh`.
4. **Batched prefill** 🟡 — measured T4 prefill (99 tok/s) is barely above decode (87), the
   signature of a token-at-a-time prefill. GEMV → GEMM for the prompt phase is a large, separate
   win that does not touch `decode_tok_per_sec`. **The CPU kernel now exists**: Phase 11a's
   `qmatmulN` is exactly this — dequantize each weight block once, fold it into all N dot products
   — and it took the encoder from 27.7 to 225.7 tok/s. Wiring it into `TextModel`'s prefill is the
   remaining half; do not build it twice.
5. **Documentation & release** ⬜ — publish results once numbers stabilize.
6. **Native vendor backends** ⬜🖥️ — the compute-backend set is CPU · CUDA · Vulkan · **HIP · Metal
   · SYCL**, of which the last three are not started:
   - **HIP/ROCm** ⬜🖥️ — AMD native fast path (needs an AMD GPU + ROCm).
   - **Metal** ⬜🖥️ — Apple Silicon native fast path (needs macOS).
   - **SYCL/oneAPI** ⬜🖥️ — Intel Arc / Xe native fast path (needs an Intel GPU + oneAPI).

   All three are a *performance* option, not a coverage one: Vulkan already runs on AMD, Apple and
   Intel functionally, which is why it was built first. None starts before cross-vendor Vulkan
   performance stabilizes — and each needs its own hardware, so none can be verified on this box
   (same constraint as ARM/RISC-V on the CPU side).

### Former architectural gap (now resolved by Phase 8.5)

The layering was real for the CPU stack but **short-circuited for the GPU stack** — two parallel
paths, not one:

- `serve` → Scheduler → paged KV → CPU `TextModel` → CPU qmatmul  *(reachable over HTTP)*
- `generate --gpu` → `GpuModel` directly, own loop, own flat VRAM KV  *(CLI only)*

`scheduler/` and `api/` contained **zero** CUDA references. Phase 8.5's unified engine fixes the
seam problem (a backend is now an `IInferenceEngine` implementation, reachable over HTTP via
`createEngine`); the GPU paged-KV and true batched `forwardBatch` items remain (listed above).
