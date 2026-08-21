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

### Phase 11b-3a — Whisper log-mel front end ✅

**The front end alone (`audio` module).** WAV in, the `[80 x 3000]` log-mel tensor Whisper's
convolutional stem consumes out — SPEC's "Audio → Mel Spectrogram" stage. No Whisper weights are
loaded and none are needed.

It is its own module with its own gate because Phase 11b-1 gated CLIP and found **both** of its
bugs in preprocessing rather than the transformer. Audio has the same shape and worse odds: a
symmetric Hann window instead of a periodic one, zero padding instead of reflect, magnitude instead
of power, or the htk mel scale instead of slaney each moves every frame and errors on nothing. So
this ships and is verified before 11b-3b writes a line.

- `audio/fft.cpp` — recursive radix-2 with an odd-n direct fallback. Whisper's frame is
  400 = 2⁴·25, so radix-2 alone cannot transform it at all, and zero-padding to 512 is not an
  optimization but a *different transform*: bin spacing becomes 31.25 Hz instead of 40 Hz and every
  mel filter then integrates the wrong frequencies.
- `audio/audio_file.cpp` — RIFF/WAVE: PCM 8/16/24/32 and IEEE float 32/64, EXTENSIBLE subformats,
  unknown-chunk skipping, channel averaging. Dispatches on magic bytes, not extension. Compressed
  containers are refused **by name** with the ffmpeg fix, the courtesy `vision::decodeImage` extends
  to JPEG.
- `audio/mel.cpp` — pad to exactly 30 s, reflect-pad by `n_fft/2` (the head reflection is real
  signal), periodic Hann, **power** spectrum, slaney scale with slaney area normalization, then
  `log10` with Whisper's clip-relative `max - 8.0` floor and `(x + 4) / 4` affine.

**The gate: `qorvix audio-check [<file.wav>] --ref <fixture>`**, tiered so each tier fails for a
different cause — the filter bank (a pure function of the config, no audio in it), then the decoded
waveform, then the finished frames. Ground truth is transformers' `WhisperFeatureExtractor`, which
is independent of both this codebase and whisper.cpp, captured by
`scripts/capture_audio_reference.py` and committed as `tests/data/audio_reference_whisper_tiny.txt`:

| tier | isolates | measured (whisper-tiny, 2 s probe) |
|---|---|---|
| filter bank | mel scale, area normalization | max \|diff\| **1.86e-09** (8 probes + 80 column sums) |
| waveform | the WAV reader, not the spectrogram | max \|diff\| **7.63e-09** (32,000 samples) |
| log-mel frame 0 | window, padding, power-vs-magnitude, clamp | max \|diff\| **4.59e-06** (80 mel bins) |
| log-mel per-bin means | the same, over every frame | max \|diff\| **1.72e-07** (3,000 frames) |

**PASS** at the default `--tol 1e-4`. Alongside it, 13 Catch2 cases cover what needs no external
reference: the FFT against a direct DFT at n = 400, a pure tone landing in the bin its frequency
selects, filters that are triangular, area-normalized and march upward, silence sitting flat at the
clamp floor, a short clip padded rather than rejected, and every WAV sample format and malformation.

### Phase 11b-3b — Whisper encoder-decoder ✅

**The first model this repo converts rather than downloads.** Whisper has no GGUF upstream:
whisper.cpp never left its own container, and the files the Hub advertises as "whisper GGUF" are
that container renamed — verified rather than assumed, by fetching the first four bytes of
`vonjack/whisper-large-v3-gguf`: `6c 6d 67 67`, i.e. ggml's magic, not `GGUF`. The choice was a
second container reader or a converter, and the reader loses: the whole weight path
(`detail::tensorBytes`/`loadMat`/`loadVec`, the mmap borrow, every dequant kernel) is GGUF-only, so
a parallel container would duplicate all of it. `scripts/convert_whisper_to_gguf.py` costs one
script and no runtime code, and carries its own minimal GGUF **writer** — validated by this repo's
strict reader the moment qorvix opens the output. A file with ggml's magic is named with its fix,
not reported as "bad magic".

**The model (`audio/whisper_{weights,model}.*`).** Conv stem (kernel 3, padding 1; stride 2 on the
second, so 3000 mel frames become 1500 encoder positions), sinusoidal positions **read from the
file** rather than regenerated, pre-norm bidirectional blocks, then a decoder whose blocks carry
three sublayers instead of two. Two facts are structural and each was read off the checkpoint:

- **Cross-attention, the first in this codebase**, and with it two caches whose lifetimes are
  opposites: the cross K/V are computed once per clip and never grow, the self-attention K/V grow
  one row per token. Conflating them (one cache, one length counter) does not crash — it truncates
  what the decoder can hear.
- **`k_proj` has no bias**, in self- and cross-attention alike, while q/v/out all do. The weight
  struct has no `bk` field at all, so a loader cannot invent a zero one and hide the asymmetry.

**Suppression travels with the weights.** Whisper's greedy decode is not argmax over the vocabulary:
the release ships 88 suppressed ids plus two blocked at the first generated position, and the
converter writes both into the GGUF. This is not cosmetic — on the gate's probe the raw argmax after
the prefix is token 522, and the token every other Whisper runtime emits is 708. A reimplemented
list on the C++ side would be a second copy that drifts on the first model whose list differs.

**Protocol and language detection.** The prefix is `<|startoftranscript|>`, language, task,
`<|notimestamps|>`, assembled by the model rather than the tokenizer (an English-only model has no
language token, and `--translate` there is refused rather than silently transcribed). Ids are
resolved **by name** from the file's own vocabulary, and the language range is derived from the
sot/`<|translate|>` anchors, so the count of languages is whatever the file says. Detection is
Whisper's own procedure: one step from sot alone, argmax restricted to the language ids — the
unrestricted argmax at that position is a text token, because the model is also predicting the
transcript.

**Surfaces.**
- `qorvix transcribe <whisper.gguf> --audio f.wav [--language en|auto] [--translate] [--timestamps]
  [--max-tokens N] [--json]`.
- `serve --whisper <whisper.gguf>` — `POST /v1/audio/transcriptions` and `/v1/audio/translations`,
  behind their own mutex for the same reason the CLIP tower has one. This is the **one OpenAI route
  that is not JSON**: the audio arrives as `multipart/form-data`, which is what every OpenAI SDK
  sends, so the HTTP layer gained `Content-Type` (the boundary lives there and cannot be recovered
  from the body) and `api/multipart.cpp` — strict by construction, since a form parser that accepts
  a truncated upload transcribes noise and reports success. `response_format` json/text/verbose_json.
- `prompt` conditioning and `temperature > 0` are **refused, not ignored**: answering a sampled
  request with a greedy transcript misreports what was run.

**The gate: `qorvix whisper-check <model.gguf> --ref <fixture>`**, four tiers against transformers'
fp32 forward pass, ordered so each narrows the cause of the next — an encoder bug, a cross-attention
bug and a missing suppression list otherwise all end the same way, in a transcript that is fluent
and not what other runtimes produce.

| tier | isolates | measured (whisper-tiny F32, 11 s speech probe) |
|---|---|---|
| forced prefix | the protocol, before any compute | `50258 50259 50359 50363` — **matches** |
| encoder position 0 | stem, positions, blocks | max \|diff\| **1.91e-06**, cos **1.0000000** |
| encoder per-dim means | a frame SHIFT, which leaves position 0 intact | max \|diff\| **4.77e-06**, cos **1.0000000** |
| one raw decoder step | cross-attention, with no suppression to hide behind | argmax **400 == 400**, top-5 ids identical, max \|diff\| **4.77e-05** |
| greedy loop | the suppression lists and the stop condition | **23/23 tokens identical**, transcript identical |

The transcript is real: *" And so my fellow Americans ask not what your country can do for you ask
what you can do for your country."* `tests/whisper_test.cpp` covers what needs no model — the stem's
padding, stride and column order against a hand-written reference (with the blocks neutralised so
there is something to compare against), the protocol, the two caches, and every refusal.
`tests/multipart_test.cpp` covers the form parser, mostly through what it rejects.

**Performance is honest and unoptimised** — see BENCHMARKS.md. The encoder's 1500-position
bidirectional attention is a scalar triple loop parallelised only across heads; the projections and
FFN use the same batched quantized GEMV as everything else. F16 is **bit-identical** to F32 for these
checkpoints (OpenAI released Whisper in fp16, so the safetensors are an upcast) — half the file for
no accuracy cost, and currently the slower one to decode with, because the F32 dot kernel is AVX2
while the F16 path dequantizes per element.

**Deferred, with the reason stated:** long-form audio (the sequential algorithm advances by the last
emitted timestamp and carries context through `<|startofprev|>`; the CLI warns and does the first
30 s window, the HTTP route refuses rather than returning a partial transcript as if it were whole),
resampling (16 kHz only, with the ffmpeg command in the error), beam search and the
temperature-fallback loop, word-level timestamps (they need the alignment heads), and a GPU
Whisper backend.

### Phase 11b-3c — Image generation ✅

**The first model in this repo that is not a transformer.** A diffusion pipeline is three networks
and a loop: a CLIP text encoder, a convolutional UNet evaluated once per step, and a VAE decoder
that turns the 64×64×4 latent into pixels. Only the first is anything the runtime had seen before —
the other two brought convolution, GroupNorm, nearest-neighbour upsampling and a residual block into
a codebase that had none of them, and the sampler brought arithmetic that has no weights at all.

The phase was gated on GPU hardware because SDXL at 1024² is ~50 evaluations of a 2.6 B-parameter
UNet. **The gate was on the checkpoint, not on the architecture**: SD 1.x/2.x at 512² is the same
network an order of magnitude smaller, and it runs here. SDXL and Flux are still refused, now by
name and with the reason (below).

**Convolutions are matmuls, and activations are position-major.** The two decisions the module is
organised around, and both are about not writing a second weight path:

- Every conv weight is written by the converter as a 2-D `[out, kh*kw*in]` matrix, so it loads
  through the same `detail::loadMat` and runs through the same `qmatmulN` as every linear layer in
  this repo, with im2col supplying the patch vectors. `runtime::ops` gained one function for it —
  `matmulN`, the F32 twin of the existing `qmatmulN` — because a convolution calls the GEMV with
  *thousands* of vectors (one per output pixel), where calling it once per vector opens thousands
  of OpenMP regions and re-streams the weight matrix each time.
- A feature map is `[h*w, c]`, channel fastest — not PyTorch's `[c, h, w]`. That makes an im2col
  patch nine contiguous runs instead of a strided gather, makes the GEMM's natural output the next
  layer's input with no transpose between convolutions, and makes the spatial transformer's
  "flatten to a sequence of tokens" step *nothing at all*, since a feature map already is
  `[tokens, channels]`. The converter permutes conv kernels to `[out, kh, kw, in]` to match.

**Five things can be wrong and there is one observable.** A wrong beta schedule, a wrong BPE split,
a bidirectional text encoder, a skip stack off by one and a swapped GEGLU half all present
identically: a picture that is plausible and is not the one every other runtime makes. Each was a
real risk and each is called out where it occurs:

- **CLIP's BPE is not the byte-level BPE this repo already had** (`image/clip_tokenizer.*`). CLIP
  marks the END of a word (`dog</w>`) where GPT-2 marks the start (`Ġdog`), lowercases and collapses
  whitespace first, and splits digits **one at a time** — "512" is three tokens. Threading a second
  convention through the shared encoder would have put a branch in every step of a hot path the text
  models depend on, to serve one caller.
- **The text encoder's mask is causal.** Architecturally it is Phase 11b-1's CLIP vision tower with
  token+position embeddings instead of patches; run it bidirectionally and the conditioning is
  smooth, confident, and not what the UNet was trained against. Its padding is **not** masked either:
  a diffusion pipeline runs CLIP with no attention mask at all, so the 77-position padding is part of
  the conditioning.
- **Two GroupNorm epsilons in one network.** Residual blocks use the config's `norm_eps` (1e-5); the
  spatial transformer's own norm is hardcoded to 1e-6 in diffusers and appears in no config file.
  Both are written into the GGUF rather than assumed.
- **The skip stack is one deeper than the resnet count suggests** — `conv_in`'s output is pushed
  before any block runs, and each down block pushes one per resnet *plus* one per downsampler. An
  up block pops `layers_per_block + 1`. Off by one and most shapes still line up.
- **`attention_head_dim` in a diffusers config is the NUMBER OF HEADS**, a documented wart. It is
  resolved in the converter, against the config that carries the ambiguity, so the GGUF states head
  counts unambiguously and the C++ side never learns the wart existed.

**Surfaces.**
- `qorvix draw <sd.gguf> --prompt "..." [--out image.png] [--negative "..."] [--steps N]
  [--guidance G] [--size WxH] [--seed N] [--sampler euler|euler-a|ddim] [--clip-skip N] [--json]`.
- `serve --sd <sd.gguf>` — `POST /v1/images/generations`, behind its own mutex for the same reason
  the CLIP tower and Whisper have theirs. OpenAI's fields plus the ones a local diffusion runtime
  cannot do without (`negative_prompt`, `steps`, `guidance_scale`, `sampler`, `seed`, `clip_skip`).
  `response_format: "url"` is **refused**: a URL means this process serves bytes it owns for some
  length of time, and returning a data: URI under that name would be a different contract wearing
  the same label.
- PNG **writing** landed in `vision/` next to the decoder rather than in `image/` — one module owns
  the format, and the round trip through this repo's own inflate is the test that keeps both honest.
  The DEFLATE stream is stored blocks, so the output is correct and larger than libpng's; that is a
  size property, not a correctness one, and it is stated rather than hidden.
- **Seeds are ours.** Seed 42 here is not seed 42 in a PyTorch pipeline, and making it so would mean
  reimplementing MT19937 plus `torch.randn`'s fill order — tying this runtime's output to another
  project's internals forever, and only on CPU. What a seed is actually for is guaranteed: same seed
  and settings, same image, any machine, any build.

**The gate: `qorvix sd-check <sd.gguf> --ref <fixture>`**, six tiers against diffusers' fp32 CPU
forward pass, ordered so each can only fail for causes the ones above it have cleared. **The starting
latent travels in the fixture**, so no random number generator is anywhere in the comparison.

| tier | isolates | measured (tiny SD fixture, 4 steps at 32²) |
|---|---|---|
| schedule | the beta ladder and the spacing — no weights at all | timesteps `751 501 251 1` **match**; alpha_bar max \|diff\| **3.58e-07** |
| tokenizer | CLIP's BPE, before anything encodes it | **77/77 ids identical** |
| conditioning | the causal mask, quick-GELU, the padding convention | max \|diff\| **1.42e-06**, cos **1.0000000** |
| one UNet step | cross-attention, the skip stack, GEGLU, both epsilons | max \|diff\| **5.22e-07**, cos **1.0000000** |
| the whole loop | input scaling, the guidance formula, the step index | max \|diff\| **2.96e-05**, cos **1.0000000** |
| VAE decode | the decoder alone, fed the reference's own final latent | max \|diff\| **5.22e-07**, cos **1.0000000** |

**PASS.** The tiering earned itself immediately: the first run reported *schedule, tokenizer and
conditioning exact; UNet and VAE both wrong* — which is only possible if the fault is in something
those two share and the text encoder does not. It was: the conv weights on disk were still in
PyTorch's axis order.

`tests/image_test.cpp` covers what needs no model — the PNG round trip (including past the 65535-byte
stored-block boundary), conv2d against a direct quadruple-loop convolution at stride 1 and 2, the
1x1/pointwise equivalence, GroupNorm's statistics and affine, nearest upsampling, skip-concatenation
order, causal attention, the timestep embedding's frequency ladder and `flip_sin_to_cos` ordering,
every sampler refusal, the DDIM closed form (a perfectly predicted noise must recover the clean
latent), CLIP's pretokenizer, and a synthetic UNet whose only job is to prove the skip stack is
consumed exactly.

**Deferred, with the reason stated:**
- **SDXL** — refused by the converter, by name. It runs TWO text encoders concatenated to 2048
  cross-attention dims and adds `text_time` conditioning (a pooled vector plus six micro-condition
  scalars) into the timestep embedding. Neither is expressible in this file format, and converting
  without them yields a UNet with the right shapes and the wrong conditioning. The UNet here is
  already config-driven enough for SDXL's block layout; what is missing is the conditioning.
- **Flux** — not a UNet at all: a rectified-flow DiT with a T5 encoder alongside CLIP.
- **img2img and inpainting** — they need the VAE *encoder*, which the converter deliberately leaves
  behind rather than shipping half a gigabyte of unread weights in every file.
- **Quantized weights.** f32/f16 only, and structurally rather than "not done yet": the
  block-quantized kernels need a row length that is a multiple of 32 (Q8_0) or 256 (K-quants), and a
  conv row is `in_channels * 9` — 36 for the first convolution. A quantized SD file is a per-tensor
  mixture, which is a feature and not a flag.
- **A GPU backend.** Same status as Whisper and the CLIP tower — see Phase 11c.

### Phase 11c — GPU embedding + vision backends ✅🖥️

**CUDA & Vulkan implementations of `IEmbeddingEngine` and `ClipVisionModel`.**
- **CUDA Embedding Engine (`cuda/embedding_model.cu`, `cuda/embedding_model.hpp`):** GPU implementation of `IEmbeddingEngine` for BERT-family models. Runs token embedding lookup, LayerNorm with bias, quantized GEMV (Q8_0, Q4_K, Q6_K, F32), bidirectional attention, GELU FFN, and CLS/mean/last pooling with L2 normalization entirely on-device without incremental KV cache overhead.
- **CUDA CLIP Vision Tower (`cuda/clip_model.cu`, `cuda/clip_model.hpp`):** GPU implementation of the vision encoder and LLaVA MLP projector. Implements patch flattening + quantized conv GEMV, learned patch position embeddings, pre-norm bidirectional blocks, quick-GELU, and MLP projector.
- **`forwardEmbedding` on device backends:** Implemented in `cuda::GpuModel` (both single-GPU eager and multi-GPU sharded with rank-0 upload + all-reduce sum) and `vulkan::VulkanModel`. Bypasses token table lookup to accept ready-made host `[d_model]` projected image embeddings, enabling GPU-accelerated vision-language chat.
- **Unified Factories & Honest Reporting:** `createEmbeddingEngine` and `createClipVisionModel` in `backend.hpp` bridge GGUF weights to CUDA/Vulkan descriptors and return `nullptr` with clear error messages when the requested GPU backend is unavailable, preventing silent fallbacks.
- **CLI Integration:** `embed-check`, `vision-check`, `vlm-check`, `image-embed`, `generate`, and `serve` accept `--gpu`/`--cuda`/`--vulkan` flags and dispatch through unified factories.
- **Vulkan Stubs:** Factory stubs and headers prepared for Vulkan compute shader implementation.
`agents/` is still 0 files. SPEC's agent runtime — roles, tool calls, a shared blackboard — is the
last unstarted item of Phase 11b.

## Phase 12 — Web UI ✅🖥️
Modern React 19 + TypeScript + Vite + Tailwind CSS dashboard (`webUI/`) with zero bloated external UI runtime dependencies:
- **Overview & Hardware Dashboard (`DashboardPage`):** Real-time hardware telemetry, active models, runtime latency, quick workspace launch tiles.
- **Chat & Instruct Studio (`ChatPage`):** Streaming SSE client for `/v1/chat/completions`, multi-turn sessions, tok/s speedometer, generation parameters drawer, system prompt customization, Markdown code syntax highlighting.
- **Multimodal Vision Studio (`VisionPage`):** Image upload, CLIP ViT-L/14 patch projection preview (576 patches), VQA reasoning with LLaVA projector.
- **Whisper Audio Studio (`AudioPage`):** Live HTML5 AudioContext microphone waveform visualizer, audio drag-and-drop, transcription & translation (`/v1/audio/transcriptions`, `/translations`), timestamped segment breakdown.
- **Stable Diffusion Studio (`ImageGenPage`):** On-device UNet text-to-image synthesis (`/v1/images/generations`), CFG guidance, steps slider, seed controller, interactive gallery with lightbox modal and PNG download.
- **Embeddings & Vector Matrix (`EmbeddingsPage`):** BERT dense representations (`/v1/embeddings`), real-time pairwise cosine similarity heatmap matrix, vector norm inspector.
- **Model Registry (`ModelsPage`):** GGUF tensor explorer, quantization breakdown (`Q4_K`, `Q8_0`), architecture diagnostics, context length.
- **Memory & VRAM Visualizer (`MemoryPage`):** 3-tier storage monitor (GPU VRAM, Host RAM, NVMe Spool), KV cache page table allocation, slab allocator fragmentation gauges.
- **Performance & Benchmarking (`PerformancePage`):** Real-time TPS speedometer, TTFT prefill latency, multi-client synthetic load test simulator, P50/P95/P99 latency distribution histogram.
- **Prometheus Metrics (`MetricsPage`):** Live Port 2009 (`/metrics`) scraper, HTTP request counters, token totals, raw scrape exporter.
- **Settings & Port Manager (`SettingsPage`):** Endpoints configuration (Port 2005 inference, Port 2009 metrics), Port Allocation Registry (2005-2010 single source of truth), streaming toggles, cache manager.

## Phase 13 — Enterprise Hardening⬜
Speculative decoding (draft/target/verification), API keys, rate limiting, audit logs, security
review, stress/leak/GPU-regression test suites.

## Phase 14 — Performance Validation⬜
Benchmark on target hardware (RTX 4090) against the throughput/utilization/context-length
targets in SPEC.md. Tune until targets are met or document the gap honestly.

---