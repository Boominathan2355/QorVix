# Qorvix Benchmarks — the single source of truth

Every performance claim, optimization, and architectural change is validated here before moving on.
Measurement beats new features: a change that doesn't move `decode_tok_per_sec` on this fixed
workload (while keeping `gpu-check` / `vulkan-check` argmax parity) does not ship.

## How to measure

One tool, one code path (`runBenchmark` in `runtime/benchmark.hpp`), every backend:

```
qorvix bench <model.gguf> [--cpu|--gpu|--vulkan|--auto] \
    --prompt 64 --gen 128 --warmup 1 --runs 3 [--json]
```

Warmup runs are discarded; timed runs are reduced to a **median**, so the number is stable across
runs and comparable across commits. `--json` is the machine-readable form to paste below.

- **Real GPU numbers (CUDA T4, discrete-GPU Vulkan):** run `scripts/colab_bench.sh` on a Colab T4.
  That is the canonical hardware harness — it builds CUDA+Vulkan, gates argmax parity, then benches
  cpu/cuda/vulkan on one fixed workload.
- **Correctness anywhere (no GPU):** the Vulkan backend verifies on Mesa lavapipe in Docker
  (`vulkan-check`), and `qorvix bench --vulkan` runs there too — but lavapipe is CPU software
  rendering, so its throughput is a **correctness** figure, not a performance one. Do NOT tune GPU
  kernels against lavapipe numbers.
- **Regression guard:** `tests/benchmark_test.cpp` runs the same `runBenchmark` on a synthetic CPU
  model in the unit suite, with a loose catastrophic-regression floor (fails only on
  order-of-magnitude slowdowns, never on CI noise).

## Fixed workload

`TinyLlama 1.1B Chat Q4_K_M`, `prompt=64 gen=128 warmup=1 runs=3`. Keep this fixed; change it only
deliberately and re-baseline every backend at once.

## Baselines

Reference-environment numbers only. GPU rows must be filled from a real device
(`scripts/colab_bench.sh`) — the dev/CI box has none, so they are marked pending rather than guessed.

| Date | Commit | Backend | Device | decode tok/s | ms/tok | prefill tok/s | Notes |
|------|--------|---------|--------|-------------:|-------:|--------------:|-------|
| 2026-07-29 | 6dea5a0 | cpu | Docker Ubuntu (container CPU, Release) | ~1.2 | ~836 | ~1.2 | small workload (prompt 8 gen 8); reference sanity only, not a target |
| 2026-07-29 | 6dea5a0 | vulkan | Mesa lavapipe (CPU software Vulkan) | ~0.14 | ~7040 | ~0.15 | correctness path; **not** a perf figure — do not optimize against this |
| 2026-07-30 | 4ad49c8 | cuda | Tesla T4 (Colab, driver 580.82.07) | 86.65 | 11.54 | 99.38 | the first CUDA baseline; superseded by the row below |
| **2026-07-30** | **cfd1fa9** | **cuda** | **Tesla T4 (Colab)** | **114.82** | **8.71** | **140.15** | **current.** `load 0.75 s`, median of 3 (min 8.70 / max 8.71 ms/tok), `gpu-check` PASS (max abs err 7.15e-06, rel err 4.61e-07) |
| 2026-07-30 | cfd1fa9 | cpu | Colab host CPU (light ref) | 1.24 | 805 | 1.41 | prompt 8 / gen 16; sanity only, not a target |
| _pending_ | — | vulkan | discrete GPU (real hardware) | — | — | — | **TIMEOUT after 900 s** on the T4 at cfd1fa9 — that build predates the device-local fix (d8cb8c5), which is still unmeasured |

Raw JSON for the CUDA row (`scripts/colab_bench.sh`, notebook cell 22):

```json
{"backend":"cuda","prompt_tokens":64,"gen_tokens":128,"timed_runs":3,"load_sec":0.889656,
 "prefill_tok_per_sec":99.3793,"decode_tok_per_sec":86.6515,
 "decode_ms_per_tok_median":11.5405,"decode_ms_per_tok_min":11.5126,"decode_ms_per_tok_max":11.6349}
```

Two things this baseline says out loud:

- **Decode is nowhere near the T4's memory roof.** 11.54 ms/tok over ~0.6 GB of Q4_K weights is
  ~52 GB/s of achieved DRAM bandwidth against the T4's ~320 GB/s. `ncu` agrees from the other side:
  the Q4_K GEMV runs at 28% DRAM throughput but 84% L1TEX throughput with 46% of warp stalls on a
  full MIO instruction queue. The kernel is **memory-instruction-issue bound, not bandwidth bound** —
  the lever is fewer, wider loads, not fewer bytes.
- **Prefill (99 tok/s) is barely faster than decode (87 tok/s)**, which is the signature of a
  token-at-a-time prefill running the same GEMV path. A batched `forwardBatch` (GEMV → GEMM) is a
  large separate win, tracked in ROADMAP Phase 8; it does not move `decode_tok_per_sec`.

The `vulkan` row on the T4 came back empty — the run was cut off after the CUDA line with no error.
`scripts/colab_bench.sh` now wraps each backend in a `timeout` and surfaces stderr so a slow or
hanging backend reports itself instead of silently truncating the run.

### Why the T4 Vulkan row never finished — root cause

Not a hang. Every Vulkan buffer was allocated `HOST_VISIBLE | HOST_COHERENT` with no
`DEVICE_LOCAL` (`makeBuffer`, documented at the time as "correctness-first: no staging, map/unmap
directly"). On a **discrete** GPU that memory is system RAM reached across PCIe, so the ~500 MB
weight set was streamed over the bus *once per token*. At a realistic 4–8 GB/s for GPU reads of
host-coherent memory that is ~60–120 ms/token before any compute — roughly 10× slower than the CUDA
path's 11.5 ms/tok, over 768 forwards (192 tokens × 4 runs). It was crawling, not stuck.

**The dev box structurally cannot catch this class of bug.** Mesa lavapipe reports exactly one
memory type:

```
memoryTypes: count = 1
  memoryTypes[0]: MEMORY_PROPERTY_DEVICE_LOCAL_BIT | HOST_VISIBLE_BIT | HOST_COHERENT_BIT | HOST_CACHED_BIT
```

so on lavapipe asking for host-visible *also* returns device-local — there is nothing else to
return. A discrete NVIDIA device exposes ~11 memory types and `findMemType` returns the **first**
match, which is host RAM. The lavapipe loop validates numerics perfectly and is blind to memory
placement; treat every future "verified on lavapipe" claim as scoped to correctness only.

Fixed by `makeDeviceBuffer` (requests `DEVICE_LOCAL`, falls back to host-visible only if the device
has no device-local heap) plus `uploadDevice`, a one-shot staging copy used at load time. Weights,
the embedding table, norms and scratch are now device-local; `blogits_` stays host-visible because
the host maps it every token (~128 KB against ~500 MB — irrelevant). The staging path runs
*unconditionally*, even where the destination is also host-visible, so lavapipe exercises the same
code a discrete GPU will take rather than skipping it.

Verified on lavapipe: 8/8 Vulkan self-tests PASS, and `vulkan-check` on the real TinyLlama gives
**max abs err 6.35e-06, rel err 4.09e-07, argmax agrees at every position**. The speedup itself is
**unmeasured** — it needs a discrete GPU, which is the whole point of the bug.

Still open (measured next, not now): the forward re-records its command buffer and re-allocates a
descriptor set for every one of ~377 dispatches per token (22 layers × 17 ops + 3), with a full
compute→compute barrier after each. That is CPU-side submission overhead the T4 run will expose once
the PCIe streaming is gone.

## Encoder axis — `embed_tok_per_sec` (Phase 11a)

Embeddings are a second seam, so they get a second measurement core (`runEmbedBenchmark` in
`embeddings/embed_benchmark.hpp`) — `runBenchmark` takes an `IInferenceEngine&` and its result is
decode-shaped. Still **one CLI and one file**: `qorvix bench` dispatches on the model family.

```
qorvix bench <encoder.gguf> --seq 256 --batch 1 --warmup 1 --runs 3 [--json]
```

**The decision rule, restated for this axis:** a change that does not move `embed_tok_per_sec` on
the fixed encoder workload — while `embed-check` still reports the same tokenizer parity and
vector cosine — does not ship. The correctness gate here is `embed-check`, not argmax parity;
it is a CLI gate rather than a CTest case because the Docker test image has no GGUF.

**Fixed encoder workload:** `bge-small-en-v1.5 F16`, `seq=256 batch=1 warmup=1 runs=3`.

| Date | Commit | Backend | Device | embed tok/s | ms/seq | Notes |
|------|--------|---------|--------|------------:|-------:|-------|
| 2026-08-13 | 6b43f25 | cpu | i7 (this box, AVX2) | 27.72 | 9236.24 | baseline after the F16 dequant-batching fix |
| 2026-08-13 | 97e9c0c | cpu | i7 (this box, AVX2) | **225.65** | 1134.50 | batched GEMV (`qmatmulN`) |
| — | — | cuda | 🖥️ pending | — | — | no GPU encoder backend yet (Phase 11c) |

End-to-end effect on the RAG pipeline, indexing the repo's own `docs/` (4 documents → 37 chunks,
5795 tokens): **562.3 s → 20.4 s (27.5×)**.

## Vision axis — CLIP tower (Phase 11b-1)

The vision tower has no throughput axis worth gating on yet: it is a fixed 577-token forward pass
with no tuning applied, and `qorvix bench` does not dispatch to it. Recorded here so the starting
point is written down rather than rediscovered.

| Date | Commit | Backend | Device | ms / image | Notes |
|------|--------|---------|--------|-----------:|-------|
| 2026-08-13 | 8605279 | cpu | i7 (this box, AVX2) | 55,000 – 66,000 | ViT-L/14-336, 23 layers, 577 tokens, F16 |

Slow because it is 577 tokens through a 300M-parameter tower on a CPU, with the same
`qmatmulN` path the text encoder uses and no vision-specific work done. The correctness gate
(`vision-check`) is what matters at this stage; a throughput axis follows when there is something
to compare against.

The first figure written here was ~176,000 ms, measured minutes after a reference capture and with
builds still running. Re-measured on a quiet machine it is 55–66 s — a 3x error, in a row that
would have become the baseline everything after it was compared against. Caught only by applying
the caution below to my own number.

**A measurement caution, recorded because it nearly produced a wrong entry above.** While tuning
`kVecTile` this box reported bge-small at 60–123 tok/s against a recorded 225.65, which looked like
a regression. Building the previous commit in a clean worktree and re-measuring gave 212–222 —
the drop was contention from a build running concurrently, not code. A follow-up "3.75× worse"
conclusion about a wider tile evaporated the same way (cleanly re-measured: 202 vs 191, i.e. noise).

The rule this implies: **on a developer box, never accept a benchmark delta measured while
anything else is running, and confirm a suspected regression by building the prior commit rather
than by reasoning about the diff.** The GPU numbers in this file come from a dedicated Colab
runtime and do not have this problem; the CPU ones do.

## Audio axis — Whisper (Phase 11b-3b)

Like the vision tower, Whisper has no tuned throughput axis yet and `qorvix bench` does not dispatch
to it. Recorded so the starting point is written down rather than rediscovered. Two numbers matter
and they scale differently: the **encoder** is a fixed cost per 30-second window no matter how much
speech is in it, the **decoder** is per token.

Workload: `qorvix transcribe models/whisper-tiny-<type>.gguf --audio tests/data/speech_probe.wav
--language en` — 11 s of speech padded to Whisper's 30 s window, 23 generated tokens. Five runs,
**median**, with the full spread given because this box is noisy (4 hardware threads).

| Date | Commit | Backend | Device | encode s / 30 s window | decode tok/s | Notes |
|------|--------|---------|--------|-----------------------:|-------------:|-------|
| 2026-08-19 | (11b-3b) | cpu | i7 (this box, AVX2, 4 threads) | **9.6** (8.2–11.1) | **36.7** (10.5–53.3) | whisper-tiny F32, 39M params |
| 2026-08-19 | (11b-3b) | cpu | i7 (this box, AVX2, 4 threads) | **9.0** (7.1–12.8) | **13.8** (8.4–17.2) | whisper-tiny F16, same numbers bit-for-bit |

**The spread is the honest part of this table.** 23 tokens is a small sample and four hardware
threads leave no headroom, so single runs differ by 2–5x on the decode axis. What survives the noise
is the direction: every F16 run decoded slower than the F32 median, and no F16 run reached it.

### Why F16 decodes slower than F32, when the numbers are identical

The two files are **bit-identical in value** — all 67 matmul tensors compare exactly after the
round trip, because OpenAI released Whisper in fp16 and the HF safetensors are an upcast of that.
So F16 halves the file for no accuracy cost, and `whisper-check` prints the same figures for both.
It is still the slower one to decode with, and the reason is in `qmatmul`:

- Every quantized weight is dequantized into a 256-element scratch buffer and fed to `vecDotF32`.
  For F32 that batch is a copy; for F16 it is `fp16ToFloat` **per element**, scalar — no F16C
  vector conversion is used.
- A decode step is dominated by the LM head: `51865 x 384` = **19.9M elements per token**. At 4
  threads that is ~23 ms/token via the copy path and ~61 ms/token via the scalar conversion, which
  is the entire measured difference.

So the first optimization on this axis is not the encoder: it is an F16C (`_mm256_cvtph_ps`)
conversion in the dequant batch, or a fused F16 dot kernel that never materializes the floats. That
would help every F16 model in the repo, not just Whisper.

### Where the encoder time goes (unmeasured, stated as a hypothesis)

~37 GFLOP per window (4 layers of 1500-position attention, projections and FFN, plus the conv stem)
in ~9.6 s is ~4 GFLOP/s — well under what 4 AVX2 threads can do. The projections and FFN go through
the same batched quantized GEMV as every other model here; the 1500x1500 bidirectional attention is
a **scalar triple loop parallelised only across the 6 heads**, which is the obvious suspect and the
obvious next thing to measure. It is recorded as a hypothesis rather than a breakdown because no
per-stage timing was taken — the rule in the vision section applies here too.

## Optimization log

Append one row per attempt: what changed, the before→after `decode_tok_per_sec` on the fixed
workload, and whether argmax parity held. A "no measurable change" result is a valid, recorded
outcome — it stops us from repeating it (as the reverted Q4_K vectorized-load attempt did).

| Date | Commit | Backend | Change | decode tok/s before→after | parity | verdict |
|------|--------|---------|--------|---------------------------|--------|---------|
| 2026-07-30 | 5404523 | cuda | Q4_K GEMV re-blocked: 4 contiguous elements/lane so activations load as `float4` and weights as one `uint32`; header decoded from one shared `uint4` instead of 8 `getScaleMinK4` calls | **86.65 → 114.82 (+32.5%)** | **PASS** | **SHIPPED** |
| 2026-08-13 | 6b43f25 | cpu | `qmatmul` filled its 256-float scratch with one BLOCK per call. F16/BF16 have `blockSize == 1`, so a 384-column row meant 384 `dequantize()` dispatches and 384 dot calls of length **one** — the SIMD kernel never engaged. Now fills with as many whole blocks as it holds. | *(encoder axis)* **5.51 s → 0.73 s** on a 4-token embed | byte-identical | **SHIPPED** |
| 2026-08-13 | 97e9c0c | cpu | `qmatmulN`: dequantize each weight block once and fold it into all N dot products, instead of re-streaming the matrix once per token. Decode has an N=1 case that hides this cost; an encoder never does. | *(encoder axis)* **27.72 → 225.65 tok/s (8.1×)** | bit-identical | **SHIPPED** |

The two CPU rows above are on the encoder axis and do **not** move `decode_tok_per_sec` — decode is
N=1, so `qmatmulN` is a no-op there by construction. That is exactly why the encoder needed its own
axis before the change could be justified at all: measured against the decode workload it would have
scored zero and been correctly rejected. Wiring `qmatmulN` into the decoder's **prefill** (where N is
the prompt length) is the open follow-up — it is the same kernel as the "batched prefill" item in
the ROADMAP priorities, and the T4 prefill/decode gap (99 vs 87 tok/s) is its signature.

**Static evidence for the pending row** (what the dev box can prove without a GPU). SASS for `sm_75`,
per super-block iteration of `qmatmulQ4_KKernel`:

| | old (4ad49c8) | new |
|---|---:|---:|
| global load instructions | 13 (`8× LDG.E` 32-bit + `5× LDG.E.U8`) | 5 (`3× LDG.E.128` + `2× LDG.E`) |
| shared load/store instructions | 17 (`16× LDS.U.U8` + `1× STS.U8`) | 2 (`1× LDS.U.128` + `1× STS.128`) |
| **memory instructions total** | **30** | **7** |
| SASS instructions in kernel | 200 | 144 |
| registers / spills | 57 / none | 43 / none |

The two reverted predecessors only ever touched the weight loads. The `8× LDG.E` were the scalar
**activation** loads and the `16× LDS.U.U8` were `getScaleMinK4` being re-evaluated from shared eight
times per super-block — together 24 of the 30 memory instructions, and neither was addressed before.
That is the argument for why this attempt should behave differently from those two; only the T4 can
settle it.

### Decision rule — RESOLVED 2026-07-30

The rule was: no further GPU kernel changes until the row had a measured number, then branch on
*decode improves over 86.65* → extend the re-blocking to Q6_K, or *decode does not move* → abandon
the line and re-profile from scratch.

**Measured: 114.82 tok/s, +32.5%, argmax parity held.** The memory-instruction-issue model of this
kernel is confirmed, and the branch taken is **extend the same re-blocking to the Q6_K GEMV** — same
shape, and per the last `ncu` run it is still at 75% L1TEX with a `__shfl` broadcast for `d` and a
per-element scale decode.

Worth keeping: the static evidence predicted the direction but **not** the size. Memory instructions
fell 4.3× (30 → 7) and decode rose 1.33×, because the kernel was never purely issue-bound — Amdahl
across the rest of the forward pass, and DRAM starts to matter as issue pressure drops. Instruction
counts are a hypothesis generator, not a speedup estimate. The freeze is what made that distinction
visible, and it is why the same rule should apply to the Q6_K attempt.

## External reference: llama.cpp

Every number above is self-relative — it says whether Qorvix got faster than Qorvix, never whether
86.65 tok/s is respectable. `scripts/colab_llamacpp_compare.sh` builds llama.cpp with CUDA next to
Qorvix and runs both on one GPU, one TinyLlama 1.1B Q4_K_M, one workload. It is deliberately separate
from `colab_bench.sh` so it cannot perturb the Phase 8c decision run.

| Date | Device | Qorvix decode | llama.cpp tg128 | Qorvix prefill | llama.cpp pp64 |
|------|--------|--------------:|----------------:|---------------:|---------------:|
| _pending_ | Tesla T4 | 86.65 | — | 99.38 | — |

### Design gap, read from llama.cpp's source (not from memory)

`ggml/src/ggml-cuda/{mmvq.cu,vecdotq.cuh,quantize.cu}`, the MMVQ path that serves batch-1 decode:

| | Qorvix `qmatmulQ4_KKernel` (5404523) | llama.cpp `vec_dot_q4_K_q8_1` |
|---|---|---|
| activation format | **f32, 4 B/element** | **q8_1: int8 + fp16 (d, s) per 32 → 1.125 B/element** |
| activation load | 2× `LDG.E.128` (`float4`) | 32-bit loads of packed int8 |
| weight nibble load | 2× `LDG.E` (`uint32`) | 32-bit `const int* q4` — **same** |
| scale/min unpack | folded register ALU off one shared `uint4` | 16-bit masked `scales[j] & 0x3f3f` — **equivalent** |
| arithmetic | FP32 FFMA, **1 MAC/instruction** | `__dp4a`, **4 int8 MACs/instruction** |

The commit we just landed closed the two right-hand rows — weight loading and scale unpacking are now
essentially what llama.cpp does. The two rows that remain open are the structural ones, and they are
exactly the two that a memory-instruction-issue-bound kernel cares about:

- **~3.6× more activation bytes** (4 B vs 1.125 B per element).
- **~4× more arithmetic instructions** for identical MACs (`FFMA` vs `__dp4a`).

The activation quantization cost is amortized to nothing: `quantize_row_q8_1_cuda` runs once per token
over `cols` elements, and the result is reused across all `rows` of the GEMV.

**This is not a free win, and it is not just an optimization.** q8_1 activations are an approximation;
part of llama.cpp's speed is bought with activation precision. Qorvix's ship gate is argmax parity
against its own f32 CPU reference, and quantizing activations on the GPU only would widen the GPU↔CPU
gap by construction (llama.cpp does not have this problem because its CPU backend quantizes too).
Adopting it means deciding whether the CPU reference quantizes as well, or whether the parity gate
loosens. That decision needs the measured comparison above first — see the decision rule.
