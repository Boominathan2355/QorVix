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
| **2026-07-30** | **4ad49c8** | **cuda** | **Tesla T4 (Colab, driver 580.82.07)** | **86.65** | **11.54** | **99.38** | **the CUDA baseline.** `load 0.89 s`, median of 3 (min 11.51 / max 11.63 ms/tok), `gpu-check` argmax parity PASS |
| _pending_ | — | vulkan | discrete GPU (real hardware) | — | — | — | Colab T4 run did not complete (see note below); needs a re-run or another device |

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

## Optimization log

Append one row per attempt: what changed, the before→after `decode_tok_per_sec` on the fixed
workload, and whether argmax parity held. A "no measurable change" result is a valid, recorded
outcome — it stops us from repeating it (as the reverted Q4_K vectorized-load attempt did).

| Date | Commit | Backend | Change | decode tok/s before→after | parity | verdict |
|------|--------|---------|--------|---------------------------|--------|---------|
| 2026-07-30 | _this commit_ | cuda | Q4_K GEMV re-blocked: 4 contiguous elements/lane so activations load as `float4` and weights as one `uint32`; header decoded from one shared `uint4` instead of 8 `getScaleMinK4` calls | 86.65 → _pending T4_ | _pending_ | _awaiting measurement_ |

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

### Decision rule for the pending row — agreed 2026-07-30, do not skip

**No further GPU kernel changes until this row has a measured number.** Static evidence (30 → 7
memory instructions) is a hypothesis about the bottleneck, not a result. Run `scripts/colab_bench.sh`
on the T4 first, then branch on what it says:

- **Decode improves over 86.65** → the memory-instruction-issue theory is confirmed. Extend the same
  re-blocking to the Q6_K GEMV (it is the same shape: 75% L1TEX, still on a `__shfl` broadcast for
  `d`, still decoding scales per element).
- **Decode does not move** → **stop this line.** Three attempts will have failed to move a kernel
  whose instruction count provably fell 4×, which would mean MIO issue pressure is not the binding
  constraint and the model of this kernel is wrong. Do not try a fourth variant of the same idea.
  Re-profile from scratch (`scripts/colab_ncu_profile.sh`) and find the next dominant bottleneck
  before writing any kernel code.

This rule exists because the same mistake has already been made twice (4637be5, a341822): a plausible
mechanism was iterated on instead of re-measured.

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
