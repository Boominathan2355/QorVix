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

## Optimization log

Append one row per attempt: what changed, the before→after `decode_tok_per_sec` on the fixed
workload, and whether argmax parity held. A "no measurable change" result is a valid, recorded
outcome — it stops us from repeating it (as the reverted Q4_K vectorized-load attempt did).

| Date | Commit | Backend | Change | decode tok/s before→after | parity | verdict |
|------|--------|---------|--------|---------------------------|--------|---------|
| _(first entry lands with the first hardware-measured optimization)_ | | | | | | |
