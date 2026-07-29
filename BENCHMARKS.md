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
| _pending_ | — | cuda | Tesla T4 (Colab) | — | — | — | prior campaign saw ~87–89 tok/s (generate); re-measure with `bench` |
| _pending_ | — | vulkan | discrete GPU (real hardware) | — | — | — | needs an AMD/NVIDIA/Intel/Apple device |

## Optimization log

Append one row per attempt: what changed, the before→after `decode_tok_per_sec` on the fixed
workload, and whether argmax parity held. A "no measurable change" result is a valid, recorded
outcome — it stops us from repeating it (as the reverted Q4_K vectorized-load attempt did).

| Date | Commit | Backend | Change | decode tok/s before→after | parity | verdict |
|------|--------|---------|--------|---------------------------|--------|---------|
| _(first entry lands with the first hardware-measured optimization)_ | | | | | | |
