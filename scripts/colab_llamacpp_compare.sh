#!/usr/bin/env bash
# Qorvix vs llama.cpp on the SAME GPU, SAME model, SAME workload.
#
# BENCHMARKS.md is the single source of truth for whether a change helped, but it has no external
# reference point — it cannot say whether 86.65 decode tok/s on a T4 is respectable or terrible.
# This script answers that by building llama.cpp with CUDA next to Qorvix and running both on one
# device, one TinyLlama 1.1B Q4_K_M, one workload (prompt 64 / gen 128, 3 runs).
#
# One-liner (Colab cell, T4 GPU runtime):
#   !curl -fsSL https://raw.githubusercontent.com/Boominathan2355/QorVix/main/scripts/colab_llamacpp_compare.sh | bash
#
# Deliberately SEPARATE from scripts/colab_bench.sh: that script gates the Phase 8c decision and
# must stay fast and untouched. This one is a reference measurement, run on its own.
set -euo pipefail

PROMPT="${QORVIX_BENCH_PROMPT:-64}"
GEN="${QORVIX_BENCH_GEN:-128}"
RUNS="${QORVIX_BENCH_RUNS:-3}"
BUILD_JOBS="$(nproc)"

echo "==================== Qorvix vs llama.cpp — same GPU, same model, same workload ===================="
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: no NVIDIA GPU. This comparison is meaningless without one (Colab: Runtime > T4 GPU)."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "---- build tools ----"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y ninja-build g++-12 gcc-12 libcurl4-openssl-dev >/dev/null 2>&1 || true
pip -q install --upgrade cmake >/dev/null 2>&1 || true

# TinyLlama 1.1B Chat Q4_K_M — the fixed workload model from BENCHMARKS.md.
MODEL=/content/models/tinyllama.gguf
if [ ! -f "${MODEL}" ]; then
  echo "---- downloading TinyLlama 1.1B Q4_K_M (~670 MB) ----"
  mkdir -p /content/models
  curl -fsSL -o "${MODEL}" \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
fi

# ---------------------------------------------------------------------------
# Qorvix
# ---------------------------------------------------------------------------
echo ""
echo "############## building Qorvix (CUDA) ##############"
rm -rf /content/qorvix
git clone --depth 1 https://github.com/Boominathan2355/QorVix.git /content/qorvix
git -C /content/qorvix log --oneline -1
CC=gcc-12 CXX=g++-12 cmake -S /content/qorvix -B /content/qorvix/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DQORVIX_ENABLE_CUDA=ON -DQORVIX_ENABLE_VULKAN=OFF \
  -DQORVIX_BUILD_TESTS=OFF -DCMAKE_CUDA_HOST_COMPILER=g++-12 >/dev/null
cmake --build /content/qorvix/build -j"${BUILD_JOBS}" >/dev/null 2>&1 || \
  cmake --build /content/qorvix/build -j"${BUILD_JOBS}"
QORVIX=/content/qorvix/build/core/qorvix

# Correctness gate. A speed comparison against a backend that computes the wrong answer is noise.
echo ""
echo "############## Qorvix correctness gate (argmax parity vs its own CPU path) ##############"
"${QORVIX}" gpu-check "${MODEL}" 2>&1 | grep -iE "Argmax|rel err|RESULT" || echo "(gpu-check unavailable)"

# ---------------------------------------------------------------------------
# llama.cpp
# ---------------------------------------------------------------------------
echo ""
echo "############## building llama.cpp (CUDA, sm_75) ##############"
rm -rf /content/llama.cpp
git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
git -C /content/llama.cpp log --oneline -1
cmake -S /content/llama.cpp -B /content/llama.cpp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >/dev/null
cmake --build /content/llama.cpp/build -j"${BUILD_JOBS}" --target llama-bench >/dev/null 2>&1 || \
  cmake --build /content/llama.cpp/build -j"${BUILD_JOBS}" --target llama-bench
LBENCH=/content/llama.cpp/build/bin/llama-bench

# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------
echo ""
echo "############## RESULTS — prompt=${PROMPT} gen=${GEN} runs=${RUNS} ##############"
echo ""
echo "---- Qorvix (qorvix bench --cuda) ----"
"${QORVIX}" bench "${MODEL}" --cuda --prompt "${PROMPT}" --gen "${GEN}" \
  --warmup 1 --runs "${RUNS}" --json || echo "(qorvix bench FAILED)"

echo ""
echo "---- llama.cpp (llama-bench, all layers offloaded) ----"
# -p = prompt/prefill tokens (pp), -n = generated tokens (tg), -ngl 99 = every layer on the GPU.
"${LBENCH}" -m "${MODEL}" -p "${PROMPT}" -n "${GEN}" -r "${RUNS}" -ngl 99 -o md || \
  echo "(llama-bench FAILED)"

echo ""
echo "==================== how to read this ===================="
cat <<'NOTE'
Map the rows onto BENCHMARKS.md like this:

  llama.cpp "pp64"  <-> qorvix prefill_tok_per_sec
  llama.cpp "tg128" <-> qorvix decode_tok_per_sec   <- the number Phase 8c is optimizing

Both run every layer on the GPU with the same file, so the gap is backend quality, not setup.

Caveat worth keeping in mind when reading the decode row: llama.cpp's CUDA GEMV quantizes the
ACTIVATION vector to q8_1 (int8 + per-32 scale) and uses __dp4a integer dot products, while Qorvix
keeps activations in f32. That is a real numerical difference, not just an optimization — part of
llama.cpp's speed is bought with activation precision. Compare tok/s and output quality together.

Paste the whole output back; it becomes the external reference row in BENCHMARKS.md.
NOTE
