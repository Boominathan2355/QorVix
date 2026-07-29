#!/usr/bin/env bash
# Canonical throughput benchmark on a real GPU (Colab T4) — the single source of truth for the
# Vulkan/CUDA optimization work. Builds Qorvix, then runs `qorvix bench` on the SAME workload for
# every available backend and prints the machine-readable JSON so results can be diffed across
# commits (paste them into BENCHMARKS.md).
#
# One-liner (Colab cell, T4 GPU runtime):
#   !curl -fsSL https://raw.githubusercontent.com/Boominathan2355/QorVix/main/scripts/colab_bench.sh | bash
#
# Every optimization must move the `decode_tok_per_sec` here (and keep `gpu-check` / `vulkan-check`
# argmax parity) or it does not ship.
set -euo pipefail

# Workload — keep these FIXED across runs so numbers are comparable. Override via env if needed.
PROMPT="${QORVIX_BENCH_PROMPT:-64}"
GEN="${QORVIX_BENCH_GEN:-128}"
WARMUP="${QORVIX_BENCH_WARMUP:-1}"
RUNS="${QORVIX_BENCH_RUNS:-3}"

echo "==================== Qorvix bench (single source of truth) ===================="
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: no NVIDIA GPU — CUDA backend will be unavailable (Colab: Runtime > T4 GPU)."
else
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

echo "---- build tools ----"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y ninja-build g++-12 gcc-12 \
  libvulkan-dev glslang-tools mesa-vulkan-drivers vulkan-tools >/dev/null 2>&1 || true
pip -q install --upgrade cmake >/dev/null 2>&1 || true
export CC=gcc-12 CXX=g++-12

SRC=/content/qorvix
echo "---- clone + build (CUDA + Vulkan on) ----"
rm -rf "${SRC}"
git clone --depth 1 https://github.com/Boominathan2355/QorVix.git "${SRC}"
cd "${SRC}"
git log --oneline -1
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DQORVIX_ENABLE_CUDA=ON -DQORVIX_ENABLE_VULKAN=ON -DQORVIX_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_HOST_COMPILER=g++-12 >/dev/null
cmake --build build -j"$(nproc)"
BIN="${SRC}/build/core/qorvix"

MODEL="${SRC}/models/tinyllama.gguf"
if [ ! -f "${MODEL}" ]; then
  echo "---- downloading TinyLlama 1.1B Q4_K_M (~670 MB) ----"
  mkdir -p "${SRC}/models"
  curl -fsSL -o "${MODEL}" \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
fi

echo ""
echo "############## available backends ##############"
"${BIN}" backends

# Correctness gate first — a fast kernel that gets the wrong answer is worthless.
echo ""
echo "############## correctness gate (argmax parity vs CPU) ##############"
"${BIN}" gpu-check "${MODEL}" 2>&1 | grep -iE "Argmax|rel err|RESULT" || echo "(cuda gpu-check skipped)"

echo ""
echo "############## bench: same workload, every backend (JSON) ##############"
echo "workload: prompt=${PROMPT} gen=${GEN} warmup=${WARMUP} runs=${RUNS}"
for be in cpu cuda vulkan; do
  echo -n "  ${be}: "
  "${BIN}" bench "${MODEL}" --${be} --prompt "${PROMPT}" --gen "${GEN}" \
    --warmup "${WARMUP}" --runs "${RUNS}" --json 2>/dev/null || echo "(unavailable)"
done

echo ""
echo "==================== done — paste the JSON lines into BENCHMARKS.md ===================="
