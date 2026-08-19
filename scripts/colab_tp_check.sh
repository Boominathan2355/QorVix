#!/usr/bin/env bash
# Phase 10b gate: does the TENSOR-PARALLEL forward pass agree with the unsharded GPU forward on a
# real model, on real hardware?
#
# One-liner (Colab cell, T4 GPU runtime):
#   !curl -fsSL https://raw.githubusercontent.com/Boominathan2355/QorVix/main/scripts/colab_tp_check.sh | bash
#
# WHY THIS RUNS ON ONE GPU. A rank in Qorvix is a (device, weight-shard) pair, and nothing stops two
# ranks from naming the SAME device — they then reduce through the host-staged collective instead of
# NCCL, and every other line of the sharded model is byte-identical. So a single T4 executes the
# real TP=2 and TP=4 code paths: same shards, same per-rank KV caches, same two all-reduces per
# layer, same column-parallel LM head. What a second GPU would add is bandwidth, not coverage.
#
# It buys no speed here (the ranks time-share one device and the host-staged reduce costs a PCIe
# round trip), and the script says so rather than printing a number that looks like a regression.
#
# What would still be untested after this passes: NCCL's own transport, and P2P/NVLink behaviour.
# Those need >= 2 GPUs, which is why `qorvix gpu` reports the NCCL build state separately.
set -euo pipefail

echo "==================== Qorvix tensor-parallel check ===================="
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU detected."
  echo "In Colab: Runtime > Change runtime type > Hardware accelerator = T4 GPU (NOT TPU)."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "visible GPUs: ${GPUS}"

echo "---- build tools ----"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y ninja-build g++-12 gcc-12 >/dev/null 2>&1 || true
pip -q install --upgrade cmake >/dev/null 2>&1 || true
export CC=gcc-12 CXX=g++-12

SRC=/content/qorvix
echo "---- clone + build (CUDA on; NCCL on if the runtime ships it) ----"
rm -rf "${SRC}"
git clone --depth 1 https://github.com/Boominathan2355/QorVix.git "${SRC}"
cd "${SRC}"
git log --oneline -1

# NCCL is optional: without it the collective seam still has a working transport, so the build is
# only asked for it when the library is actually present.
NCCL_FLAG="-DQORVIX_ENABLE_NCCL=OFF"
if [ -f /usr/include/nccl.h ] || [ -f /usr/local/cuda/include/nccl.h ]; then
  NCCL_FLAG="-DQORVIX_ENABLE_NCCL=ON"
fi
echo "nccl: ${NCCL_FLAG}"

cmake -S . -B build-tp -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DQORVIX_ENABLE_CUDA=ON "${NCCL_FLAG}" -DQORVIX_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_HOST_COMPILER=g++-12 >/dev/null
cmake --build build-tp -j"$(nproc)"
BIN="${SRC}/build-tp/core/qorvix"

MODEL="${SRC}/models/tinyllama.gguf"
if [ ! -f "${MODEL}" ]; then
  echo "---- downloading TinyLlama 1.1B Q4_K_M (~670 MB) ----"
  mkdir -p "${SRC}/models"
  curl -fsSL -o "${MODEL}" \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
fi

echo ""
echo "############## device self-tests (includes the collective transport) ##############"
"${BIN}" gpu || true

# Baseline first: if the UNSHARDED GPU path disagrees with the CPU runtime, a tensor-parallel
# mismatch below would say nothing about sharding.
echo ""
echo "############## baseline: unsharded GPU vs CPU ##############"
"${BIN}" gpu-check "${MODEL}" 2>&1 | grep -iE "Argmax|rel err|RESULT" || true

RC=0
# TinyLlama has 4 KV heads, so 4 is its ceiling: a rank must own whole KV heads or it would refetch
# half a head every step. TP=8 is expected to be REJECTED, and that rejection is part of the gate.
for W in 2 4; do
  echo ""
  echo "############## tensor-parallel: ${W} ranks ##############"
  "${BIN}" tp-check "${MODEL}" --tp "${W}" || RC=$?
done

echo ""
echo "############## refusal: 8 ranks (above the 4 KV-head cap) ##############"
if "${BIN}" tp-check "${MODEL}" --tp 8 >/dev/null 2>&1; then
  echo "UNEXPECTED: TP=8 was accepted on a 4-KV-head model."
  RC=1
else
  echo "OK: TP=8 refused, as the GQA cap requires."
fi

echo ""
echo "=================================================================="
if [ "${RC}" -eq 0 ]; then
  echo "RESULT: tensor-parallel forward matches the unsharded GPU model at TP=2 and TP=4."
  echo "NOTE: throughput is NOT measured here - ranks share one device on purpose."
else
  echo "RESULT: FAILED (exit ${RC}) - see the mismatch above."
fi
exit "${RC}"
