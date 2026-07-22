#!/usr/bin/env bash
# Build Qorvix with the real CUDA backend and run its GPU self-tests on an actual NVIDIA GPU.
# Intended for Google Colab with a *GPU* runtime (Runtime > Change runtime type > T4 GPU).
#
# One-liner (paste into a Colab cell):
#   !curl -fsSL https://raw.githubusercontent.com/Boominathan2355/QorVix/main/scripts/colab_cuda_test.sh | bash
#
# NOTE: CUDA requires an NVIDIA GPU. A TPU runtime will NOT work (no nvcc / no CUDA device).
# The CUDA backend is not yet wired into the text forward pass (that's Phase 8); this verifies
# device enumeration + a custom kernel launch (H2D->kernel->D2H) + cuBLAS GEMM on real hardware.
set -euo pipefail

echo "==================== Qorvix CUDA GPU test ===================="

# --- 1. Require an NVIDIA GPU ---------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU detected."
  echo "In Colab: Runtime > Change runtime type > Hardware accelerator = T4 GPU (NOT TPU),"
  echo "then re-run this cell. CUDA cannot run on a TPU."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. This runtime has a GPU but no CUDA toolkit."
  exit 1
fi
nvcc --version | grep -i release || true

# --- 2. Build tooling -----------------------------------------------------------------------
echo "---- installing build tools (ninja, gcc-12) ----"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y ninja-build g++-12 gcc-12 >/dev/null 2>&1
pip -q install --upgrade cmake >/dev/null 2>&1 || true   # ensure a recent CMake
export CC=gcc-12 CXX=g++-12
echo "cmake: $(cmake --version | head -1)"

# --- 3. Get the source ----------------------------------------------------------------------
SRC=/content/qorvix
echo "---- cloning Qorvix into ${SRC} ----"
rm -rf "${SRC}"
git clone --depth 1 https://github.com/Boominathan2355/QorVix.git "${SRC}"
cd "${SRC}"

# --- 4. Configure + build (CUDA on; no vcpkg / no tests needed for the GPU self-test) --------
# The core executable and its libs don't use any vcpkg package; Boost/Catch2 are only for the
# HTTP API and unit tests. So a CUDA build needs only nvcc + a C++23 host compiler.
echo "---- configuring (QORVIX_ENABLE_CUDA=ON) ----"
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DQORVIX_ENABLE_CUDA=ON \
  -DQORVIX_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_HOST_COMPILER=g++-12

echo "---- building ----"
cmake --build build-cuda -j"$(nproc)"

# --- 5. Run the GPU self-tests --------------------------------------------------------------
echo ""
echo "==================== qorvix gpu (real device) ===================="
./build-cuda/core/qorvix gpu
RC=$?
echo "=================================================================="
if [ "${RC}" -eq 0 ]; then
  echo "RESULT: GPU self-tests PASSED — CUDA backend executes on this device."
else
  echo "RESULT: GPU self-tests FAILED (exit ${RC}) — see messages above."
fi
exit "${RC}"
