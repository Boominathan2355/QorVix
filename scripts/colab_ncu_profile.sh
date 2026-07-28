#!/usr/bin/env bash
# Profile the Q4_K / Q6_K GEMV decode kernels on a real GPU with Nsight Compute (ncu).
#
# WHY this script exists: `qorvix gpu`'s self-test profiles a tiny 32-row, cache-resident
# correctness matrix (DRAM ~1% — meaningless). And the decode path captures the whole forward
# into ONE CUDA graph node, so ncu can't see individual kernels. This script sets
# QORVIX_NO_GRAPH=1 (eager launches) and profiles the REAL generate workload: full-size,
# DRAM-resident weight GEMVs — the launches that actually bound decode throughput.
#
# One-liner (paste into a Colab cell with a T4 GPU runtime):
#   !curl -fsSL https://raw.githubusercontent.com/Boominathan2355/QorVix/main/scripts/colab_ncu_profile.sh | bash
#
# Copy ALL output back to the chat. The decisive numbers:
#   - DRAM Throughput %      (SpeedOfLight)  — how close to the memory wall we actually are
#   - Warp stall breakdown   (WarpStateStats) — WHY the warps idle:
#       * "Stall Long Scoreboard" dominant  -> global-load latency not hidden -> multi-row-per-warp
#       * "Stall Barrier"/"Stall MIO Throttle" dominant -> the __shfl header broadcast serializes
set -euo pipefail

echo "==================== Qorvix GEMV profiling (ncu) ===================="

# --- 1. Require an NVIDIA GPU + ncu ---------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU. Colab: Runtime > Change runtime type > T4 GPU (NOT TPU), then rerun."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: ncu (Nsight Compute) not found on this runtime; cannot profile."
  exit 1
fi
ncu --version | head -2

# --- 2. Build tooling -----------------------------------------------------------------------
echo "---- installing build tools (ninja, gcc-12) ----"
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y ninja-build g++-12 gcc-12 >/dev/null 2>&1
pip -q install --upgrade cmake >/dev/null 2>&1 || true
export CC=gcc-12 CXX=g++-12

# --- 3. Source + build ----------------------------------------------------------------------
SRC=/content/qorvix
echo "---- cloning + building Qorvix (CUDA on) ----"
rm -rf "${SRC}"
git clone --depth 1 https://github.com/Boominathan2355/QorVix.git "${SRC}"
cd "${SRC}"
git log --oneline -1
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DQORVIX_ENABLE_CUDA=ON -DQORVIX_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_HOST_COMPILER=g++-12 >/dev/null
cmake --build build-cuda -j"$(nproc)"
BIN="${SRC}/build-cuda/core/qorvix"

# --- 4. Model -------------------------------------------------------------------------------
MODEL="${SRC}/models/tinyllama.gguf"
if [ ! -f "${MODEL}" ]; then
  echo "---- downloading TinyLlama 1.1B Q4_K_M (~670 MB) ----"
  mkdir -p "${SRC}/models"
  curl -fsSL -o "${MODEL}" \
    https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
fi

# Correctness + speed on THIS clone, before profiling — so a single `curl | bash` answers all
# three questions (argmax parity, tok/s, and the stall breakdown) on the same commit. This exists
# because running the profiler alone left the fix's on-device argmax and tok/s unmeasured while the
# notebook's gpu-check/generate cells rebuilt an older clone.
echo ""
echo "############## correctness gate: GPU vs CPU argmax ##############"
"${BIN}" gpu-check "${MODEL}" 2>&1 | grep -iE "Argmax|rel err|RESULT"
echo ""
echo "############## end-to-end decode throughput (tok/s) ##############"
"${BIN}" generate "${MODEL}" --gpu --prompt "The capital of France is" --temp 0 --max 40 2>&1 \
  | grep -iE "tok/s|forwards"

# --- whole-forward time breakdown -----------------------------------------------------------
# The single most important table: where the per-token time ACTUALLY goes across all kernels.
# Micro-optimizing one GEMV is pointless if it is a small slice of the forward — this ranks every
# kernel by total GPU time so the real bottleneck is obvious.
echo ""
echo "############## per-kernel time breakdown (whole forward) ##############"
# Prefer nsys (clean summary table). It ships with the CUDA toolkit but is often not on PATH on
# Colab, so search the usual install dirs, and try a quiet apt install as a last resort.
NSYS="$(command -v nsys 2>/dev/null || true)"
[ -z "${NSYS}" ] && NSYS="$(ls /opt/nvidia/nsight-systems/*/target-linux-x64/nsys 2>/dev/null | head -1 || true)"
[ -z "${NSYS}" ] && NSYS="$(ls /usr/local/cuda*/bin/nsys 2>/dev/null | head -1 || true)"
if [ -z "${NSYS}" ]; then
  apt-get -qq install -y nsight-systems-cli >/dev/null 2>&1 || true
  NSYS="$(command -v nsys 2>/dev/null || true)"
fi

if [ -n "${NSYS}" ]; then
  echo "(using nsys: ${NSYS})"
  "${NSYS}" profile -o /tmp/qorvix_prof --force-overwrite true --stats=true \
      env QORVIX_NO_GRAPH=1 "${BIN}" generate "${MODEL}" --gpu --prompt "hi" --temp 0 --max 8 \
      2>/dev/null | sed -n '/CUDA GPU Kernel Summary/,/CUDA GPU MemOps Summary/p'
else
  # Fallback that always works: ncu times one metric across a token's worth of launches and
  # aggregates total GPU time per kernel. ncu replays each launch, so bound the count.
  echo "(nsys unavailable — ncu per-kernel duration summary instead)"
  ncu --target-processes all --launch-skip 22 --launch-count 200 \
      --metrics gpu__time_duration.sum --csv \
      env QORVIX_NO_GRAPH=1 "${BIN}" generate "${MODEL}" --gpu --prompt "hi" --temp 0 --max 4 \
      2>/dev/null \
    | awk -F'","' 'NR>1 { name=$5; val=$NF; gsub(/"/,"",val); sum[name]+=val; n[name]++ }
        END { for (k in sum) printf "%12.1f us total  x%-4d  %s\n", sum[k]/1000.0, n[k], k }' \
    | sort -rn \
    || echo "(ncu duration summary failed)"
fi

# The generate workload with graphs OFF so ncu sees individual kernel launches.
GEN=(env QORVIX_NO_GRAPH=1 "${BIN}" generate "${MODEL}" --gpu --prompt "hi" --temp 0 --max 2)
SECTIONS=(--section SpeedOfLight --section Occupancy --section WarpStateStats \
          --section MemoryWorkloadAnalysis)
# Skip past prompt/warm-up launches to land on a steady, full-size, DRAM-resident weight GEMV.
SKIP=40

echo ""
echo "############## Q4_K GEMV (qmatmulQ4_KKernel) — the decode bottleneck ##############"
ncu --target-processes all --launch-skip "${SKIP}" --launch-count 1 \
    --kernel-name "regex:qmatmulQ4_KKernel" "${SECTIONS[@]}" "${GEN[@]}" 2>&1 \
  | grep -iE "Throughput|Occupancy|Duration|Registers Per Thread|Waves Per|Block Limit|Stall|Elapsed|Bytes|Achieved" \
  || echo "(no Q4_K launch captured — try lowering SKIP)"

echo ""
echo "############## Q6_K GEMV (qmatmulQ6_KKernel) — the faster-per-byte comparison ##############"
ncu --target-processes all --launch-skip "${SKIP}" --launch-count 1 \
    --kernel-name "regex:qmatmulQ6_KKernel" "${SECTIONS[@]}" "${GEN[@]}" 2>&1 \
  | grep -iE "Throughput|Occupancy|Duration|Registers Per Thread|Waves Per|Block Limit|Stall|Elapsed|Bytes|Achieved" \
  || echo "(no Q6_K launch captured — try lowering SKIP)"

echo ""
echo "==================== done — copy ALL of the above back to chat ===================="
