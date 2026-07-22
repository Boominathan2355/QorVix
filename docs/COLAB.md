# Testing the CUDA backend on a real GPU (Google Colab)

The dev box and CI have no NVIDIA device, so the CUDA path is only *compile*-verified there. To
verify **execution** on real hardware, build and run the GPU self-tests on a Colab GPU runtime.

## ⚠️ Use a GPU runtime, not TPU

CUDA runs only on NVIDIA GPUs. In Colab: **Runtime → Change runtime type → Hardware accelerator =
T4 GPU** (or any GPU). A **TPU** runtime cannot run CUDA (no `nvcc`, no cuBLAS) and the script will
stop with an error.

## ⚠️ Private repo — you need a GitHub token

QorVix is a **private** repository, so the old public `curl … | bash` one-liner (and any anonymous
`git clone`) now returns 404. Create a **fine-grained personal access token** with read-only
**Contents** permission scoped to the `QorVix` repo only
(GitHub → Settings → Developer settings → Fine-grained tokens), then use it below. Do **not** paste
the token into a file that gets committed; keep it in the Colab cell.

## Run it (one cell)

```python
import os
os.environ['GH_TOKEN'] = 'github_pat_...'   # fine-grained, read-only Contents, QorVix only

# Authenticated shallow clone, then run the build+test script (it reuses this checkout).
!git clone --depth 1 "https://x-access-token:${GH_TOKEN}@github.com/Boominathan2355/QorVix.git" /content/qorvix
!bash /content/qorvix/scripts/colab_cuda_test.sh
```

The script: checks for an NVIDIA GPU, installs `ninja`/`gcc-12`, reuses (or authenticated-clones)
the repo, configures with `-DQORVIX_ENABLE_CUDA=ON` (no vcpkg — the CUDA executable needs only
`nvcc` + a C++23 host compiler), builds, and runs `qorvix gpu`. `GH_TOKEN` set in the cell is also
picked up by the script, so a fresh runtime works without the explicit `git clone` line too.

## What it verifies

`qorvix gpu` runs on the actual device and prints:
- CUDA build status and **device enumeration** (name, compute capability, SM count, free/total VRAM),
- a **scale-kernel self-test** — host→device copy, a custom CUDA kernel launch, device→host copy,
  result checked on the host,
- a **cuBLAS SGEMM self-test** — `C = A·B` with `A` an identity, verified on the host,
- a **native Q8_0 matmul self-test** (Phase 8) — a GPU GEMV that dequantizes Q8_0 weights in
  registers, checked against a host reference, then timed on a 4096×4096 matrix (reports GFLOP/s
  and GB/s — compare the GB/s to the CPU runtime's ~3.7 GB/s).

Exit code is nonzero if any self-test fails. Copy the output back here.

## What it does NOT (yet) do

The CUDA backend is not wired into the text forward pass — inference still runs on CPU (Phase 8
moves the validated forward pass onto the GPU). So this confirms the CUDA **backend** (kernels,
cuBLAS, device/memory management) executes correctly; it is not yet an end-to-end GPU generation
benchmark.

## Full unit-test suite on GPU (optional)

`ctest` needs Catch2 v3 (not on Colab by default). The full suite already runs green in the
`Dockerfile.cuda` image; on Colab, `qorvix gpu` is the quick execution check without that
dependency.
