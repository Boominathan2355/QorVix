# Docker builds

The Qorvix reference target is Linux (see [../ROADMAP.md](../ROADMAP.md)). These images
give you the exact Linux toolchain — GCC 14, CMake + Ninja, and a bootstrapped
[vcpkg](https://github.com/microsoft/vcpkg) — without needing any of it installed on the
host. That's the supported way to build Boost.Beast and run the Catch2 tests from a
Windows machine.

Everything is driven by the `linux-release` preset in
[`../CMakePresets.json`](../CMakePresets.json); the images don't invent their own build
flags.

## Files

| File | Purpose |
| --- | --- |
| `Dockerfile` | Multi-stage CPU build (Ubuntu 24.04). Builds + tests, then produces a slim runtime image. |
| `Dockerfile.cuda` | Same, from an NVIDIA CUDA base with `-DQORVIX_ENABLE_CUDA=ON`. Optional, Phase 4+. |
| `.dockerignore` | Keeps `build/`, `.git/`, models, `node_modules`, etc. out of the build context. |
| `docker-compose.yml` | `build` / `test` / `dev` convenience services for the CPU image. |
| `.devcontainer/devcontainer.json` | VS Code Dev Container using the CPU toolchain. |

## CPU image (`Dockerfile`)

### Build

From the repo root:

```sh
docker build -f Dockerfile -t qorvix:cpu .
```

The build runs four stages:

1. **toolchain** — Ubuntu 24.04 + GCC 14 (24.04's default is GCC 13, so `gcc-14`/`g++-14`
   are installed and made the default via `update-alternatives`, with `CC`/`CXX` set) +
   CMake/Ninja + a vcpkg clone bootstrapped into `/opt/vcpkg` (`VCPKG_ROOT`).
2. **deps** — copies only `vcpkg.json` and runs `vcpkg install`, warming the vcpkg binary
   cache. This is the slow part (Boost, sqlite3, …) and it lives *below* the source copy,
   so editing C++ doesn't rebuild dependencies.
3. **builder** — copies the repo and runs
   `cmake --preset linux-release && cmake --build --preset linux-release && ctest --preset linux-release`.
   A failing test fails the image build.
4. **runtime** — a fresh Ubuntu 24.04 with only `libstdc++6`, carrying the `qorvix` binary
   (on `PATH`) and the example plugin `.so`, running as an unprivileged user.

> Dependency versions are pinned by `builtin-baseline` in `vcpkg.json`, not by the vcpkg
> clone's HEAD — so a plain `git clone` of vcpkg stays reproducible.

### Run

```sh
docker run --rm qorvix:cpu                 # prints help (default CMD)
docker run --rm qorvix:cpu version
docker run --rm qorvix:cpu plugins         # loads the bundled example plugin from ./plugins
docker run --rm -v "$PWD/models:/opt/qorvix/models" qorvix:cpu scan-models
```

`ENTRYPOINT` is `qorvix`, so anything after the image name is passed straight to the CLI.
The working directory is `/opt/qorvix`; `plugins` defaults to `./plugins` (where the
example `.so` is installed) and `scan-models`/`list` default to `./models`.

### Run the tests

Tests already run during the image build. To (re)run them explicitly, target the builder
stage — it contains the configured + built tree:

```sh
docker build -f Dockerfile --target builder -t qorvix:builder .
docker run --rm qorvix:builder ctest --preset linux-release
```

## Compose

```sh
docker compose build build          # build the slim runtime image (tests run during build)
docker compose run --rm build version
docker compose run --rm test        # re-run the Catch2 suite via ctest
docker compose run --rm dev         # shell into the toolchain with the source mounted
```

The `dev` service mounts the host source at `/src` over a toolchain image that already has
a warmed vcpkg cache, and persists that cache in a named volume. Inside it:

```sh
cmake --preset linux-release
cmake --build --preset linux-release
ctest --preset linux-release
./build/linux-release/core/qorvix version
```

An anonymous volume masks `/src/build` so in-container builds never collide with the
host's `build/` directory.

## Dev Container (VS Code)

Open the repo in VS Code with the *Dev Containers* extension and "Reopen in Container".
`.devcontainer/devcontainer.json` builds the `deps` stage (toolchain + warmed vcpkg
cache), sets `VCPKG_ROOT`, installs the C/C++ and CMake Tools extensions, and runs
`cmake --preset linux-release` as `postCreateCommand`. After that, build/test from the
integrated terminal exactly as in the `dev` service above.

## CUDA image (`Dockerfile.cuda`) — optional, Phase 4+

Built from `nvidia/cuda:12.6.0-devel-ubuntu24.04` (a current CUDA 12.x devel tag) and
configured with `-DQORVIX_ENABLE_CUDA=ON`. The runtime stage uses the matching
`nvidia/cuda:12.6.0-runtime-ubuntu24.04`.

### Status today

There is **no CUDA source in the tree yet** (`cuda/` isn't added in `CMakeLists.txt` — it's
a Phase 4 item). Turning the option on only makes CMake detect `nvcc` and enable the CUDA
language; with no `.cu` files, nothing CUDA is compiled. If the toolkit's host-compiler
check rejects the compiler, top-level `CMakeLists.txt` **degrades to CPU-only with a
warning** instead of failing. Either way the image builds. (CUDA 12.6 officially supports
GCC ≤ 13 as the host compiler, so `CUDAHOSTCXX=g++-13` is set while the C++23 code itself
uses GCC 14.)

### Prerequisites (run time)

- An NVIDIA GPU host with a recent driver.
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed and configured on the host.
- Pass `--gpus all` when running (harmless today; required once real kernels land).

> Building this image does **not** need a GPU. Running it with GPU access does.

### Build and run

```sh
docker build -f Dockerfile.cuda -t qorvix:cuda .
docker run --rm --gpus all qorvix:cuda version
```

## Notes / troubleshooting

- **First build is slow.** vcpkg compiles Boost from source. Subsequent builds reuse the
  cached `deps` layer unless `vcpkg.json` changes.
- **Docker daemon.** These commands need the Docker daemon running (Docker Desktop on
  Windows/macOS). `docker info` should succeed before you build.
- **Triplet.** vcpkg uses the default `x64-linux` triplet, which links dependencies
  statically — that's why the runtime image needs only `libstdc++6`.
