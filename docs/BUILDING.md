# Building Qorvix

## Prerequisites

- CMake 3.28+
- Ninja
- A C++23 compiler (GCC 14+, Clang 17+, or MSVC 19.38+)
- [vcpkg](https://github.com/microsoft/vcpkg), for anything beyond the dependency-free `core`
  module (Boost.Beast, spdlog, CLI11, sqlite3, Catch2)
- CUDA 12+ toolkit, only once you build with `-DQORVIX_ENABLE_CUDA=ON` (Phase 4+)

Linux is the reference/CI target (see [../ROADMAP.md](../ROADMAP.md) for why). Windows builds
are kept working where practical but aren't the CI gate.

## First-time vcpkg setup

```sh
git clone https://github.com/microsoft/vcpkg.git
./vcpkg/bootstrap-vcpkg.sh      # or bootstrap-vcpkg.bat on Windows
export VCPKG_ROOT=$(pwd)/vcpkg  # set this in your shell profile
```

`vcpkg.json` pins `builtin-baseline` to a specific vcpkg commit so everyone (and CI) resolves
the same dependency versions. Bump it deliberately with:

```sh
$VCPKG_ROOT/vcpkg x-update-baseline
```

## Configuring and building

With `VCPKG_ROOT` set, use the matching preset for your platform:

```sh
cmake --preset linux-debug          # or linux-release / windows-debug
cmake --build --preset linux-debug
ctest --preset linux-debug
```

### Quick local iteration without vcpkg

The `quick` preset builds only targets with zero third-party dependencies (currently: `core`,
the `qorvix` executable, and it silently skips tests since they need Catch2). Useful for
verifying the toolchain itself or iterating on dependency-free modules:

```sh
cmake --preset quick
cmake --build --preset quick
./build/quick/core/qorvix
```

## CUDA builds

CUDA is off by default so the project stays buildable on machines without a GPU toolchain
(including CI runners). Once Phase 4 lands actual CUDA sources:

```sh
cmake --preset linux-release -DQORVIX_ENABLE_CUDA=ON
```

If no CUDA compiler is found, configuration falls back to CPU-only with a warning rather than
failing outright.

## Docker

If you don't want to install the Linux toolchain and vcpkg locally (e.g. on Windows), a
reproducible Docker build environment is provided. It ships GCC 14 + CMake/Ninja + a
bootstrapped vcpkg and builds/tests with the `linux-release` preset:

```sh
docker build -f Dockerfile -t qorvix:cpu .   # CPU build (runs the tests during the build)
docker run --rm qorvix:cpu version
```

There's also a CUDA variant (`Dockerfile.cuda`, Phase 4+), a `docker-compose.yml`, and a
VS Code Dev Container. See [DOCKER.md](DOCKER.md) for the full workflow.
