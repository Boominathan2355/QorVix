# syntax=docker/dockerfile:1
#
# Qorvix AI Core — reproducible Linux CPU build (multi-stage).
#
# Linux is the project's reference/CI target (see ROADMAP.md). This image gives the
# real Linux toolchain + vcpkg that a Windows dev box lacks, so Boost.Beast and the
# Catch2 test suite build and run.
#
# Stages:
#   toolchain  — Ubuntu 24.04 + GCC 14 + CMake/Ninja + a bootstrapped vcpkg. No source.
#   deps       — warms the vcpkg binary cache from vcpkg.json only, so source-only
#                changes below don't force Boost/etc. to rebuild.
#   builder    — copies the repo, then configures/builds/tests with the linux-release preset.
#   runtime    — slim image carrying only the qorvix binary + example plugin .so.
#
# Build:  docker build -f Dockerfile -t qorvix:cpu .
# Run:    docker run --rm qorvix:cpu version
# See docs/DOCKER.md for the full workflow (including running the tests in-container).

# ---------------------------------------------------------------------------
# Stage 1: toolchain — compilers, build tools, and vcpkg (no project source)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS toolchain

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 24.04 ships GCC 13 as the default `gcc`, but the project is C++23 and wants
# GCC 14+, so install gcc-14/g++-14 explicitly. Everything else here is what vcpkg
# needs to fetch and build the manifest dependencies (Boost via b2, sqlite3, etc.):
#   git                  — vcpkg registry + builtin-baseline resolution
#   curl/zip/unzip/tar   — vcpkg bootstrap + port archive handling
#   cmake/ninja-build    — the project build system (also used by CMake-based ports)
#   pkg-config           — dependency discovery for several ports
#   ca-certificates      — HTTPS clones/downloads
#   linux-libc-dev       — kernel/libc headers needed to build Boost and friends
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc-14 g++-14 \
      cmake \
      ninja-build \
      git \
      curl \
      zip \
      unzip \
      tar \
      pkg-config \
      ca-certificates \
      linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

# Make GCC 14 the default toolchain and point CC/CXX at it so both CMake and vcpkg
# (which detect the compiler from the environment) use it consistently.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100
ENV CC=gcc-14 \
    CXX=g++-14

# Bootstrap vcpkg into /opt/vcpkg. The pinned `builtin-baseline` in vcpkg.json — not the
# vcpkg clone's HEAD — is what fixes dependency versions, so a plain clone is reproducible.
ENV VCPKG_ROOT=/opt/vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git "${VCPKG_ROOT}" \
    && "${VCPKG_ROOT}/bootstrap-vcpkg.sh" -disableMetrics
ENV PATH="${VCPKG_ROOT}:${PATH}"

# ---------------------------------------------------------------------------
# Stage 2: deps — warm the vcpkg binary cache from the manifest only
# ---------------------------------------------------------------------------
# By copying just vcpkg.json and installing here, the (slow) Boost/etc. build lands in
# an earlier layer than the source. Editing C++ files then reuses this cached layer and
# vcpkg restores the prebuilt packages from its binary cache (~/.cache/vcpkg) instantly.
FROM toolchain AS deps

WORKDIR /src
COPY vcpkg.json ./
RUN vcpkg install --triplet x64-linux \
      --x-install-root=/tmp/vcpkg_installed \
      --clean-after-build

# ---------------------------------------------------------------------------
# Stage 3: builder — configure, build, and test the whole project
# ---------------------------------------------------------------------------
FROM deps AS builder

WORKDIR /src
# Bring in the rest of the repo (see .dockerignore for what's excluded).
COPY . .

# Configure + build + test with the linux-release preset from CMakePresets.json. The
# preset's vcpkg toolchain re-runs the manifest install, but every package is already in
# the binary cache warmed above, so this is fast. `ctest` fails the build on a red test.
RUN cmake --preset linux-release \
    && cmake --build --preset linux-release \
    && ctest --preset linux-release

# ---------------------------------------------------------------------------
# Stage 4: runtime — slim image with just the built artifacts
# ---------------------------------------------------------------------------
# The x64-linux triplet links dependencies statically, so the only runtime deps of the
# qorvix binary are libc/libstdc++/libgcc. Ubuntu 24.04 already ships the GCC 14 runtime
# (libstdc++6 14.x), which matches the g++-14 the binary was built with.
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      libstdc++6 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 qorvix

WORKDIR /opt/qorvix

# The main executable (onto PATH) and the example plugin. `qorvix plugins` defaults to a
# ./plugins directory, and `qorvix scan-models` to ./models, both relative to this WORKDIR.
COPY --from=builder /src/build/linux-release/core/qorvix /usr/local/bin/qorvix
COPY --from=builder /src/build/linux-release/plugins/example/qorvix_plugin_example.so \
     /opt/qorvix/plugins/qorvix_plugin_example.so

USER qorvix

# Qorvix reserves 2005-2010; this image only serves the Runtime. Metadata only — publish with
# `docker run -p 2005:2005 ... serve <model.gguf>`.
EXPOSE 2005

ENTRYPOINT ["qorvix"]
CMD ["help"]
