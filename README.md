# Qorvix AI Core

A local, single-process, multi-GPU-capable AI inference runtime written in C++23/CUDA — text,
vision, audio, image generation, embeddings, RAG, and multi-agent workflows from one executable.

- Mission and full requirements: [docs/SPEC.md](docs/SPEC.md)
- Build plan and current status: [ROADMAP.md](ROADMAP.md)

**Status:** Phase 2 complete — runtime skeleton (plugin framework, model discovery/watcher, CLI)
plus a full GGUF v1/v2/v3 parser (`qorvix gguf-info`). No inference yet.
