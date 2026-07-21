# Qorvix AI Core

A local, single-process, multi-GPU-capable AI inference runtime written in C++23/CUDA — text,
vision, audio, image generation, embeddings, RAG, and multi-agent workflows from one executable.

- Mission and full requirements: [docs/SPEC.md](docs/SPEC.md)
- Build plan and current status: [ROADMAP.md](ROADMAP.md)

**Status:** Phase 3 complete — runtime skeleton (plugin framework, model discovery/watcher, CLI),
full GGUF v1/v2/v3 parser (`qorvix gguf-info`), and the unified memory manager (paged tiered
allocation, refcounted tensor registry, LRU offload eviction). No inference yet.
