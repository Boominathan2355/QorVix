# Qorvix AI Core

A local, single-process, multi-GPU-capable AI inference runtime written in C++23/CUDA — text,
vision, audio, image generation, embeddings, RAG, and multi-agent workflows from one executable.

- Mission and full requirements: [docs/SPEC.md](docs/SPEC.md)
- Build plan and current status: [ROADMAP.md](ROADMAP.md)

**Status:** Phase 4 code-complete — runtime skeleton (plugin framework, model discovery/watcher,
CLI), full GGUF v1/v2/v3 parser (`qorvix gguf-info`), unified memory manager (paged tiered
allocation, refcounted registry, LRU offload eviction), and a CUDA backend (`qorvix gpu`: device
management, GpuVram memory tier, scale-kernel + cuBLAS GEMM self-tests). The CUDA path builds as a
CPU stub by default; the nvcc build is being compile-checked in Docker and is not yet run on a
GPU.

Phase 5 (text runtime) is in progress, built CPU-first: transformer CPU ops
(RMSNorm/RoPE/attention/SwiGLU/…), GGUF dequantization (Q4_0…Q6_K → F32), model-config extraction
(`qorvix model-info`), the weight loader, and the Llama-style forward pass (GQA + KV cache + greedy
decode) are done and verified on synthetic models. Still to come: BPE tokenizer, full sampling, the
streaming generation loop, and validation against a real GGUF model. No end-to-end text generation
from real weights yet.
