# QORVIX  AI CORE — Project Specification

> Preserved verbatim as provided on 2026-07-21. This is the source-of-truth mission
> document for the project. See ../ROADMAP.md for how this gets broken into phases.

## MISSION

Build a next-generation local AI runtime and inference platform written entirely in modern C++ and CUDA.

The goal is to create a single unified executable that surpasses llama.cpp in:

- Throughput
- GPU Utilization
- Memory Efficiency
- Multimodal Capability
- Multi-Model Support
- Long Context Handling
- Concurrent User Handling

while supporting:

- Text Generation
- Reasoning Models
- Vision Models
- Image Understanding
- OCR
- Speech To Text
- Text To Speech
- Image Generation
- Embeddings
- RAG
- Multi-Agent Workflows

Everything must run from ONE server process.

No microservices. No external inference servers. No llama.cpp. No Ollama. No vLLM. No ExLlama. No TGI.
Build runtime from scratch.

## TECH STACK

**Core Runtime**
- C++23
- CUDA 12+
- CUTLASS
- cuBLAS
- NCCL
- TensorRT (optional)
- CMake

**Networking**
- CrowCPP or Boost Beast
- WebSocket
- SSE Streaming

**Frontend**
- React
- TypeScript
- Vite
- TailwindCSS
- shadcn/ui
- Recharts

**Storage**
- SQLite
- PostgreSQL (optional)

**Monitoring**
- Prometheus
- Grafana Exporter

## PRIMARY OBJECTIVES

1. Outperform llama.cpp
2. Use less overall memory for multimodal workloads
3. Load user models automatically
4. Support GGUF
5. Support future model formats
6. Support multi GPU
7. Support enterprise deployment
8. Provide modern UI

## PERFORMANCE TARGETS (RTX 4090)

| Model | Target |
|---|---|
| 7B Q4 | 250-450 tok/s |
| 32B | 75-200 tok/s |
| 70B | 30-100 tok/s |

- GPU Utilization: 90-98%
- Memory Utilization: 95% efficient
- Long Context: 128K+ / 256K+ / 512K+ (future goal: 1M)

## SYSTEM ARCHITECTURE

```
Client
  |
Web UI
  |
API Gateway
  |
Global Scheduler
  |
Unified Memory Manager
  |
Runtime Engine
  |
CUDA Backend
  |
Hardware
```

## PROJECT STRUCTURE

```
qorvix/
  core/
  runtime/
  cuda/
  memory/
  scheduler/
  gguf/
  tokenizer/
  api/
  plugins/
  models/
  vision/
  audio/
  image/
  embeddings/
  rag/
  agents/
  ui/
  monitoring/
  cli/
  tests/
  docs/
```

## GGUF SUPPORT

Implement a full parser supporting:

- Header Parser
- Metadata Parser
- Tensor Table Parser
- Tokenizer Metadata
- Rope Metadata
- Quantization Metadata
- Tensor Offsets
- Architecture Detection

Supported quantizations: F32, F16, BF16, Q4_0, Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, with future quantization plugin support.

## MODEL DISCOVERY

```
models/
  llama3.gguf
  qwen3.gguf
  deepseek.gguf
  gemma.gguf
  whisper.gguf
  flux.gguf
```

Automatically detect. No restart required. Watcher monitors model directory. Auto register.

## PLUGIN SYSTEM

```
plugins/
  llama/
  qwen/
  gemma/
  phi/
  mistral/
  deepseek/
  vision/
  audio/
  diffusion/
```

Interface:

```cpp
class IPlugin
{
public:
    virtual bool load() = 0;
    virtual bool unload() = 0;
    virtual bool infer() = 0;
    virtual std::string architecture() = 0;
};
```

Auto discovery. Hot load. Hot unload.

## UNIFIED MEMORY MANAGER

Advanced memory architecture:

- Tier 1: GPU VRAM
- Tier 2: System RAM
- Tier 3: Memory Mapped NVMe

Components: MemoryManager, TensorRegistry, MemoryPool, TensorPool, KVCacheManager, DiskCacheManager

Features: Global Allocation, Reference Counting, Shared Buffers, Memory Pooling, Defragmentation, Offloading,
Prefetching, Predictive Loading, Page Migration, Smart Eviction

## MEMORY PAGE SYSTEM

Use memory pages: 4MB / 8MB / 16MB / 32MB / 64MB. Avoid direct tensor allocations — everything allocated via
page manager.

## UNIFIED KV CACHE

Do not use per-model cache. Create `GlobalKVCache`.

Features: Shared Pool, Paged KV, Session Restore, Context Reuse, KV Compression, Context Pruning,
Automatic Cleanup, Cross Session Optimization

## CUDA OPTIMIZATION (Mandatory)

FlashAttention 3, CUDA Graphs, CUTLASS Kernels, Tensor Core Support, Fused Operations, Persistent Kernels,
Kernel Fusion, Asynchronous Compute, Overlapped Transfers, Pinned Memory, GPUDirect Ready

## NATIVE QUANTIZATION KERNELS

Implement direct kernels — do not dequantize to FP16 first.

Support: Q4_K, Q5_K, Q6_K, Q8_0 — Direct GPU MatMul, Quantized Attention, Quantized FFN, Quantized KV Cache

## TEXT RUNTIME

RMSNorm, LayerNorm, RoPE, GQA, MQA, Multi Head Attention, SwiGLU, FFN, Residual Connections

Sampling: Greedy, Top K, Top P, Min P, Temperature, Frequency Penalty, Presence Penalty, Repetition Penalty,
Streaming Generation

## SPECULATIVE DECODING

Draft Model, Target Model, Verification Pipeline, Automatic Draft Selection, Multi Draft Support

## SCHEDULER

Dynamic Batching, Continuous Batching, Request Prioritization, Fair Scheduling, Queue Management,
Session Routing, GPU Aware Scheduling, Memory Aware Scheduling, Model Aware Scheduling

## MULTI GPU SUPPORT

Tensor Parallelism, Pipeline Parallelism, Expert Parallelism, Load Balancing, Automatic Distribution

Support: 1 / 2 / 4 / 8 GPU

## VISION ENGINE

Support: Qwen VL, Llama Vision, MiniCPM-V, InternVL

Pipeline: Image → Vision Encoder → Projected Embeddings → Language Decoder → Answer

Features: Image Understanding, OCR, Multi Image Chat, Grounding, Object Detection Hooks

## AUDIO ENGINE

Speech To Text, Voice Detection, Streaming Transcription, Speaker Separation, Text To Speech,
Voice Cloning Plugin Architecture, Audio Embeddings

## IMAGE GENERATION ENGINE

Support: Flux, SDXL, Stable Diffusion, future diffusion architectures

Workflows: txt2img, img2img, inpaint, outpaint, upscale, control image

## EMBEDDINGS ENGINE

Text Embeddings, Vision Embeddings, Audio Embeddings, Cross Modal Embeddings

## RAG SYSTEM

Built in: Document Loader (PDF, Word, CSV, TXT, Markdown), Chunking, Embedding, Hybrid Search,
Vector Store (SQLite Vector, Postgres Vector)

## OPENAI COMPATIBLE API

```
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings
POST /v1/audio/transcriptions
POST /v1/audio/speech
POST /v1/images/generations
POST /v1/responses
```

WebSocket Streaming, SSE Streaming

## CLI

```
qorvix list
qorvix load llama3
qorvix unload llama3
qorvix benchmark llama3
qorvix monitor
qorvix stats
qorvix scan-models
```

## WEB UI

Modern AI studio with pages: Dashboard, Chat, Vision, Audio, Image Generation, Model, Memory, Performance,
Settings. Theme: professional dark mode / glassmorphism, Open WebUI, Linear, Vercel.
Stack: TailwindCSS, shadcn/ui, Recharts, Framer Motion.

## SECURITY

API Keys, Rate Limiting, Audit Logs.

## TESTING

Unit Tests, Integration Tests, Performance Tests, Stress Tests, Memory Leak Tests, GPU Regression Tests

## FINAL GOAL

Deliver a complete C++ and CUDA multimodal AI operating system capable of loading user supplied GGUF models,
supporting text, vision, speech, image generation, embeddings and agents inside a single runtime while
maximizing GPU utilization, minimizing memory overhead, implementing a unified memory manager, FlashAttention 3,
CUDA Graphs, speculative decoding, continuous batching, native quantized kernels and advanced scheduling to
outperform llama.cpp in throughput, concurrency and multimodal workloads.
