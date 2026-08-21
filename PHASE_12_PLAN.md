# Phase 12 — Modern Web UI & Dashboard

## Overview
A cutting-edge, ultra-responsive, zero-external-bloat Web Dashboard for QorVix built with **React 19 + TypeScript + Vite + Tailwind CSS**.

The Web UI serves as the unified graphical interface to all QorVix inference capabilities (Port 2005 `qorvix serve`, Port 2006 Gateway, Port 2007 Dashboard, and Port 2009 Metrics):
- **Chat & Instruct:** OpenAI-compatible streaming chat (`/v1/chat/completions`), SSE parser, real-time speedometers (tok/s, TTFT), parameter controls, system prompt manager.
- **Multimodal Vision:** High-resolution image upload, patch token preview, visual question answering, OCR & visual analysis via CLIP + LLaVA projector.
- **Audio & Speech (Whisper):** Real-time microphone audio recording with HTML5 AudioContext waveform visualizer, file drag-and-drop, transcription & translation (`/v1/audio/transcriptions`, `/translations`), timestamps, JSON export.
- **Image Generation (Stable Diffusion):** Text-to-image synthesis (`/v1/images/generations`), guidance scale, steps, seed, aspect ratio picker, interactive image gallery with lightbox, metadata inspection, PNG download.
- **Embeddings & Vector Tools:** BERT embedding generation (`/v1/embeddings`), real-time pairwise cosine similarity matrix, semantic search sandbox.
- **Models & Registry:** Model inventory (`/v1/models`), architecture inspection, parameter counts, quantization breakdown, tensor shape diagnostics.
- **Memory & VRAM Visualizer:** 3-tier memory visualizer (GPU VRAM, Host RAM, NVMe Spool), KV cache utilization gauge, slab allocator fragmentation monitor.
- **Performance & Real-Time Metrics:** Live throughput (TPS) graph, latency distributions, interactive multi-client load benchmark simulator, Prometheus `/metrics` scraper.
- **Settings & API Configuration:** Custom backend URL, streaming mode toggle, system presets, dark/light theme, keyboard shortcuts.

---

## Architectural Principles

1. **Modern, Handcrafted UI (Zero Bloated Component Frameworks)**:
   - Beautiful, custom-engineered components with modern glassmorphism, subtle borders (`border-white/10`), smooth transition animations, and dark-mode-first aesthetic (Zinc/Slate base with Cyan/Emerald/Indigo accents).
   - Bespoke vector icon system (`Icons.tsx`) with zero external icon package overhead.
   - Built-in Markdown renderer with code highlighting and copy-to-clipboard buttons.

2. **Direct Streaming Integration**:
   - Fetch API + `ReadableStream` reader for OpenAI-compatible Server-Sent Events (`data: {...}`).
   - Live token-by-token rendering with auto-scroll and instant stop generation (`AbortController`).
   - Real-time token throughput calculation based on arrival timestamps.

3. **Standalone or Served**:
   - Can run as a standalone Vite development server on port 2007.
   - Can be built into a single static bundle (`dist/`) that can be embedded into the C++ runtime or served via Nginx/Docker.

---

## Page Breakdown & Features

| Page | Route | Description |
|------|-------|-------------|
| **Dashboard** | `/` | System overview, active models, hardware status, quick-action tiles, latency meters. |
| **Chat & Vision** | `/chat` | Multi-turn conversational UI, model selector, image attachment, token speed, system prompt drawer. |
| **Audio Studio** | `/audio` | Whisper transcription & translation, live mic waveform, drag-and-drop WAV/MP3, segment view. |
| **Image Generation**| `/images` | SD text-to-image generator, step/scale sliders, seed locking, gallery with lightbox & download. |
| **Embeddings** | `/embeddings` | Text embedding calculator, vector similarity comparison, interactive similarity heatmap. |
| **Model Registry** | `/models` | GGUF model explorer, architecture inspector, quantization details, memory footprint. |
| **Memory Manager** | `/memory` | GPU VRAM vs Host RAM tier monitor, slab allocator visualizer, KV cache page allocation. |
| **Performance** | `/performance` | Real-time TPS speedometer, TTFT distribution, interactive client load test runner. |
| **Server Metrics** | `/metrics` | Prometheus metrics scraper, histogram charts, request counters, HTTP status breakdowns. |
| **Settings** | `/settings` | Endpoint configuration, theme switcher, API key management, default sampling parameters. |

---

## Implementation Roadmap

- [x] **Step 1:** Project initialization (`package.json`, `tsconfig.json`, `vite.config.ts`, `tailwind.config.js`, `postcss.config.js`, `index.html`).
- [x] **Step 2:** Core theme, CSS styles, animations, and vector icon library (`src/components/icons/Icons.tsx`).
- [x] **Step 3:** Handcrafted UI component library (`Button`, `Card`, `Input`, `Slider`, `Switch`, `Badge`, `Tabs`, `Modal`, `Dropdown`, `Toast`, `Tooltip`, `MarkdownView`).
- [x] **Step 4:** API client services with streaming SSE, audio multipart, and metrics scraping (`src/services/api.ts`, `src/services/sse.ts`, `src/services/metrics.ts`).
- [x] **Step 5:** Layout components (`Sidebar`, `Header`, `StatusIndicator`, `Navigation`).
- [x] **Step 6:** All 10 application pages (`DashboardPage`, `ChatPage`, `VisionPage`, `AudioPage`, `ImageGenPage`, `EmbeddingsPage`, `ModelsPage`, `MemoryPage`, `PerformancePage`, `MetricsPage`, `SettingsPage`).
- [x] **Step 7:** Integration, routing, state management, and production build verification (`npm run build` completed cleanly).
