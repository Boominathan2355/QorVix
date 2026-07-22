# Models

Drop GGUF model files here. Qorvix discovers them automatically — `qorvix scan-models` lists
them, and the runtime's directory watcher registers new files without a restart.

```
models/
  llama3.gguf
  qwen3.gguf
  ...
```

Model files (`*.gguf`, `*.safetensors`, `*.bin`) are gitignored and never committed.

## Validating the CPU runtime (Phase 5)

To validate `qorvix generate` against a real model, a small one is ideal (fast to load, easy to
eyeball). Good first choices, all Llama-family and supported by the current loader:

- **TinyLlama 1.1B Chat** — `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`, file
  `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (~670 MB). SPM tokenizer.
- **Qwen2 0.5B Instruct** — `Qwen/Qwen2-0.5B-Instruct-GGUF`, a `*q4_k_m.gguf` (~400 MB). BPE
  tokenizer.

Once a file is here:

```sh
qorvix model-info models/<file>.gguf        # confirm the derived config
qorvix generate  models/<file>.gguf --prompt "The capital of France is" --temp 0 --max 40
```

Greedy (`--temp 0`) output should match llama.cpp for the same prompt/model. A mismatch most
likely points at the RoPE mode (interleaved vs NeoX) for that architecture — the one parameter
the synthetic tests can't pin down.

