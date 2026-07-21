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
