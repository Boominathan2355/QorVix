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

## Validating the embeddings engine (Phase 11a)

Two encoder models, chosen so they differ in the one parameter most likely to be hardcoded by
accident — **pooling** — and in quantization:

```sh
curl -fsSL -o models/bge-small-en-v1.5-f16.gguf \
  https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf
curl -fsSL -o models/all-MiniLM-L6-v2-Q4_K_M.gguf \
  https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf
```

| | bge-small-en-v1.5 | all-MiniLM-L6-v2 |
|---|---|---|
| size | 67 MB (F16) | 21 MB (Q4_K_M) |
| pooling | **cls** | **mean** |
| layers | 12 | 6 |
| quant | F16 throughout | Q8_0 embeddings, Q4_K matmuls |

F16 is the gate model deliberately: against an fp32 reference it should reach cosine > 0.9999, so
a 0.999 threshold flags real bugs and nothing else. At Q4_K_M the honest floor is ~0.99, and a
0.99 threshold is loose enough to hide a genuine pooling error — so the quantized model gets its
own looser run purely to prove the quantized path works.

```sh
qorvix model-info  models/bge-small-en-v1.5-f16.gguf
qorvix embed       models/bge-small-en-v1.5-f16.gguf --text "hello world"
qorvix embed-check models/bge-small-en-v1.5-f16.gguf
```

`embed-check` is a **CLI gate, not a CTest case** — the Docker test image has no GGUF (they are
gitignored), so a model-dependent test would fail the image build. Same status as `gpu-check` and
`vulkan-check`.

Its strongest tier needs a reference captured once from sentence-transformers:

```sh
pip install sentence-transformers torch
python scripts/capture_embed_reference.py BAAI/bge-small-en-v1.5 \
  > tests/data/embed_reference_bge_small_en_v15.txt
qorvix embed-check models/bge-small-en-v1.5-f16.gguf \
  --ref tests/data/embed_reference_bge_small_en_v15.txt
```

Without `--ref` the invariant and triplet-ordering tiers still run and still gate.

### What these files actually contain

Read off the real GGUFs with `qorvix gguf-info`, not assumed — two of these corrected wrong
guesses during Phase 11a:

- Architecture `bert`; tokenizer `bert` (WordPiece, uncased, vocab 30522).
- `bert.attention.layer_norm_epsilon = 1e-12` — **not** the Llama-family `...rms_epsilon` key.
- `bert.attention.causal = false`, `bert.pooling_type` (2 = cls, 1 = mean).
- `tokenizer.ggml.token_type_count` — under the **tokenizer** prefix, not the architecture one.
- `tokenizer.ggml.seperator_token_id` — llama.cpp's misspelling is what is actually on disk.
- The vocabulary uses SentencePiece shape: word-initial `▁the` (1996), continuation pieces bare
  (`the` at 10760), **not** HuggingFace's `##` convention. Punctuation is marked too (`▁,` = 1010).
- Tensors: `token_embd`, `token_embd_norm.{weight,bias}`, `token_types`, `position_embd`, and per
  layer `attn_{q,k,v,output}.{weight,bias}`, `attn_output_norm.{weight,bias}`,
  `ffn_{up,down}.{weight,bias}`, `layer_output_norm.{weight,bias}`. No `ffn_gate`, no
  `output_norm`, no `output.weight` — an encoder has no LM head.

