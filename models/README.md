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
python scripts/capture_embed_reference.py sentence-transformers/all-MiniLM-L6-v2 \
  > tests/data/embed_reference_all_minilm_l6_v2.txt
```

Without `--ref` the invariant and triplet-ordering tiers still run and still gate.

### Measured results

```sh
qorvix embed-check models/bge-small-en-v1.5-f16.gguf \
  --ref tests/data/embed_reference_bge_small_en_v15.txt
# Tokenizer parity: 7/7 exact | Vector parity: min cos 1.00000 | RESULT: PASS

qorvix embed-check models/all-MiniLM-L6-v2-Q4_K_M.gguf \
  --ref tests/data/embed_reference_all_minilm_l6_v2.txt --min-cos 0.97
# Tokenizer parity: 7/7 exact | Vector parity: min cos 0.97598 | RESULT: PASS
```

**Thresholds are set from evidence, not taste.** F16 reaches cosine 1.00000 against the fp32
reference, so 0.999 flags real bugs and nothing else. Q4_K_M costs real accuracy — measured
per probe:

| probe | cos |
|---|---|
| "hello world" (4 tok) | 0.99013 |
| "The capital of France is Paris." (9 tok) | 0.99424 |
| "Café naïve résumé" (5 tok) | 0.99282 |
| "antidisestablishmentarianism" (10 tok) | 0.98658 |
| "query: what is machine learning?" (9 tok) | 0.99298 |
| long paragraph (522 tok, truncated) | 0.98610 |
| **empty string (2 tok)** | **0.97598** |

Real text sits at 0.986–0.994. The outlier is the empty string, which is `[CLS] [SEP]` and
nothing else: with mean pooling over two tokens, quantization noise has no other tokens to average
against. That is a property of 4-bit weights on a degenerate input, not an encoder bug — so
`--min-cos 0.97` is the honest gate for Q4_K_M, and the probe stays in the fixture because it also
verifies an empty input does not crash.

`embed-check` prints the full per-probe breakdown whenever the vector tier fails, so a future
regression names which input moved rather than reporting one aggregate number.

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


## Validating vision-language chat (Phase 11b-2)

Two files are needed and **they must be a matching pair** — the mmproj's projector emits vectors
in one specific decoder's input space:

```sh
# the vision half (already required by Phase 11b-1's vision-check)
curl -fsSL -o models/llava-v1.5-7b-mmproj-f16.gguf \
  https://huggingface.co/mys/ggml_llava-v1.5-7b/resolve/main/mmproj-model-f16.gguf
# the matching language model (~4.1 GB)
curl -fsSL -o models/llava-v1.5-7b-Q4_K_M.gguf \
  https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q4_K_M.gguf
```

```sh
qorvix vlm-check models/llava-v1.5-7b-Q4_K_M.gguf \
  --mmproj models/llava-v1.5-7b-mmproj-f16.gguf --image tests/data/vision_probe.png

qorvix generate models/llava-v1.5-7b-Q4_K_M.gguf \
  --mmproj models/llava-v1.5-7b-mmproj-f16.gguf --image tests/data/vision_probe.png \
  --prompt "USER: <image>
What is in this image? ASSISTANT:" --temp 0 --max 40
```

**Pairing is checked before any work happens.** The projector's output width is compared against
the decoder's `d_model` as soon as both headers are mapped, so a mismatched pair costs a second,
not a full CLIP encode:

```
$ qorvix vlm-check models/tinyllama-1.1b-chat-q4km.gguf \
    --mmproj models/llava-v1.5-7b-mmproj-f16.gguf --image tests/data/vision_probe.png
tier 3  image splice
        projector 1024 -> 4096 vs decoder d_model 2048  -> MISMATCH (this mmproj belongs to a different model)
```

### What is and is not verified here

`vlm-check` tiers 1 and 2 need **no vision model at all**, because the property they assert is
self-checking: feeding a token's own embedding row through `forwardEmbedding` must reproduce, bit
for bit, what feeding its id through `forward` produced. Measured on TinyLlama 1.1B Q4_K_M:

```
tier 1  single-step splice identity over 6 tokens
        max |diff| 0.00e+00, argmax mismatches 0  -> PASS
tier 2  full-prefill splice identity (6 positions)
        max |diff| 0.00e+00, argmax identical  -> PASS
```

Tier 3 is a **smoke test, and says so in its own output** — it asserts the projector width, that
the spliced prefill runs, and that the logits are finite. It does not assert output fidelity,
because no LLaVA reference is captured in this repo. Capturing one (the `capture_vision_reference.py`
treatment applied to a full LLaVA forward) is the honest next step.

Like `embed-check`, `gpu-check` and `vulkan-check`, this is a **CLI gate rather than a CTest case**:
the Docker test image has no GGUFs, so a model-dependent test would fail the image build. The
seam's mechanics are covered by `tests/multimodal_test.cpp`, which needs no model.


## Validating speech transcription (Phase 11b-3b)

**Whisper is the one model here that is not downloaded as GGUF, because no GGUF exists.**
whisper.cpp never left its own container, and the Hub files advertised as "whisper GGUF" are that
container renamed — checked rather than assumed:

```sh
curl -sL -r 0-3 https://huggingface.co/vonjack/whisper-large-v3-gguf/resolve/main/whisper-large-v3-f16.gguf | xxd
# 00000000: 6c6d 6767    "lmgg" — ggml's magic 0x67676d6c, not "GGUF"
```

`qorvix transcribe` names that case with its fix rather than reporting "bad magic". Convert the
HuggingFace checkpoint instead:

```sh
pip install transformers torch numpy
python scripts/convert_whisper_to_gguf.py openai/whisper-tiny models/whisper-tiny-f32.gguf
python scripts/convert_whisper_to_gguf.py openai/whisper-tiny models/whisper-tiny-f16.gguf --outtype f16
```

The converter carries the vocabulary (byte-level BPE, so the existing tokenizer reads it), the
protocol token ids, and the model's **suppression lists** — the 88 ids whisper-tiny never emits as
text plus the two blocked at the first generated position. Those are part of the decoding contract,
not a runtime heuristic: without them the greedy argmax after the prefix is token 522, and every
other Whisper implementation produces 708.

| file | size | notes |
|---|---|---|
| `whisper-tiny-f32.gguf` | 153 MB | 167 tensors, 27 metadata keys |
| `whisper-tiny-f16.gguf` | 79 MB | **bit-identical numbers** — see below |

**F16 is lossless for these checkpoints, and slower.** All 67 matmul tensors compare byte-exact
after the F32→F16 round trip, because OpenAI released Whisper in fp16 and the safetensors are an
upcast of that — so the F16 file is half the size for no accuracy cost and `whisper-check` returns
identical figures for both. It is nevertheless the slower file to decode with here (see
BENCHMARKS.md): the F32 dot kernel is AVX2, the F16 path dequantizes per element, and the LM head
is a 51,865 × 384 matmul on every token.

```sh
qorvix transcribe models/whisper-tiny-f32.gguf --audio tests/data/speech_probe.wav
qorvix transcribe models/whisper-tiny-f32.gguf --audio clip.wav --language auto --timestamps
qorvix serve models/tinyllama-1.1b-chat-q4km.gguf --whisper models/whisper-tiny-f32.gguf
curl -X POST localhost:2005/v1/audio/transcriptions -F file=@tests/data/speech_probe.wav
```

### The gate

```sh
pip install transformers torch numpy
python scripts/capture_whisper_reference.py openai/whisper-tiny tests/data/speech_probe.wav \
  > tests/data/whisper_reference_tiny.txt

qorvix whisper-check models/whisper-tiny-f32.gguf --ref tests/data/whisper_reference_tiny.txt
```

Like `embed-check`, `vision-check`, `gpu-check` and `vulkan-check`, this is a **CLI gate rather
than a CTest case**: the Docker test image has no GGUFs. The model-free mechanics — the stem's
padding and stride, the protocol prefix, the two caches' lifetimes, the refusals — are covered by
`tests/whisper_test.cpp`.

### Measured results

```
Forced prefix:                   50258 50259 50359 50363   MATCHES reference
Encoder position 0:              max |diff| 1.91e-06  cos 1.0000000
Encoder per-dimension means:     max |diff| 4.77e-06  cos 1.0000000  (over 1500 positions)
Raw argmax after the prefix:     400 vs reference 400   MATCHES
Top-5 logits:                    max |diff| 4.77e-05  ids identical
Fixed logit probes:              max |diff| 4.48e-05  (5 ids)
Greedy token stream:             23 tokens vs 23   IDENTICAL
Transcript:                      IDENTICAL
  ours:      " And so my fellow Americans ask not what your country can do for you ask what you can do for your country."

RESULT: PASS - encoder, decoder step and greedy transcript all match transformers.
```

The tiers are ordered so each one narrows the cause of the next, because an encoder bug, a
cross-attention bug and a missing suppression list all end the same way — a transcript that is
fluent and not what other runtimes produce. Tier 2 compares **raw** logits (no suppression), so a
suppression difference cannot hide inside it and a cross-attention difference cannot be blamed on
it.

### What is not implemented, and why it is refused rather than approximated

- **Long-form audio.** The encoder consumes exactly one 30-second window. Whisper's sequential
  algorithm advances by the last emitted timestamp and carries context through `<|startofprev|>`;
  neither exists here. The CLI prints a warning and transcribes the first window; the HTTP route
  refuses with a 400, because there is no one on that side to read a warning.
- **Resampling.** 16 kHz only, with the `ffmpeg` command in the error. An ungated resampler in
  front of the mel filters returns a transcript that is fluent and slightly wrong.
- **`prompt` conditioning and `temperature > 0`** on `/v1/audio/transcriptions`: both refused, not
  ignored — answering a sampled request with a greedy transcript misreports what was run.
- **Beam search and the temperature-fallback loop.** Decoding is greedy, which is what the
  reference capture compares against.
