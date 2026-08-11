#!/usr/bin/env python3
"""Capture reference embeddings for `qorvix embed-check`.

Run this ONCE per model, commit the output, and never run it in CI or the build. It is a
development harness, not a dependency — the same status as scripts/colab_llamacpp_compare.sh.
docs/SPEC.md forbids depending on another runtime *inside* Qorvix; nothing here is linked into
the binary.

Why sentence-transformers specifically: it is independent of BOTH this codebase and llama.cpp, so
a match validates the GGUF conversion as well as the C++ encoder. Comparing against
llama.cpp's llama-embedding instead would share the exact quantized weights (a tighter threshold)
but also share any tokenizer quirk, which is the failure mode most worth catching.

    pip install sentence-transformers torch
    python scripts/capture_embed_reference.py BAAI/bge-small-en-v1.5 \
        > tests/data/embed_reference_bge_small_en_v15.txt

Then:
    qorvix embed-check models/bge-small-en-v1.5-f16.gguf
"""

import sys

# The seven probes, each chosen so a failure names its own cause. Keep them in sync with the
# fixture's documented intent — adding one is fine, reordering is not (the file is positional
# only in the sense that each record is self-describing, but the comments below explain each).
TEXTS = [
    "hello world",                          # all-common-vocab: a failure here is encoder math
    "The capital of France is Paris.",      # lowercasing + punctuation splitting
    "Café naïve résumé",  # accent stripping
    "antidisestablishmentarianism",         # ## continuation / longest-match-first
    "query: what is machine learning?",     # ':' and '?' splitting; a realistic RAG query
    "",                                     # [CLS][SEP] alone must not crash
    (                                       # long: truncation at ctx=512 and long-seq pooling
        "Retrieval augmented generation combines a document retriever with a language model. "
        * 40
    ),
]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    name = sys.argv[1]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name)
    tok = model.tokenizer
    # sentence-transformers >= 3 exposes a single `pooling_mode` string; older versions used
    # boolean pooling_mode_* flags. Read both, and fail loudly rather than defaulting — a fixture
    # that silently claims the wrong pooling would make embed-check compare cls against mean and
    # report an encoder bug that does not exist.
    pooling = None
    for mod in model.modules():
        if type(mod).__name__ != "Pooling":
            continue
        mode = getattr(mod, "pooling_mode", None)
        if mode:
            pooling = str(mode)
        elif getattr(mod, "pooling_mode_mean_tokens", False):
            pooling = "mean"
        elif getattr(mod, "pooling_mode_cls_token", False):
            pooling = "cls"
        break
    if pooling not in ("mean", "cls", "lasttoken"):
        raise SystemExit(f"could not determine pooling mode (got {pooling!r})")
    if pooling == "lasttoken":
        pooling = "last"

    # Truncate the recorded ids exactly as the model truncates internally when encoding. Calling
    # the tokenizer bare returns the FULL sequence (522 tokens for the long probe), while the
    # vector beside it was computed from the first 512 — so an untruncated id row would make the
    # tokenizer tier report a mismatch that is an artefact of the capture, not a real difference.
    max_len = model.max_seq_length
    vecs = model.encode(TEXTS, normalize_embeddings=True, convert_to_numpy=True)

    print("# qorvix embedding reference fixture v1")
    print(f"# model:  {name}")
    print("# source: sentence-transformers, torch fp32, CPU — NOT qorvix and NOT llama.cpp")
    print("# capture: scripts/capture_embed_reference.py (see models/README.md)")
    print("#")
    print("# Regenerate rather than hand-edit. A fixture produced by Qorvix itself would make")
    print("# embed-check self-referential and prove nothing.")
    print(f"model {name.split('/')[-1]}")
    print(f"dim {vecs.shape[1]}")
    print(f"pooling {pooling}")
    print("normalize 1")
    print(f"max_seq_len {max_len}")
    for text, vec in zip(TEXTS, vecs):
        ids = tok(text, truncation=True, max_length=max_len)["input_ids"]
        print(f"text {text}")
        print("ids " + " ".join(str(i) for i in ids))
        print("vec " + " ".join(f"{v:.8g}" for v in vec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
