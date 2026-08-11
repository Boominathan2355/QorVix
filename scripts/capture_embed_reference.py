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
    pooling = "cls"
    for mod in model.modules():
        if type(mod).__name__ == "Pooling":
            if getattr(mod, "pooling_mode_mean_tokens", False):
                pooling = "mean"
            elif getattr(mod, "pooling_mode_cls_token", False):
                pooling = "cls"
            break

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
    for text, vec in zip(TEXTS, vecs):
        ids = tok(text)["input_ids"]
        print(f"text {text}")
        print("ids " + " ".join(str(i) for i in ids))
        print("vec " + " ".join(f"{v:.8g}" for v in vec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
