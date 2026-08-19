#!/usr/bin/env python3
"""Convert a HuggingFace Whisper checkpoint to GGUF for `qorvix transcribe`.

    pip install transformers torch numpy
    python scripts/convert_whisper_to_gguf.py openai/whisper-tiny models/whisper-tiny-f32.gguf
    python scripts/convert_whisper_to_gguf.py openai/whisper-tiny models/whisper-tiny-f16.gguf --outtype f16

WHY THIS SCRIPT EXISTS AT ALL. Every other model this repo loads is downloaded as GGUF, so no
converter was ever needed. Whisper has no GGUF upstream: whisper.cpp never migrated off its own
legacy container, and the files the Hub advertises as "whisper GGUF" are that container renamed.
Verified rather than assumed - the first four bytes of vonjack/whisper-large-v3-gguf's
`whisper-large-v3-f16.gguf` are `6c 6d 67 67`, i.e. "lmgg", which is ggml's magic (0x67676d6c),
not "GGUF". qorvix names that case explicitly and points here.

So the choice was a second container reader or a converter. The reader loses: the whole weight
path (`detail::tensorBytes` / `loadMat` / `loadVec`, the mmap borrow, every dequant kernel) is
GGUF-only, and a parallel container would need all of it duplicated - the kind of second path
Phase 8.5 spent its time deleting. A converter costs one script and no runtime code.

WHY THE GGUF WRITER IS HAND-ROLLED. llama.cpp's `gguf` package would do it, but it is not a
dependency of this repo and pulling one in for a dev-only script is worse than 60 lines of struct
packing. The format is a magic, a version, typed KV pairs, tensor infos and aligned data - and
this repo owns a strict GGUF *reader*, which validates the output the moment qorvix opens it (bad
alignment, wrong element counts and truncation are all fatal there, not silent).

TENSOR NAMING. HF's names are kept only as the source side of an explicit table. The written names
follow this repo's conventions - `enc.`/`dec.` prefixes, `blk.N.`, `attn_q`/`ffn_up` - so the
loader reads like the BERT and CLIP loaders rather than like a port of transformers. Two Whisper
facts are encoded in the table and would each be silent if guessed:

  * `k_proj` has NO bias, in self-attention AND cross-attention, while q/v/out all do. A loader
    that expects one fails on a missing tensor (loud); a converter that invented a zero one would
    be silently fine, which is worse - it hides the asymmetry from anyone reading the file.
  * `fc1`/`fc2` are an expand/contract pair with the ORDINARY meaning of up/down, unlike the CLIP
    mmproj format where `ffn_down` is the expansion. Named here for what they do.

The proj_out head is tied to the decoder's token embedding, so it is not written twice; the tie is
asserted rather than trusted, because an untied checkpoint silently losing its head would show up
only as fluent nonsense.
"""

import argparse
import json
import struct

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32

# gguf_metadata_value_type, fixed by the format.
T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY = 4, 5, 6, 7, 8, 9
# ggml_type
GGML_F32, GGML_F16 = 0, 1


def _str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pad(n):
    return (n + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


class GgufWriter:
    """Minimal GGUF v3 writer: typed KV metadata, then 32-byte-aligned tensor data."""

    def __init__(self):
        self.kv = []       # (key, type_tag, payload_bytes)
        self.tensors = []  # (name, ggml_type, ne, data_bytes)

    def _kv(self, key, tag, payload):
        self.kv.append((key, tag, payload))

    def u32(self, key, v):
        self._kv(key, T_UINT32, struct.pack("<I", int(v)))

    def f32(self, key, v):
        self._kv(key, T_FLOAT32, struct.pack("<f", float(v)))

    def flag(self, key, v):
        self._kv(key, T_BOOL, struct.pack("<B", 1 if v else 0))

    def string(self, key, v):
        self._kv(key, T_STRING, _str(v))

    def string_array(self, key, values):
        payload = struct.pack("<IQ", T_STRING, len(values)) + b"".join(_str(v) for v in values)
        self._kv(key, T_ARRAY, payload)

    def int32_array(self, key, values):
        payload = struct.pack("<IQ", T_INT32, len(values))
        payload += b"".join(struct.pack("<i", int(v)) for v in values)
        self._kv(key, T_ARRAY, payload)

    def tensor(self, name, array, ggml_type):
        # GGUF stores `ne` with the fastest-varying dimension first, the reverse of a row-major
        # numpy shape. The BYTES are identical either way - only the declared shape differs - so
        # this is about the file being readable by other tools, not about our loader.
        ne = list(reversed(array.shape))
        self.tensors.append((name, ggml_type, ne, array.tobytes()))

    def write(self, path):
        head = bytearray()
        head += GGUF_MAGIC
        head += struct.pack("<IQQ", GGUF_VERSION, len(self.tensors), len(self.kv))
        for key, tag, payload in self.kv:
            head += _str(key) + struct.pack("<I", tag) + payload

        # Tensor offsets are relative to the start of the data section and each one is aligned, so
        # the layout has to be computed before the infos that name it can be written.
        offsets, cursor = [], 0
        for _, _, _, data in self.tensors:
            offsets.append(cursor)
            cursor += _pad(len(data))
        for (name, ttype, ne, _), off in zip(self.tensors, offsets):
            head += _str(name) + struct.pack("<I", len(ne))
            for dim in ne:
                head += struct.pack("<Q", dim)
            head += struct.pack("<IQ", ttype, off)

        with open(path, "wb") as f:
            f.write(head)
            f.write(b"\0" * (_pad(len(head)) - len(head)))  # the data section starts aligned
            for _, _, _, data in self.tensors:
                f.write(data)
                f.write(b"\0" * (_pad(len(data)) - len(data)))
        return _pad(len(head)) + cursor


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a HF Whisper checkpoint to GGUF.")
    ap.add_argument("model_id", help="HF id or local path, e.g. openai/whisper-tiny")
    ap.add_argument("out", help="output .gguf path")
    ap.add_argument("--outtype", choices=("f32", "f16"), default="f32",
                    help="f32 (default) keeps the checkpoint exact; f16 halves the file")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperTokenizerFast

    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.eval()
    tok = WhisperTokenizerFast.from_pretrained(args.model_id)
    cfg = model.config
    sd = model.state_dict()

    d = cfg.d_model
    if cfg.encoder_attention_heads and d % cfg.encoder_attention_heads:
        raise SystemExit(f"d_model {d} is not divisible by {cfg.encoder_attention_heads} heads")

    w = GgufWriter()
    # A tensor is F16 only if it is a matmul weight. Norms and biases stay F32 the way real GGUFs
    # keep them: they are a rounding error's worth of bytes, and they are the first thing a
    # quantization bug would corrupt.
    mat_type = GGML_F16 if args.outtype == "f16" else GGML_F32

    def put(name, key, expect_shape, ggml_type):
        if key not in sd:
            raise SystemExit(f"checkpoint has no '{key}' - is this a Whisper model?")
        t = sd[key].detach().cpu().float().numpy()
        if tuple(t.shape) != tuple(expect_shape):
            raise SystemExit(f"'{key}' is {tuple(t.shape)}, expected {tuple(expect_shape)}")
        t = t.astype(np.float16 if ggml_type == GGML_F16 else np.float32)
        w.tensor(name, np.ascontiguousarray(t), ggml_type)

    def put_norm(prefix, key):
        put(f"{prefix}.weight", f"{key}.weight", (d,), GGML_F32)
        put(f"{prefix}.bias", f"{key}.bias", (d,), GGML_F32)

    def put_attn(prefix, key):
        # k_proj carries no bias in Whisper, in either attention. See the module docstring.
        put(f"{prefix}_q.weight", f"{key}.q_proj.weight", (d, d), mat_type)
        put(f"{prefix}_q.bias", f"{key}.q_proj.bias", (d,), GGML_F32)
        put(f"{prefix}_k.weight", f"{key}.k_proj.weight", (d, d), mat_type)
        if f"{key}.k_proj.bias" in sd:
            raise SystemExit(f"'{key}.k_proj.bias' exists - this checkpoint is not Whisper-shaped")
        put(f"{prefix}_v.weight", f"{key}.v_proj.weight", (d, d), mat_type)
        put(f"{prefix}_v.bias", f"{key}.v_proj.bias", (d,), GGML_F32)
        put(f"{prefix}_out.weight", f"{key}.out_proj.weight", (d, d), mat_type)
        put(f"{prefix}_out.bias", f"{key}.out_proj.bias", (d,), GGML_F32)

    # ---- encoder ----
    n_mels = cfg.num_mel_bins
    enc_ffn = cfg.encoder_ffn_dim
    # Conv1d weights are [out, in, k]. Written flat as an [out, in*k] matmul weight because that is
    # what the CPU stem does with them: one dot product per output channel per output frame over
    # the (in_channels x kernel) window. The row-major bytes are identical, so this is a shape
    # declaration, not a repacking.
    put("enc.conv1.weight", "model.encoder.conv1.weight", (d, n_mels, 3), mat_type)
    put("enc.conv1.bias", "model.encoder.conv1.bias", (d,), GGML_F32)
    put("enc.conv2.weight", "model.encoder.conv2.weight", (d, d, 3), mat_type)
    put("enc.conv2.bias", "model.encoder.conv2.bias", (d,), GGML_F32)
    # Whisper's encoder positions are sinusoidal rather than learned, but they are in the
    # checkpoint, so they are written rather than recomputed: a sin/cos table reimplemented from
    # the paper is one off-by-one - or one interleaving convention - away from a silent drift.
    put("enc.position_embd.weight", "model.encoder.embed_positions.weight",
        (cfg.max_source_positions, d), GGML_F32)
    for i in range(cfg.encoder_layers):
        src = f"model.encoder.layers.{i}"
        put_attn(f"enc.blk.{i}.attn", f"{src}.self_attn")
        put_norm(f"enc.blk.{i}.attn_norm", f"{src}.self_attn_layer_norm")
        put(f"enc.blk.{i}.ffn_up.weight", f"{src}.fc1.weight", (enc_ffn, d), mat_type)
        put(f"enc.blk.{i}.ffn_up.bias", f"{src}.fc1.bias", (enc_ffn,), GGML_F32)
        put(f"enc.blk.{i}.ffn_down.weight", f"{src}.fc2.weight", (d, enc_ffn), mat_type)
        put(f"enc.blk.{i}.ffn_down.bias", f"{src}.fc2.bias", (d,), GGML_F32)
        put_norm(f"enc.blk.{i}.ffn_norm", f"{src}.final_layer_norm")
    put_norm("enc.output_norm", "model.encoder.layer_norm")

    # ---- decoder ----
    dec_ffn = cfg.decoder_ffn_dim
    vocab = cfg.vocab_size
    put("dec.token_embd.weight", "model.decoder.embed_tokens.weight", (vocab, d), mat_type)
    put("dec.position_embd.weight", "model.decoder.embed_positions.weight",
        (cfg.max_target_positions, d), GGML_F32)
    for i in range(cfg.decoder_layers):
        src = f"model.decoder.layers.{i}"
        put_attn(f"dec.blk.{i}.attn", f"{src}.self_attn")
        put_norm(f"dec.blk.{i}.attn_norm", f"{src}.self_attn_layer_norm")
        put_attn(f"dec.blk.{i}.cross_attn", f"{src}.encoder_attn")
        put_norm(f"dec.blk.{i}.cross_attn_norm", f"{src}.encoder_attn_layer_norm")
        put(f"dec.blk.{i}.ffn_up.weight", f"{src}.fc1.weight", (dec_ffn, d), mat_type)
        put(f"dec.blk.{i}.ffn_up.bias", f"{src}.fc1.bias", (dec_ffn,), GGML_F32)
        put(f"dec.blk.{i}.ffn_down.weight", f"{src}.fc2.weight", (d, dec_ffn), mat_type)
        put(f"dec.blk.{i}.ffn_down.bias", f"{src}.fc2.bias", (d,), GGML_F32)
        put_norm(f"dec.blk.{i}.ffn_norm", f"{src}.final_layer_norm")
    put_norm("dec.output_norm", "model.decoder.layer_norm")

    # The LM head is tied to the token embedding. Asserted, not assumed: an untied checkpoint whose
    # head was dropped here would still load and still decode - into fluent nonsense.
    if "proj_out.weight" in sd and not torch.equal(sd["proj_out.weight"],
                                                   sd["model.decoder.embed_tokens.weight"]):
        raise SystemExit("proj_out is NOT tied to embed_tokens; this converter assumes the tie")

    # ---- metadata ----
    w.string("general.architecture", "whisper")
    w.string("general.name", str(args.model_id).replace("\\", "/").split("/")[-1])
    w.u32("general.file_type", 1 if args.outtype == "f16" else 0)
    w.u32("general.alignment", ALIGNMENT)
    w.u32("whisper.embedding_length", d)
    w.u32("whisper.mel_bins", n_mels)
    w.u32("whisper.vocab_size", vocab)
    w.u32("whisper.encoder.block_count", cfg.encoder_layers)
    w.u32("whisper.encoder.attention.head_count", cfg.encoder_attention_heads)
    w.u32("whisper.encoder.feed_forward_length", enc_ffn)
    w.u32("whisper.encoder.context_length", cfg.max_source_positions)
    w.u32("whisper.decoder.block_count", cfg.decoder_layers)
    w.u32("whisper.decoder.attention.head_count", cfg.decoder_attention_heads)
    w.u32("whisper.decoder.feed_forward_length", dec_ffn)
    w.u32("whisper.decoder.context_length", cfg.max_target_positions)
    # HF's Whisper layers use nn.LayerNorm's default eps and the config carries no override, so it
    # is written here explicitly rather than left to a default on the C++ side.
    w.f32("whisper.attention.layer_norm_epsilon", 1e-5)

    # ---- tokenizer ----
    # Byte-level BPE, which this repo's Tokenizer already implements for gpt2/qwen2/llama3 - so the
    # keys are llama.cpp's and no new tokenizer code is needed. The vocabulary is written at the
    # model's full vocab_size, not the tokenizer's length: Whisper's embedding table is larger than
    # its named vocabulary, and a short table would make every id past the end decode as empty
    # instead of as the placeholder it is.
    tokens = tok.convert_ids_to_tokens(list(range(vocab)))
    tokens = [t if isinstance(t, str) else f"<|unused_{i}|>" for i, t in enumerate(tokens)]
    w.string("tokenizer.ggml.model", "gpt2")
    w.string_array("tokenizer.ggml.tokens", tokens)

    tk = json.loads(tok.backend_tokenizer.to_str())
    merges = tk.get("model", {}).get("merges", [])
    # `tokenizers` switched merges from "a b" strings to [a, b] pairs; both are accepted rather
    # than pinning a version, since the pairs carry the same data either way.
    merges = [m if isinstance(m, str) else " ".join(m) for m in merges]
    if not merges:
        raise SystemExit("tokenizer carries no BPE merges - encode() would fall back to bytes")
    w.string_array("tokenizer.ggml.merges", merges)

    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    eot = tok.convert_tokens_to_ids("<|endoftext|>")
    if sot is None or eot is None:
        raise SystemExit("vocabulary has no <|startoftranscript|>/<|endoftext|>")
    w.u32("tokenizer.ggml.bos_token_id", sot)
    w.u32("tokenizer.ggml.eos_token_id", eot)
    # Whisper's prompt is a SEQUENCE of specials (sot, language, task, timestamps), assembled by
    # the model rather than by the tokenizer, so the automatic wrapping is turned off here. With
    # add_bos on, every encode() would silently prepend a second sot.
    w.flag("tokenizer.ggml.add_bos_token", False)
    w.flag("tokenizer.ggml.add_eos_token", False)
    # Everything from <|endoftext|> up is a special token: language, task, timestamps and the
    # unused tail. One boundary is all the decoder needs to strip them from the transcript.
    w.u32("whisper.text_token_end", eot)
    w.flag("whisper.is_multilingual", tok.convert_tokens_to_ids("<|zh|>") is not None)

    # ---- suppression ----
    # Whisper's greedy decode is not argmax over the whole vocabulary. The released models ship a
    # suppression list (88 ids for tiny: mostly punctuation-only and control tokens the model would
    # otherwise emit as text) plus a start-of-sequence list that blocks a leading space and an
    # immediate <|endoftext|>. They are part of the model's decoding contract, not a decoder-side
    # heuristic, so they travel WITH the weights - reimplementing the list on the C++ side would be
    # a second copy of it that drifts on the first model whose list differs, and the resulting
    # transcripts would differ from every other Whisper runtime by a token here and there.
    gen = model.generation_config
    suppress = sorted(set(int(t) for t in (gen.suppress_tokens or [])))
    begin_suppress = [int(t) for t in (gen.begin_suppress_tokens or [])]
    if not suppress:
        print("warning: checkpoint carries no suppress_tokens; transcripts may include punctuation "
              "tokens other Whisper runtimes suppress")
    w.int32_array("whisper.suppress_tokens", suppress)
    w.int32_array("whisper.begin_suppress_tokens", begin_suppress)

    size = w.write(args.out)
    print(f"wrote {args.out}  ({size / 1e6:.1f} MB, {len(w.tensors)} tensors, "
          f"{len(w.kv)} metadata keys, {args.outtype})")
    print(f"  d_model {d}  mels {n_mels}  enc {cfg.encoder_layers}x{cfg.encoder_attention_heads}"
          f"  dec {cfg.decoder_layers}x{cfg.decoder_attention_heads}  vocab {vocab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
