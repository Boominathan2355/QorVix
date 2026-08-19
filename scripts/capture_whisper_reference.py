#!/usr/bin/env python3
"""Capture a reference Whisper encoder/decoder trace for `qorvix whisper-check`.

Run once, commit the output, never run in CI or the build — a development harness, the same status
as the other capture scripts.

    pip install transformers torch numpy
    python scripts/capture_whisper_reference.py openai/whisper-tiny tests/data/audio_probe.wav \
        > tests/data/whisper_reference_tiny.txt

Ground truth is transformers' own WhisperForConditionalGeneration in fp32 — independent of this
codebase and of whisper.cpp, so it validates the GGUF conversion as well as the forward pass.

WHY THREE TIERS AND NOT ONE VERDICT. Phase 11b-1 gated CLIP with one aggregate number, found both
of its bugs in preprocessing, and would have read them as "the transformer is slightly wrong".
Whisper has three places to be wrong that fail identically at the end — a transcript that is
plausible and not what every other runtime produces — so each is pinned separately:

  * `enc_*` — the encoder alone: the convolutional stem, the sinusoidal positions and the
    bidirectional blocks. Wrong padding on either convolution, or a stride-2 off-by-one, shifts
    every frame; nothing downstream can distinguish that from a decoder bug.
  * `logits_*` — one decoder step over the forced prefix. This is where cross-attention lives, and
    a cross-attention that reads the wrong keys still produces confident logits.
  * `tokens` / `text` — the greedy loop, including suppression. Two runtimes agreeing on logits can
    still disagree here if one of them ignores the model's suppress lists.

The probe may be synthetic (a tone sweep rather than speech). That is fine and deliberate for a
PARITY gate: both implementations see identical input and must produce identical output, whether or
not the text means anything. Accuracy on real speech is a separate claim, made by transcribing real
speech, not by this fixture.
"""

import sys
import wave


def read_wav_mono_16k(path):
    """Decode a 16-bit PCM WAV to mono float in [-1, 1], refusing anything else loudly.

    Deliberately not librosa: it would resample and could dither, so the fixture would encode
    librosa's decode of the file rather than the file itself.
    """
    import numpy as np

    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16-bit PCM, got {w.getsampwidth() * 8}-bit")
        rate = w.getframerate()
        if rate != 16000:
            raise SystemExit(
                f"{path}: expected 16 kHz (Whisper's only rate), got {rate}. "
                f"Convert with: ffmpeg -i {path} -ac 1 -ar 16000 -c:a pcm_s16le out.wav"
            )
        raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
        return rate, data


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    model_id, wav_path = sys.argv[1], sys.argv[2]
    language = sys.argv[3] if len(sys.argv) > 3 else "en"
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 64

    import numpy as np
    import torch
    from transformers import (WhisperFeatureExtractor, WhisperForConditionalGeneration,
                             WhisperTokenizerFast)

    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    tok = WhisperTokenizerFast.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=torch.float32)
    model.eval()

    rate, samples = read_wav_mono_16k(wav_path)
    feats = fe(samples, sampling_rate=rate, return_tensors="pt")["input_features"]

    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    lang_tok = tok.convert_tokens_to_ids(f"<|{language}|>")
    task_tok = tok.convert_tokens_to_ids("<|transcribe|>")
    nots = tok.convert_tokens_to_ids("<|notimestamps|>")
    if None in (sot, lang_tok, task_tok, nots):
        raise SystemExit(f"vocabulary is missing a protocol token for language '{language}'")
    prompt = [sot, lang_tok, task_tok, nots]

    with torch.no_grad():
        enc = model.model.encoder(feats).last_hidden_state[0].numpy()  # [enc_ctx, d]
        # The same encoder output feeds the step below, so a logits mismatch cannot be blamed on
        # the encoder once the encoder tier passes.
        out = model(input_features=feats, decoder_input_ids=torch.tensor([prompt]))
        step_logits = out.logits[0, -1].numpy()  # the first generated position
        gen = model.generate(feats, language=language, task="transcribe", return_timestamps=False,
                             do_sample=False, num_beams=1, max_new_tokens=max_new)
    gen_ids = [int(i) for i in gen[0].tolist()]
    text = tok.decode(gen_ids, skip_special_tokens=True)

    ctx, d = enc.shape
    order = np.argsort(-step_logits)[:5]

    print("# qorvix whisper reference fixture v1")
    print(f"# model:   {model_id}")
    print(f"# audio:   {wav_path}")
    print("# source:  transformers WhisperForConditionalGeneration, torch fp32, CPU — NOT qorvix")
    print("# capture: scripts/capture_whisper_reference.py")
    print("#")
    print("# Regenerate rather than hand-edit; a fixture produced by qorvix itself would make")
    print("# whisper-check self-referential and prove nothing.")
    print(f"model {model_id.split('/')[-1]}")
    print(f"d_model {d}")
    print(f"enc_ctx {ctx}")
    print(f"vocab {model.config.vocab_size}")
    print(f"language {language}")
    print(f"max_new_tokens {max_new}")
    print("prompt " + " ".join(str(i) for i in prompt))

    # Encoder: position 0 in full, plus the per-dimension mean over every position. The means catch
    # a frame shift that leaves position 0 intact (a stride-2 off-by-one does exactly that).
    print("enc_frame0 " + " ".join(f"{v:.6f}" for v in enc[0]))
    print("enc_dim_means " + " ".join(f"{v:.6f}" for v in enc.mean(axis=0)))

    print(f"argmax0 {int(order[0])}")
    print("logits_top " + " ".join(f"{int(i)} {float(step_logits[i]):.5f}" for i in order))
    # A few fixed ids as well, so the tier still says something when the top-5 happen to agree by
    # luck on a degenerate clip.
    for probe in (0, 1000, 20000, 50256, model.config.vocab_size - 1):
        if 0 <= probe < len(step_logits):
            print(f"logit_probe {probe} {float(step_logits[probe]):.5f}")

    print("tokens " + " ".join(str(i) for i in gen_ids))
    # One line, whitespace preserved as-is; the C++ side compares the raw remainder of the line.
    print("text " + text.replace("\n", "\\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
