#!/usr/bin/env python3
"""Capture a reference Whisper log-mel spectrogram for `qorvix audio-check`.

Run once, commit the output, never run in CI or the build — a development harness, the same status
as scripts/capture_vision_reference.py and scripts/capture_embed_reference.py.

    pip install transformers torch numpy
    python scripts/capture_audio_reference.py openai/whisper-tiny tests/data/audio_probe.wav \
        > tests/data/audio_reference_whisper_tiny.txt

The reference is transformers' WhisperFeatureExtractor, which is independent of this codebase and
of whisper.cpp — so it validates the front end rather than confirming that two ports of the same
code agree with each other.

WHAT IS RECORDED, AND WHY EACH PIECE IS SEPARABLE. Phase 11b-1 gated CLIP and found that BOTH of
its bugs were in preprocessing rather than the transformer, with one aggregate verdict they would
both have read as "the transformer is slightly wrong". So the audio front end is tiered the same
way, each tier failing for a different cause:

  * `filter` / `filtersums` — the mel filter bank alone. A pure function of the config with no
    audio in it, so a mismatch means the mel SCALE (slaney vs htk) or the area normalization is
    wrong, and neither the FFT nor the window nor the padding is implicated.
  * `waveform_*` — a checksum of the decoded samples. Separates "the WAV was read wrong" from
    "the spectrogram was computed wrong"; without it those two present identically.
  * `frame0` / `melmeans` — the finished features. Only meaningful once the tiers below agree, at
    which point the remaining suspects are the window shape, the reflect padding, power-vs-
    magnitude, and the dynamic-range clamp.

The WAV is read here WITHOUT librosa on purpose. librosa would resample and could apply its own
dithering, so the fixture would encode librosa's decode of the file rather than the file. Reading
the PCM directly means qorvix and this script start from identical samples, and any difference in
`waveform_*` is qorvix's WAV reader rather than a disagreement about what the file contains.
"""

import sys
import wave


def read_wav_mono_16k(path):
    """Decode a 16-bit PCM WAV to mono float in [-1, 1], refusing anything else loudly."""
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

    import numpy as np
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained(model_id)
    rate, samples = read_wav_mono_16k(wav_path)

    feats = fe(samples, sampling_rate=rate, return_tensors="np")["input_features"][0]  # [mels, T]
    n_mels, frames = feats.shape

    # The filter bank as the extractor itself built it — not rebuilt here, or the fixture would be
    # checking this script's arithmetic instead of the extractor's.
    bank = np.asarray(fe.mel_filters)  # [bins, n_mels]
    bins = bank.shape[0]

    print("# qorvix audio reference fixture v1")
    print(f"# model:   {model_id}")
    print(f"# audio:   {wav_path}")
    print("# source:  transformers WhisperFeatureExtractor, numpy — NOT qorvix")
    print("# capture: scripts/capture_audio_reference.py")
    print("#")
    print("# Regenerate rather than hand-edit; a fixture produced by qorvix itself would make")
    print("# audio-check self-referential and prove nothing.")
    print(f"model {model_id.split('/')[-1]}")
    print(f"sample_rate {rate}")
    print(f"n_fft {fe.n_fft}")
    print(f"hop_length {fe.hop_length}")
    print(f"n_mels {n_mels}")
    print(f"frames {frames}")

    print(f"waveform_samples {len(samples)}")
    print(f"waveform_mean {float(samples.mean()):.9f}")
    print(f"waveform_absmean {float(np.abs(samples).mean()):.9f}")

    # Probes spread across the bank rather than clustered: low mels are narrow and land on few
    # bins, high mels are wide, and a normalization error shows up far more strongly at the top.
    for mel in range(0, n_mels, max(1, n_mels // 8)):
        col = bank[:, mel]
        peak = int(np.argmax(col))
        print(f"filter {peak} {mel} {float(col[peak]):.9f}")
    print("filtersums " + " ".join(f"{v:.9f}" for v in bank.sum(axis=0)))
    print(f"# filter bank is [{bins} bins x {n_mels} mels]")

    print("frame0 " + " ".join(f"{v:.7f}" for v in feats[:, 0]))
    print("melmeans " + " ".join(f"{v:.7f}" for v in feats.mean(axis=1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
