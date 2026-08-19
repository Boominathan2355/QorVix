#!/usr/bin/env python3
"""Generate tests/data/audio_probe.wav — the deterministic clip audio-check runs on.

    python scripts/make_audio_probe.py tests/data/audio_probe.wav

Run once and commit the result. It is regenerated only if the probe's content needs to change, in
which case the reference fixture must be recaptured too (scripts/capture_audio_reference.py) —
they are a matched pair, and a fixture captured from a different waveform would fail in a way that
looks like a front-end bug.

WHY THIS CONTENT. A front end can be wrong in ways a plain tone will not reveal, so each section
targets one:

  * chirp 80 Hz -> 7600 Hz sweeps every mel filter in turn, so a bank that is shifted or wrongly
    normalized shows up as a diagonal that drifts rather than a single bad bin.
  * a 440 Hz + 1970 Hz pair straddles 1000 Hz, which is exactly where the slaney mel scale switches
    from linear to logarithmic. Picking the htk scale instead moves these two tones by different
    amounts, which a single tone anywhere could not distinguish from a gain error.
  * a silent stretch INSIDE the clip exercises the 1e-10 log floor and the max-8 dynamic-range
    clamp somewhere other than the zero padding.
  * a click is the broadband case: it excites every bin in one frame, so a window applied with the
    wrong length or symmetry smears it visibly.
  * an LCG noise tail is deterministic across platforms (Python's `random` is not pinned across
    versions, numpy's default_rng is, but an explicit LCG needs neither).

Two seconds at 16 kHz mono is 64 KB, and Whisper pads it to 30 s regardless — so the fixture also
covers the padding path, where ~87% of the frames sit at the clamp floor.
"""

import math
import struct
import sys
import wave

RATE = 16000
SECONDS = 2.0
N = int(RATE * SECONDS)


def build():
    s = [0.0] * N

    def idx(t):
        return int(t * RATE)

    # 0.00 - 1.20 s: linear chirp, 80 Hz -> 7600 Hz. Phase is integrated analytically so the
    # instantaneous frequency really is linear in t (stepping the phase by a changing increment
    # would drift).
    f0, f1 = 80.0, 7600.0
    a, b = idx(0.0), idx(1.20)
    dur = (b - a) / RATE
    for i in range(a, b):
        t = (i - a) / RATE
        phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t * t)
        s[i] = 0.6 * math.sin(phase)

    # 1.20 - 1.40 s: silence.

    # 1.40 - 1.60 s: two tones straddling the 1 kHz mel knee.
    a, b = idx(1.40), idx(1.60)
    for i in range(a, b):
        t = (i - a) / RATE
        s[i] = 0.3 * math.sin(2 * math.pi * 440.0 * t) + 0.3 * math.sin(2 * math.pi * 1970.0 * t)

    # 1.60 - 1.62 s: three clicks, one per ~6 ms.
    for k in range(3):
        i = idx(1.60) + k * 96
        if i < N:
            s[i] = 0.9

    # 1.62 - 2.00 s: deterministic LCG noise (glibc constants), mapped to [-0.25, 0.25].
    state = 12345
    for i in range(idx(1.62), N):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        s[i] = 0.25 * ((state / 0x7FFFFFFF) * 2.0 - 1.0)

    return s


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/data/audio_probe.wav"
    samples = build()
    # Round-half-away-from-zero to int16, clamped. Truncation would bias the whole clip toward
    # zero by half a level, which is small but would sit in the fixture forever.
    pcm = bytearray()
    for v in samples:
        q = int(math.floor(v * 32767.0 + 0.5)) if v >= 0 else int(math.ceil(v * 32767.0 - 0.5))
        pcm += struct.pack("<h", max(-32768, min(32767, q)))

    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(pcm))
    print(f"wrote {out}: {N} samples, {SECONDS}s @ {RATE} Hz mono 16-bit ({len(pcm)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
