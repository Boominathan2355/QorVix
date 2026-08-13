#!/usr/bin/env python3
"""Capture reference CLIP vision features for `qorvix vision-check`.

Run once, commit the output, never run in CI or the build — a development harness, the same
status as scripts/capture_embed_reference.py and scripts/colab_llamacpp_compare.sh.

    pip install transformers torch pillow
    python scripts/capture_vision_reference.py openai/clip-vit-large-patch14-336 \
        tests/data/vision_probe.png > tests/data/vision_reference_clip_vit_l14_336.txt

What it records, and why each piece is separable:

  * `pixels` — a checksum and a few samples of the PREPROCESSED tensor. Preprocessing is the
    highest-risk part of the vision path: the weights are exact, but a different resize filter,
    crop origin or normalization constant moves every output value while erroring on nothing.
    Recording it separately means a failure says "preprocessing" or "the transformer", not just
    "something".
  * `hidden` — the second-to-last hidden state, patch tokens only (the class token dropped).
    That is precisely what a LLaVA mmproj tower produces: its conversion already removes the
    final block, so `hidden_states[-2]` of the 24-layer model is the output of all 23 blocks
    the GGUF actually contains.
"""

import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 1
    model_id, image_path = sys.argv[1], sys.argv[2]

    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModel

    proc = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPVisionModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()

    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    pixels = inputs["pixel_values"][0].numpy()  # [3, H, W]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-2][0].numpy()  # [1 + patches, d]
    patches = hidden[1:]  # drop the class token, as LLaVA does

    print("# qorvix vision reference fixture v1")
    print(f"# model:  {model_id}")
    print(f"# image:  {image_path}")
    print("# source: transformers CLIPVisionModel, torch fp32, CPU — NOT qorvix")
    print("# capture: scripts/capture_vision_reference.py")
    print("#")
    print("# Regenerate rather than hand-edit; a fixture produced by qorvix itself would make")
    print("# vision-check self-referential and prove nothing.")
    print(f"model {model_id.split('/')[-1]}")
    print(f"image_size {pixels.shape[1]}")
    print(f"patches {patches.shape[0]}")
    print(f"dim {patches.shape[1]}")
    print(f"pixel_mean {float(pixels.mean()):.8g}")
    print(f"pixel_absmean {float(np.abs(pixels).mean()):.8g}")
    # A handful of exact pixel values pins the resize + crop + normalize chain without committing
    # a 336x336x3 tensor.
    for c, y, x in [(0, 0, 0), (1, 100, 150), (2, 200, 50), (0, 335, 335), (1, 168, 168)]:
        print(f"pixel {c} {y} {x} {float(pixels[c, y, x]):.8g}")

    # Every patch row's mean, plus the first row in full: cheap to store, and enough that a
    # localized error (one layer, one head) still shows up somewhere.
    print("row0 " + " ".join(f"{v:.8g}" for v in patches[0]))
    print("rowmeans " + " ".join(f"{float(patches[i].mean()):.8g}" for i in range(patches.shape[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
