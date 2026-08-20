#!/usr/bin/env python3
"""Capture a diffusers reference trace for `qorvix sd-check`.

    pip install diffusers transformers torch numpy
    python scripts/capture_sd_reference.py hf-internal-testing/tiny-stable-diffusion-torch \
        tests/data/sd_reference_tiny.txt --size 32 --steps 4

The ground truth is diffusers running on fp32 CPU torch - independent of this codebase and of
stable-diffusion.cpp. Never regenerate a fixture from qorvix's own output; a self-referential gate
passes forever and proves nothing.

WHY THE TRACE IS TIERED. A wrong image has too many possible causes to debug from the image. Each
tier below is chosen so that it can only fail for reasons the tiers above it have already cleared:

  schedule    the beta ladder, the visited timesteps, alpha_bar - pure arithmetic over the config,
              no weights at all. A wrong beta schedule produces a picture, just a washed-out one.
  tokenizer   the 77 ids the prompt becomes. CLIP's BPE is not the byte-level BPE this repo
              already had, and a wrong split is invisible downstream: the text encoder happily
              encodes whatever it is given.
  conditioning the text encoder's output. Catches the causal mask, quick-GELU vs GELU, and the
              padding convention - all of which produce confident, smooth, wrong vectors.
  unet        ONE forward pass over a latent the fixture supplies, so no sampling and no RNG is
              in the comparison. This is where cross-attention, the skip stack, the GEGLU halves
              and the two GroupNorm epsilons live.
  loop        the whole sampler from the same latent, with guidance. Two runtimes that agree on a
              single step still diverge here if they disagree about `scale_model_input`, the
              guidance formula, or the step index the sigma table is read at.
  vae         the decode, compared as floats rather than as pixels, so an 8-bit rounding rule is
              not mistaken for a decoder bug.

THE LATENT IS IN THE FIXTURE. qorvix's noise is its own (see image/rng.hpp), so a shared seed would
prove nothing. Writing the starting latent out means both runtimes denoise the SAME tensor and the
random number generator is not part of what is being compared.
"""

import argparse
import sys


def fmt(values, precision=6):
    return " ".join("%.*f" % (precision, float(v)) for v in values)


def main():
    ap = argparse.ArgumentParser(description="Capture a diffusers reference trace for sd-check")
    ap.add_argument("model", help="HF repo id or local diffusers directory")
    ap.add_argument("out", help="fixture path to write")
    ap.add_argument("--prompt", default="a photograph of an astronaut riding a horse")
    ap.add_argument("--negative", default="blurry, low quality")
    ap.add_argument("--size", type=int, default=32, help="square pixel size")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--sampler", choices=["ddim", "euler"], default="ddim")
    ap.add_argument("--clip-skip", type=int, default=1,
                    help="1 = final hidden layer (the usual default), 2 = penultimate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import torch
        from diffusers import DDIMScheduler, EulerDiscreteScheduler, StableDiffusionPipeline
    except ImportError as exc:
        raise SystemExit("missing dependency: %s\n  pip install diffusers transformers torch numpy" % exc)

    torch.set_grad_enabled(False)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, safety_checker=None, requires_safety_checker=False)
    pipe.to("cpu")

    sched_cls = DDIMScheduler if args.sampler == "ddim" else EulerDiscreteScheduler
    scheduler = sched_cls.from_config(pipe.scheduler.config)

    tokenizer, text_encoder, unet, vae = pipe.tokenizer, pipe.text_encoder, pipe.unet, pipe.vae
    ctx = text_encoder.config.max_position_embeddings
    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1)
    if args.size % (vae_scale * 2 ** (len(unet.config.block_out_channels) - 1)) != 0:
        raise SystemExit("--size must be a multiple of %d for this model"
                         % (vae_scale * 2 ** (len(unet.config.block_out_channels) - 1)))

    def encode(text):
        ids = tokenizer(text, padding="max_length", max_length=ctx, truncation=True,
                        return_tensors="pt").input_ids
        if args.clip_skip <= 1:
            emb = text_encoder(ids)[0]
        else:
            # diffusers spells "use the penultimate layer" as clip_skip=1; the tooling around
            # these models spells the same thing as 2. This script and qorvix both use the latter.
            hs = text_encoder(ids, output_hidden_states=True).hidden_states
            emb = text_encoder.text_model.final_layer_norm(hs[-args.clip_skip])
        return ids[0].tolist(), emb

    ids, cond = encode(args.prompt)
    neg_ids, uncond = encode(args.negative)

    scheduler.set_timesteps(args.steps)
    timesteps = [int(t) for t in scheduler.timesteps]

    latent_edge = args.size // vae_scale
    generator = torch.Generator("cpu").manual_seed(args.seed)
    latents = torch.randn((1, unet.config.in_channels, latent_edge, latent_edge),
                          generator=generator, dtype=torch.float32)
    latents = latents * scheduler.init_noise_sigma

    # Tier: one UNet forward pass, over the fixture's own latent, with no guidance mixed in.
    t0 = scheduler.timesteps[0]
    unet_in = scheduler.scale_model_input(latents, t0)
    unet_out = unet(unet_in, t0, encoder_hidden_states=cond).sample

    # Tier: the whole loop, with guidance.
    lat = latents.clone()
    guided = args.guidance > 1.0
    for t in scheduler.timesteps:
        model_in = scheduler.scale_model_input(lat, t)
        pred = unet(model_in, t, encoder_hidden_states=cond).sample
        if guided:
            pred_u = unet(model_in, t, encoder_hidden_states=uncond).sample
            pred = pred_u + args.guidance * (pred - pred_u)
        lat = scheduler.step(pred, t, lat).prev_sample

    image = vae.decode(lat / vae.config.scaling_factor).sample

    alphas = scheduler.alphas_cumprod.tolist()
    # Probe indices rather than the whole 1000-entry ladder: enough to pin the curve's ends, its
    # middle, and the exact entries this run reads.
    probe_idx = sorted(set([0, 1, len(alphas) // 2, len(alphas) - 1] + [max(0, t) for t in timesteps]))

    def dim_means(t):
        return t.reshape(-1, t.shape[-1]).mean(dim=0).tolist()

    def channel_means(t):  # NCHW
        return t[0].reshape(t.shape[1], -1).mean(dim=1).tolist()

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        w = lambda s: f.write(s + "\n")  # noqa: E731
        w("# qorvix stable-diffusion reference fixture v1")
        w("# model:   %s" % args.model)
        w("# source:  diffusers %s, torch fp32, CPU - NOT qorvix" % args.sampler)
        w("# capture: scripts/capture_sd_reference.py")
        w("#")
        w("# Regenerate rather than hand-edit; a fixture produced by qorvix itself would make")
        w("# sd-check self-referential and prove nothing.")
        w("model %s" % args.model)
        w("size %d %d" % (args.size, args.size))
        w("steps %d" % args.steps)
        w("guidance %.6f" % args.guidance)
        w("sampler %s" % args.sampler)
        w("clip_skip %d" % args.clip_skip)
        w("prompt %s" % args.prompt)
        w("negative %s" % args.negative)
        w("tokens %s" % " ".join(str(i) for i in ids))
        w("neg_tokens %s" % " ".join(str(i) for i in neg_ids))
        w("timesteps %s" % " ".join(str(t) for t in timesteps))
        for i in probe_idx:
            w("alpha_probe %d %.9f" % (i, alphas[i]))
        w("cond_row0 %s" % fmt(cond[0, 0].tolist()))
        w("cond_dim_means %s" % fmt(dim_means(cond)))
        w("uncond_row0 %s" % fmt(uncond[0, 0].tolist()))
        w("latent_shape %d %d %d" % (latents.shape[1], latents.shape[2], latents.shape[3]))
        # NCHW -> the position-major order qorvix reads it back in, so the fixture is not also a
        # transpose puzzle.
        w("latent %s" % fmt(latents[0].permute(1, 2, 0).flatten().tolist()))
        w("unet_t %d" % int(t0))
        w("unet_row0 %s" % fmt(unet_out[0, :, 0, 0].tolist()))
        w("unet_channel_means %s" % fmt(channel_means(unet_out)))
        w("final_latent %s" % fmt(lat[0].permute(1, 2, 0).flatten().tolist()))
        w("image_shape %d %d" % (image.shape[2], image.shape[3]))
        w("image_row0 %s" % fmt(image[0, :, 0, 0].tolist()))
        w("image_channel_means %s" % fmt(channel_means(image)))

    print("wrote %s" % args.out, file=sys.stderr)
    print("  %d steps at %dx%d, sampler %s, guidance %.2f, %d prompt tokens"
          % (args.steps, args.size, args.size, args.sampler, args.guidance, len(ids)), file=sys.stderr)
    print("  timesteps %s" % timesteps, file=sys.stderr)


if __name__ == "__main__":
    main()
