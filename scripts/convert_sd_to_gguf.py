#!/usr/bin/env python3
"""Convert a HuggingFace diffusers Stable Diffusion checkpoint to GGUF for `qorvix draw`.

    pip install diffusers transformers torch numpy
    python scripts/convert_sd_to_gguf.py stable-diffusion-v1-5/stable-diffusion-v1-5 \
        models/sd15-f16.gguf --outtype f16
    python scripts/convert_sd_to_gguf.py hf-internal-testing/tiny-stable-diffusion-torch \
        models/sd-tiny-f32.gguf

WHY A CONVERTER AND NOT A READER. The same answer Phase 11b-3b gave for Whisper, for a second
reason on top of it. Diffusers ships safetensors, not GGUF; the GGUF files that do exist for
Stable Diffusion come from stable-diffusion.cpp and carry the *original CompVis* tensor names
(`model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight`), which is a
third naming scheme neither this repo nor diffusers uses. Reading those would mean owning a
name-translation table anyway - so the table lives here, in Python, next to the checkpoint whose
names it translates, and the C++ loader reads names that look like the rest of this repo.

THREE MODELS, ONE FILE. A Stable Diffusion pipeline is three networks - a CLIP text encoder, a
UNet, and a VAE - plus a scheduler that is pure arithmetic over constants. They ship as three
safetensors and a pile of JSON. They convert into ONE GGUF with three tensor-name prefixes
(`te.`, `unet.`, `vae.`), because at run time they are one artifact: a UNet paired with the wrong
text encoder produces a fluent-looking image of the wrong thing, and nothing errors. Keeping them
in one file makes that pairing unbreakable rather than a deployment convention.

CONVOLUTIONS ARE WRITTEN AS MATRICES. Every conv weight [out, in, kh, kw] is written as a 2-D
[out, kh*kw*in] tensor. That lets the C++ side load conv weights through the SAME
`detail::loadMat` + `qmatmulN` path as every linear layer in this repo, with im2col supplying the
patch vectors - a conv-specific weight struct and a conv-specific kernel would have been a second
weight path to keep correct.

Note the axis order: [out, KH, KW, IN], not PyTorch's [out, IN, KH, KW]. The permutation is the
one thing here that is not free, and it is done for the reader rather than the writer. Activations
in this module are position-major (`[h*w, c]`), so an im2col patch is nine runs of `in_channels`
contiguous floats; with PyTorch's axis order the same patch would be `in_channels * 9` individually
strided loads. The kernel extent is not stored per tensor because it is structural - 3x3 for the
resnet/sampler convs, 1x1 for the shortcuts and projections - and the loader asserts the element
count that implies, so a mismatch is a load error rather than a silently reshaped tensor.

WHAT IS REFUSED. SDXL and Flux are refused by name rather than half-converted:

  * SDXL runs TWO text encoders (CLIP-L plus OpenCLIP-bigG) whose outputs are concatenated to
    2048 dims, and adds `text_time` conditioning (the bigG pooled vector plus six micro-condition
    scalars) into the timestep embedding. Neither is a config knob on this file format, and a
    converter that dropped them would produce a UNet whose cross-attention is the right shape and
    the wrong content.
  * Flux is not a UNet at all - it is a rectified-flow DiT with a T5 encoder alongside CLIP.

Both are named in the error with what they would need, which is the shape of the next phase.

QUANTIZATION. f32 and f16 only, and the reason is structural rather than "not done yet": the
block-quantized kernels need a row length that is a multiple of the block size (32 for Q8_0, 256
for Q4_K/Q6_K), and a conv row is `in_channels * 9` - 36 for the first conv, 2880 for a 320-channel
one. Some rows qualify and some do not, so a quantized SD file would be a per-tensor mixture, which
is a real feature and not a flag. f16 halves the file for these checkpoints at no measurable cost
(the UNet was trained in fp16).
"""

import argparse
import json
import struct
import sys

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
    """Minimal GGUF v3 writer: typed KV metadata, then 32-byte-aligned tensor data.

    Deliberately a copy of the one in convert_whisper_to_gguf.py rather than a shared import: a
    dev-only script that a user runs with `python scripts/...` should not need this repo on its
    import path, and the writer is 60 lines of struct packing that the format pins in place.
    """

    def __init__(self):
        self.kv = []       # (key, type_tag, payload_bytes)
        self.tensors = []  # (name, ggml_type, ne, nbytes, produce)
        self.names = set()

    def _kv(self, key, tag, payload):
        self.kv.append((key, tag, payload))

    def u32(self, key, v):
        self._kv(key, T_UINT32, struct.pack("<I", int(v)))

    def i32(self, key, v):
        self._kv(key, T_INT32, struct.pack("<i", int(v)))

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

    def tensor(self, name, produce, shape, ggml_type):
        """Registers a tensor by SHAPE and a `produce()` that returns its numpy array later.

        Lazy on purpose, and the reason is a real limit rather than tidiness: a 1B-parameter
        pipeline is ~2 GiB of f16 output, and accumulating every tensor's bytes before writing
        holds that entire second copy in memory on top of the loaded model. Producing each array
        at write time - including the conv permutation, which is itself a full-size copy - keeps
        the transient to one tensor.
        """
        if name in self.names:
            raise SystemExit("internal error: tensor '%s' written twice" % name)
        self.names.add(name)
        count = 1
        for dim in shape:
            count *= int(dim)
        nbytes = count * (2 if ggml_type == GGML_F16 else 4)
        # GGUF stores `ne` with the fastest-varying dimension first, the reverse of a row-major
        # numpy shape. The BYTES are identical either way - only the declared shape differs.
        self.tensors.append((name, ggml_type, list(reversed(list(shape))), nbytes, produce))

    def write(self, path):
        head = bytearray()
        head += GGUF_MAGIC
        head += struct.pack("<IQQ", GGUF_VERSION, len(self.tensors), len(self.kv))
        for key, tag, payload in self.kv:
            head += _str(key) + struct.pack("<I", tag) + payload

        # Offsets are computed from the declared sizes, so the table can be written before a single
        # tensor has been materialized.
        offsets, cursor = [], 0
        for _, _, _, nbytes, _ in self.tensors:
            offsets.append(cursor)
            cursor += _pad(nbytes)
        for (name, ttype, ne, _, _), off in zip(self.tensors, offsets):
            head += _str(name) + struct.pack("<I", len(ne))
            for dim in ne:
                head += struct.pack("<Q", dim)
            head += struct.pack("<IQ", ttype, off)
        head += b"\0" * (_pad(len(head)) - len(head))

        with open(path, "wb") as f:
            f.write(bytes(head))
            for name, _, _, nbytes, produce in self.tensors:
                data = produce().tobytes()
                if len(data) != nbytes:
                    raise SystemExit("tensor %s produced %d bytes, declared %d - the header would "
                                     "point at the wrong offsets" % (name, len(data), nbytes))
                f.write(data)
                f.write(b"\0" * (_pad(nbytes) - nbytes))
                del data
        return len(head) + cursor


# --------------------------------------------------------------------------------------------
# Tensor plumbing


class Emitter:
    """Pulls named tensors out of a state dict and writes them under this repo's names.

    Every read is `take()`d, so `unused()` at the end reports anything the table missed. A
    checkpoint carrying a tensor nobody claimed is a converter bug, not a curiosity - it is how a
    forgotten `conv_shortcut` or a second attention head layout gets noticed at conversion time
    instead of as a subtly wrong image.
    """

    def __init__(self, writer, outtype):
        import numpy as np

        self.np = np
        self.w = writer
        self.outtype = outtype
        self.sources = {}   # prefix -> (state_dict, set_of_taken_keys)

    def add_source(self, prefix, state_dict):
        self.sources[prefix] = (state_dict, set())

    def _get(self, prefix, key):
        sd, taken = self.sources[prefix]
        if key not in sd:
            return None
        taken.add(key)
        return sd[key]

    def has(self, prefix, key):
        return key in self.sources[prefix][0]

    def unused(self):
        out = []
        for prefix, (sd, taken) in self.sources.items():
            for k in sd:
                if k not in taken:
                    out.append(prefix + "/" + k)
        return out

    def _write(self, name, tensor, shape, force_f32, permute=None):
        count = 1
        for dim in shape:
            count *= int(dim)
        if int(tensor.numel()) != count:
            raise SystemExit("tensor %s has %d elements, expected %s"
                             % (name, int(tensor.numel()), shape))
        # Norm weights, biases and the position table stay F32. They are a rounding error of the
        # file size, they are consumed elementwise rather than through a matmul, and f16 rounding
        # of a GroupNorm scale is the kind of drift that shows up as a tint rather than as an
        # error. Same rule the BERT/CLIP/Whisper loaders already assume when they call loadVec.
        ttype = GGML_F16 if (self.outtype == "f16" and not force_f32) else GGML_F32
        np = self.np

        def produce(t=tensor, shape=shape, ttype=ttype, permute=permute):
            # Everything here happens at WRITE time, one tensor at a time - including the conv
            # permutation, which is a full-size copy of the largest tensors in the file.
            t = t.detach().to("cpu")
            if permute is not None:
                t = t.permute(*permute).contiguous()
            t = t.half() if ttype == GGML_F16 else t.float()
            return np.ascontiguousarray(t.numpy()).reshape(shape)

        self.w.tensor(name, produce, shape, ttype)

    def mat(self, dst, prefix, src, shape=None):
        """A 2-D matmul weight [out, in]. Conv weights arrive as [out, in, kh, kw] and are
        flattened here - see the module docstring."""
        t = self._get(prefix, src)
        if t is None:
            raise SystemExit("missing tensor %s/%s" % (prefix, src))
        if shape is None:
            shape = (t.shape[0], int(t.numel()) // t.shape[0])
        self._write(dst, t, shape, force_f32=False)

    def vec(self, dst, prefix, src, n=None):
        """A 1-D bias / norm scale. Always F32."""
        t = self._get(prefix, src)
        if t is None:
            raise SystemExit("missing tensor %s/%s" % (prefix, src))
        self._write(dst, t, (int(t.numel()),) if n is None else (n,), force_f32=True)

    def conv(self, dst, prefix, src, kernel):
        """A conv weight + bias pair, permuted [out, in, kh, kw] -> [out, kh, kw, in].

        `kernel` is asserted, not read: a 3x3 where a 1x1 was expected is a structural surprise
        worth failing on rather than reshaping around."""
        t = self._get(prefix, src + ".weight")
        if t is None:
            raise SystemExit("missing tensor %s/%s.weight" % (prefix, src))
        if tuple(t.shape[2:]) != (kernel, kernel):
            raise SystemExit("%s/%s.weight has kernel %s, expected %dx%d"
                             % (prefix, src, tuple(t.shape[2:]), kernel, kernel))
        self._write(dst + ".weight", t, (int(t.shape[0]), int(t.numel()) // int(t.shape[0])),
                    force_f32=False, permute=(0, 2, 3, 1))
        self.vec(dst + ".bias", prefix, src + ".bias")

    def linear(self, dst, prefix, src, bias=True):
        self.mat(dst + ".weight", prefix, src + ".weight")
        if bias:
            self.vec(dst + ".bias", prefix, src + ".bias")
        elif self.has(prefix, src + ".bias"):
            raise SystemExit("%s/%s has a bias but this architecture says it should not" % (prefix, src))

    def norm(self, dst, prefix, src):
        self.vec(dst + ".weight", prefix, src + ".weight")
        self.vec(dst + ".bias", prefix, src + ".bias")


# --------------------------------------------------------------------------------------------
# Sub-network conversions


def emit_text_encoder(e, cfg, prefix="te"):
    """CLIP text encoder: pre-norm blocks, causal attention, learned positions.

    Structurally the CLIP *vision* tower Phase 11b-1 already runs, with token+position embeddings
    instead of patch embeddings and a causal mask instead of a bidirectional one. The FFN here is
    an honest up/down pair (`fc1` expands, `fc2` contracts) - unlike the clip mmproj format, where
    `ffn_down` is the expansion.
    """
    e.mat(prefix + ".token_embd.weight", "text", "embeddings.token_embedding.weight")
    e.vec(prefix + ".position_embd.weight", "text", "embeddings.position_embedding.weight",
          n=cfg["ctx"] * cfg["d"])
    for i in range(cfg["layers"]):
        src = "encoder.layers.%d." % i
        dst = "%s.blk.%d." % (prefix, i)
        e.norm(dst + "ln1", "text", src + "layer_norm1")
        e.linear(dst + "attn_q", "text", src + "self_attn.q_proj")
        e.linear(dst + "attn_k", "text", src + "self_attn.k_proj")
        e.linear(dst + "attn_v", "text", src + "self_attn.v_proj")
        e.linear(dst + "attn_out", "text", src + "self_attn.out_proj")
        e.norm(dst + "ln2", "text", src + "layer_norm2")
        e.linear(dst + "ffn_up", "text", src + "mlp.fc1")
        e.linear(dst + "ffn_down", "text", src + "mlp.fc2")
    e.norm(prefix + ".ln_final", "text", "final_layer_norm")


def emit_resnet(e, dst, prefix, src, has_time_emb=True):
    e.norm(dst + ".norm1", prefix, src + "norm1")
    e.conv(dst + ".conv1", prefix, src + "conv1", 3)
    if has_time_emb:
        e.linear(dst + ".time_emb", prefix, src + "time_emb_proj")
    e.norm(dst + ".norm2", prefix, src + "norm2")
    e.conv(dst + ".conv2", prefix, src + "conv2", 3)
    # The 1x1 shortcut exists only where in_channels != out_channels. Optional by architecture,
    # so its absence is normal and its presence is loaded by name - never synthesized.
    if e.has(prefix, src + "conv_shortcut.weight"):
        e.conv(dst + ".skip", prefix, src + "conv_shortcut", 1)


def emit_spatial_transformer(e, dst, prefix, src, depth, linear_proj):
    """Transformer2DModel: GroupNorm, project in, `depth` BasicTransformerBlocks, project out.

    `proj_in`/`proj_out` are a 1x1 conv when use_linear_projection is false (SD 1.x) and an
    nn.Linear when it is true (SD 2.x). Those are the same matrix and the same math - a 1x1 conv
    IS a per-position linear - so both flatten to the identical [out, in] tensor here and the C++
    side needs no flag to tell them apart.
    """
    e.norm(dst + ".norm", prefix, src + "norm")
    if linear_proj:
        e.linear(dst + ".proj_in", prefix, src + "proj_in")
        proj_out_linear = True
    else:
        e.conv(dst + ".proj_in", prefix, src + "proj_in", 1)
        proj_out_linear = False
    for d in range(depth):
        b = src + "transformer_blocks.%d." % d
        bd = dst + ".tf.%d." % d
        e.norm(bd + "ln1", prefix, b + "norm1")
        # attn1/attn2 q/k/v carry NO bias while to_out does. Same asymmetry Whisper's k_proj has,
        # and the same handling: the loader has no bias field for them, so it cannot invent one.
        e.linear(bd + "attn1_q", prefix, b + "attn1.to_q", bias=False)
        e.linear(bd + "attn1_k", prefix, b + "attn1.to_k", bias=False)
        e.linear(bd + "attn1_v", prefix, b + "attn1.to_v", bias=False)
        e.linear(bd + "attn1_out", prefix, b + "attn1.to_out.0")
        e.norm(bd + "ln2", prefix, b + "norm2")
        e.linear(bd + "attn2_q", prefix, b + "attn2.to_q", bias=False)
        e.linear(bd + "attn2_k", prefix, b + "attn2.to_k", bias=False)
        e.linear(bd + "attn2_v", prefix, b + "attn2.to_v", bias=False)
        e.linear(bd + "attn2_out", prefix, b + "attn2.to_out.0")
        e.norm(bd + "ln3", prefix, b + "norm3")
        # GEGLU: one [2*inner, d] matrix whose halves are (value, gate) in that order.
        e.linear(bd + "ffn_geglu", prefix, b + "ff.net.0.proj")
        e.linear(bd + "ffn_out", prefix, b + "ff.net.2")
    if proj_out_linear:
        e.linear(dst + ".proj_out", prefix, src + "proj_out")
    else:
        e.conv(dst + ".proj_out", prefix, src + "proj_out", 1)


def emit_vae_attention(e, dst, prefix, src):
    """The VAE's single-head self-attention over every spatial position. q/k/v/out all have
    biases here, the opposite of the UNet's cross-attention blocks."""
    e.norm(dst + ".norm", prefix, src + "group_norm")
    e.linear(dst + ".q", prefix, src + "to_q")
    e.linear(dst + ".k", prefix, src + "to_k")
    e.linear(dst + ".v", prefix, src + "to_v")
    e.linear(dst + ".out", prefix, src + "to_out.0")


def emit_unet(e, unet_cfg):
    n_blocks = len(unet_cfg["block_out_channels"])
    layers = unet_cfg["layers_per_block"]
    depths = unet_cfg["transformer_depth"]
    linear_proj = bool(unet_cfg["use_linear_projection"])

    e.conv("unet.conv_in", "unet", "conv_in", 3)
    e.linear("unet.time_mlp.0", "unet", "time_embedding.linear_1")
    e.linear("unet.time_mlp.1", "unet", "time_embedding.linear_2")

    for i in range(n_blocks):
        for j in range(layers):
            emit_resnet(e, "unet.down.%d.resnet.%d" % (i, j), "unet",
                        "down_blocks.%d.resnets.%d." % (i, j))
            if depths[i] > 0:
                emit_spatial_transformer(e, "unet.down.%d.attn.%d" % (i, j), "unet",
                                         "down_blocks.%d.attentions.%d." % (i, j),
                                         depths[i], linear_proj)
        # Every down block except the last carries a stride-2 conv.
        if i < n_blocks - 1:
            e.conv("unet.down.%d.downsample" % i, "unet",
                   "down_blocks.%d.downsamplers.0.conv" % i, 3)

    emit_resnet(e, "unet.mid.resnet.0", "unet", "mid_block.resnets.0.")
    emit_spatial_transformer(e, "unet.mid.attn", "unet", "mid_block.attentions.0.",
                             unet_cfg["mid_transformer_depth"], linear_proj)
    emit_resnet(e, "unet.mid.resnet.1", "unet", "mid_block.resnets.1.")

    rev_depths = list(reversed(depths))
    for i in range(n_blocks):
        # An up block has layers_per_block + 1 resnets: one per down-block residual, plus one for
        # the residual the block BEFORE it pushed. Getting this count wrong shifts every skip
        # connection by one and still runs.
        for j in range(layers + 1):
            emit_resnet(e, "unet.up.%d.resnet.%d" % (i, j), "unet",
                        "up_blocks.%d.resnets.%d." % (i, j))
            if rev_depths[i] > 0:
                emit_spatial_transformer(e, "unet.up.%d.attn.%d" % (i, j), "unet",
                                         "up_blocks.%d.attentions.%d." % (i, j),
                                         rev_depths[i], linear_proj)
        if i < n_blocks - 1:
            e.conv("unet.up.%d.upsample" % i, "unet", "up_blocks.%d.upsamplers.0.conv" % i, 3)

    e.norm("unet.norm_out", "unet", "conv_norm_out")
    e.conv("unet.conv_out", "unet", "conv_out", 3)


def emit_vae_decoder(e, vae_cfg):
    n_blocks = len(vae_cfg["block_out_channels"])
    layers = vae_cfg["layers_per_block"] + 1

    if e.has("vae", "post_quant_conv.weight"):
        e.conv("vae.post_quant_conv", "vae", "post_quant_conv", 1)
    e.conv("vae.dec.conv_in", "vae", "decoder.conv_in", 3)
    emit_resnet(e, "vae.dec.mid.resnet.0", "vae", "decoder.mid_block.resnets.0.",
                has_time_emb=False)
    emit_vae_attention(e, "vae.dec.mid.attn", "vae", "decoder.mid_block.attentions.0.")
    emit_resnet(e, "vae.dec.mid.resnet.1", "vae", "decoder.mid_block.resnets.1.",
                has_time_emb=False)
    for i in range(n_blocks):
        for j in range(layers):
            emit_resnet(e, "vae.dec.up.%d.resnet.%d" % (i, j), "vae",
                        "decoder.up_blocks.%d.resnets.%d." % (i, j), has_time_emb=False)
        if i < n_blocks - 1:
            e.conv("vae.dec.up.%d.upsample" % i, "vae",
                   "decoder.up_blocks.%d.upsamplers.0.conv" % i, 3)
    e.norm("vae.dec.norm_out", "vae", "decoder.conv_norm_out")
    e.conv("vae.dec.conv_out", "vae", "decoder.conv_out", 3)


# --------------------------------------------------------------------------------------------
# Config


def resolve_head_counts(unet_config, n_blocks):
    """Diffusers' `attention_head_dim` is the NUMBER OF HEADS, not the head width.

    This is a documented wart in diffusers (`num_attention_heads` was introduced later and is
    None on every released SD checkpoint, at which point the code falls back to
    `attention_head_dim`). SD 1.5 says 8 and means 8 heads of width 40/80/160; SD 2.1 says
    [5,10,20,20] and means head width 64 throughout. Resolving it HERE, against the config that
    carries the ambiguity, means the GGUF states head counts unambiguously and the C++ side never
    has to know the wart existed.
    """
    heads = unet_config.get("num_attention_heads", None)
    if heads is None:
        heads = unet_config.get("attention_head_dim", 8)
    if not isinstance(heads, (list, tuple)):
        heads = [heads] * n_blocks
    if len(heads) != n_blocks:
        raise SystemExit("attention head config has %d entries for %d blocks" % (len(heads), n_blocks))
    return [int(h) for h in heads]


def resolve_transformer_depth(unet_config, down_block_types, n_blocks):
    depth = unet_config.get("transformer_layers_per_block", 1)
    if not isinstance(depth, (list, tuple)):
        depth = [depth] * n_blocks
    # A block with no cross-attention has depth 0 - that is what distinguishes DownBlock2D from
    # CrossAttnDownBlock2D, and it is read off the block type rather than guessed from position.
    return [int(d) if "CrossAttn" in t else 0 for d, t in zip(depth, down_block_types)]


def main():
    ap = argparse.ArgumentParser(description="Convert a diffusers Stable Diffusion checkpoint to GGUF")
    ap.add_argument("model", help="HF repo id or local diffusers directory")
    ap.add_argument("out", help="output .gguf path")
    ap.add_argument("--outtype", choices=["f32", "f16"], default="f32")
    ap.add_argument("--name", default=None, help="general.name (defaults to the repo id)")
    args = ap.parse_args()

    try:
        import numpy as np  # noqa: F401
        import torch
        from diffusers import StableDiffusionPipeline
        from transformers import CLIPTokenizer
    except ImportError as exc:
        raise SystemExit("missing dependency: %s\n  pip install diffusers transformers torch numpy" % exc)

    print("loading %s ..." % args.model, file=sys.stderr)
    # Load at the OUTPUT precision. For --outtype f16 that halves resident memory on a real
    # checkpoint (2.6 GiB rather than 5.2 GiB for a 1B-parameter pipeline) and loses nothing: the
    # cast happens either way, and doing it at load time means it never happens twice.
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, safety_checker=None, requires_safety_checker=False,
        torch_dtype=torch.float16 if args.outtype == "f16" else torch.float32)

    unet_config = dict(pipe.unet.config)
    vae_config = dict(pipe.vae.config)
    text_config = pipe.text_encoder.config
    sched_config = dict(pipe.scheduler.config)

    # Refuse the architectures this file format cannot describe, by name and with the reason.
    if unet_config.get("addition_embed_type") is not None:
        raise SystemExit(
            "this is an SDXL-family UNet (addition_embed_type=%r).\n"
            "SDXL needs two text encoders concatenated to %d cross-attention dims plus the\n"
            "text_time conditioning vector added into the timestep embedding; neither is\n"
            "expressible in this file format, and converting without them would produce a UNet\n"
            "with the right shapes and the wrong conditioning."
            % (unet_config["addition_embed_type"], unet_config.get("cross_attention_dim", 2048)))
    if unet_config.get("class_embed_type") is not None:
        raise SystemExit("class-conditioned UNets (class_embed_type=%r) are not supported"
                         % unet_config["class_embed_type"])
    if unet_config.get("encoder_hid_dim") is not None:
        raise SystemExit("UNets with an encoder_hid_proj are not supported")
    if unet_config.get("time_embedding_type", "positional") != "positional":
        raise SystemExit("only positional timestep embeddings are supported, not %r"
                         % unet_config["time_embedding_type"])
    if unet_config.get("resnet_time_scale_shift", "default") != "default":
        raise SystemExit("only the default resnet time conditioning is supported, not %r"
                         % unet_config["resnet_time_scale_shift"])
    if unet_config.get("dual_cross_attention"):
        raise SystemExit("dual cross attention is not supported")
    if unet_config.get("only_cross_attention"):
        raise SystemExit("only_cross_attention UNets are not supported")
    if text_config.model_type != "clip_text_model":
        raise SystemExit("expected a CLIP text encoder, found %r" % text_config.model_type)
    if not vae_config.get("mid_block_add_attention", True):
        raise SystemExit("VAEs without mid-block attention are not supported")
    if vae_config.get("shift_factor") is not None:
        raise SystemExit("VAEs with a latent shift_factor are not supported")

    block_out = [int(c) for c in unet_config["block_out_channels"]]
    n_blocks = len(block_out)
    heads = resolve_head_counts(unet_config, n_blocks)
    depths = resolve_transformer_depth(unet_config, unet_config["down_block_types"], n_blocks)
    raw_depth = unet_config.get("transformer_layers_per_block", 1)
    mid_depth = int(raw_depth[-1] if isinstance(raw_depth, (list, tuple)) else raw_depth)
    time_embed_dim = int(unet_config.get("time_embedding_dim") or block_out[0] * 4)
    unet_cfg = {
        "block_out_channels": block_out,
        "layers_per_block": int(unet_config["layers_per_block"]),
        "transformer_depth": depths,
        "use_linear_projection": bool(unet_config.get("use_linear_projection", False)),
        "mid_transformer_depth": mid_depth,
    }

    w = GgufWriter()
    w.string("general.architecture", "sd")
    w.string("general.name", args.name or args.model)
    w.u32("general.file_type", 1 if args.outtype == "f16" else 0)

    # Pipeline-level constants.
    w.f32("sd.scale_factor", vae_config.get("scaling_factor", 0.18215))
    w.u32("sd.latent_channels", vae_config["latent_channels"])
    w.u32("sd.vae_scale", 2 ** (len(vae_config["block_out_channels"]) - 1))
    w.u32("sd.sample_size", unet_config.get("sample_size", 64))
    w.string("sd.prediction_type", sched_config.get("prediction_type", "epsilon"))

    # Scheduler constants. Every sampler in the C++ side is arithmetic over these; keeping them in
    # the file means a v-prediction or a trailing-spaced checkpoint samples correctly without a
    # per-model table in the source.
    w.u32("sd.scheduler.train_timesteps", sched_config.get("num_train_timesteps", 1000))
    w.f32("sd.scheduler.beta_start", sched_config.get("beta_start", 0.00085))
    w.f32("sd.scheduler.beta_end", sched_config.get("beta_end", 0.012))
    w.string("sd.scheduler.beta_schedule", sched_config.get("beta_schedule", "scaled_linear"))
    w.string("sd.scheduler.timestep_spacing", sched_config.get("timestep_spacing", "leading"))
    w.u32("sd.scheduler.steps_offset", sched_config.get("steps_offset", 1))
    w.flag("sd.scheduler.set_alpha_to_one", bool(sched_config.get("set_alpha_to_one", False)))
    if sched_config.get("trained_betas") is not None:
        raise SystemExit("checkpoints with explicit trained_betas are not supported")

    # Text encoder.
    w.u32("sd.text.embedding_length", text_config.hidden_size)
    w.u32("sd.text.block_count", text_config.num_hidden_layers)
    w.u32("sd.text.attention.head_count", text_config.num_attention_heads)
    w.u32("sd.text.feed_forward_length", text_config.intermediate_size)
    w.u32("sd.text.context_length", text_config.max_position_embeddings)
    w.u32("sd.text.vocab_size", text_config.vocab_size)
    w.f32("sd.text.layer_norm_epsilon", text_config.layer_norm_eps)
    # quick_gelu for the SD 1.x CLIP-L encoder, gelu for the SD 2.x OpenCLIP one. A silent
    # substitution here shifts every conditioning vector without erroring - the same trap the CLIP
    # vision tower documents at clip.use_gelu.
    w.string("sd.text.activation", text_config.hidden_act)

    # UNet.
    w.u32("sd.unet.in_channels", unet_config["in_channels"])
    w.u32("sd.unet.out_channels", unet_config["out_channels"])
    w.int32_array("sd.unet.channels", block_out)
    w.int32_array("sd.unet.head_counts", heads)
    w.int32_array("sd.unet.transformer_depth", depths)
    w.u32("sd.unet.layers_per_block", unet_cfg["layers_per_block"])
    # The mid block always has cross-attention, and its depth is the LAST entry of
    # transformer_layers_per_block BEFORE the block types zero it out - on SD 1.x the last down
    # block is a plain DownBlock2D, so `sd.unet.transformer_depth` ends in 0 while the mid block
    # is still depth 1. Deriving one from the other would be wrong on exactly the models that
    # matter.
    w.u32("sd.unet.mid_transformer_depth", mid_depth)
    # time_embedding.linear_1 maps block_out_channels[0] -> this. Written rather than assumed to
    # be 4x the first channel count, because `time_embedding_dim` is a config knob.
    w.u32("sd.unet.time_embed_dim", time_embed_dim)
    w.u32("sd.unet.cross_attention_dim", unet_config["cross_attention_dim"])
    w.u32("sd.unet.norm_groups", unet_config["norm_num_groups"])
    w.f32("sd.unet.norm_epsilon", unet_config["norm_eps"])
    # The spatial transformer's own GroupNorm is hardcoded to 1e-6 in diffusers while the resnets
    # use norm_eps (1e-5). Two different epsilons in one network is exactly the sort of detail
    # that is invisible until an image is subtly wrong, so both are written out.
    w.f32("sd.unet.transformer_norm_epsilon", 1e-6)
    w.flag("sd.unet.flip_sin_to_cos", bool(unet_config.get("flip_sin_to_cos", True)))
    w.f32("sd.unet.freq_shift", float(unet_config.get("freq_shift", 0)))
    w.flag("sd.unet.linear_projection", unet_cfg["use_linear_projection"])

    # VAE decoder.
    w.int32_array("sd.vae.channels", [int(c) for c in vae_config["block_out_channels"]])
    w.u32("sd.vae.layers_per_block", vae_config["layers_per_block"])
    w.u32("sd.vae.norm_groups", vae_config["norm_num_groups"])
    w.f32("sd.vae.norm_epsilon", 1e-6)

    # Tokenizer. CLIP's BPE is not the byte-level GPT-2 BPE this repo already carries - words end
    # in a `</w>` marker and the pretokenizer lowercases - so the vocabulary and merge table are
    # written under the usual keys with a model name that says which algorithm reads them.
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    vocab = tokenizer.get_vocab()
    tokens = [None] * (max(vocab.values()) + 1)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    if any(t is None for t in tokens):
        raise SystemExit("tokenizer vocabulary has holes in its id space")
    # transformers 5 dropped the pure-Python CLIPTokenizer, so the merge table now lives inside
    # the Rust backend rather than in a `bpe_ranks` dict. Its serialized form has changed shape
    # across tokenizers releases (a list of "a b" strings, then a list of ["a","b"] pairs), so
    # both are accepted and anything else is a hard error rather than a guess.
    backend = json.loads(tokenizer.backend_tokenizer.to_str())["model"]
    if backend.get("end_of_word_suffix") != "</w>":
        raise SystemExit("expected a CLIP BPE with a '</w>' end-of-word suffix, found %r"
                         % backend.get("end_of_word_suffix"))
    merges = []
    for m in backend["merges"]:
        if isinstance(m, str):
            merges.append(m)
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            merges.append(m[0] + " " + m[1])
        else:
            raise SystemExit("unrecognised merge entry %r" % (m,))
    w.string("tokenizer.ggml.model", "clip")
    w.string_array("tokenizer.ggml.tokens", tokens)
    w.string_array("tokenizer.ggml.merges", merges)
    # The TOKENIZER's ids, not the text encoder config's. They usually agree; on this repo's own
    # gate fixture they do not (the tiny test checkpoint declares eos_token_id 2 while its
    # tokenizer emits 1), and the ids that must be right are the ones the prompt is built from.
    w.i32("tokenizer.ggml.bos_token_id", tokenizer.bos_token_id)
    w.i32("tokenizer.ggml.eos_token_id", tokenizer.eos_token_id)
    # CLIP pads a prompt out to the full 77 positions with <|endoftext|> and passes NO attention
    # mask, so the padding is attended to and is part of the conditioning. Dropping it - or
    # masking it - changes every image.
    w.i32("tokenizer.ggml.padding_token_id", tokenizer.pad_token_id)

    e = Emitter(w, args.outtype)
    e.add_source("text", pipe.text_encoder.state_dict())
    e.add_source("unet", pipe.unet.state_dict())
    e.add_source("vae", pipe.vae.state_dict())

    emit_text_encoder(e, {"layers": text_config.num_hidden_layers,
                          "d": text_config.hidden_size,
                          "ctx": text_config.max_position_embeddings})
    emit_unet(e, unet_cfg)
    emit_vae_decoder(e, vae_config)

    # The VAE ENCODER is deliberately left behind: text-to-image never runs it, and shipping half
    # a gigabyte of unread weights in every file to reserve img2img for later is a cost paid on
    # every load. It is named in `unused` below rather than silently dropped.
    leftovers = [k for k in e.unused()
                 if not k.startswith("vae/encoder.") and k != "vae/quant_conv.weight"
                 and k != "vae/quant_conv.bias"]
    if leftovers:
        raise SystemExit("checkpoint has %d tensors this converter did not claim, starting with:\n  %s"
                         % (len(leftovers), "\n  ".join(sorted(leftovers)[:10])))

    size = w.write(args.out)
    print("wrote %s: %d tensors, %.1f MiB (%s)"
          % (args.out, len(w.tensors), size / (1 << 20), args.outtype), file=sys.stderr)
    print("  text encoder: %d layers x %d heads, d=%d, ctx=%d, vocab=%d, act=%s"
          % (text_config.num_hidden_layers, text_config.num_attention_heads,
             text_config.hidden_size, text_config.max_position_embeddings,
             text_config.vocab_size, text_config.hidden_act), file=sys.stderr)
    print("  unet:         channels=%s heads=%s depth=%s cross=%d"
          % (block_out, heads, depths, unet_config["cross_attention_dim"]), file=sys.stderr)
    print("  vae decoder:  channels=%s scale=%s latent=%d"
          % (list(vae_config["block_out_channels"]), vae_config.get("scaling_factor", 0.18215),
             vae_config["latent_channels"]), file=sys.stderr)
    print(json.dumps({"tensors": len(w.tensors), "bytes": size}), file=sys.stderr)


if __name__ == "__main__":
    main()
