#!/usr/bin/env python3
"""Writes a synthetic SD 1.5 checkpoint for the tests that need a real model.

  python3 test/make_sd15_synth.py OUT.safetensors

The file has the standard LDM tensor names and shapes of SD 1.5 (UNet, VAE and CLIP text encoder) in F16, about
2 GiB: weights are random with a fixed seed, biases are zero and norms are one, so every run writes the same bytes.
It exercises loading and the whole generation pipeline on any backend; the images it makes are noise.
server_cancel_e2e.py, server_guardian_e2e.py and agenti_guardiano_test.py --server use it, and so does the
workflow .github/workflows/fork-tests.yml. It needs numpy.
"""
import json
import os
import struct
import sys

import numpy as np

tensors = []  # (name, shape)


def add(name, *shape):
    tensors.append((name, tuple(shape)))


def unet_res(prefix, cin, cout):
    add(prefix + "in_layers.0.weight", cin)
    add(prefix + "in_layers.0.bias", cin)
    add(prefix + "in_layers.2.weight", cout, cin, 3, 3)
    add(prefix + "in_layers.2.bias", cout)
    add(prefix + "emb_layers.1.weight", cout, 1280)
    add(prefix + "emb_layers.1.bias", cout)
    add(prefix + "out_layers.0.weight", cout)
    add(prefix + "out_layers.0.bias", cout)
    add(prefix + "out_layers.3.weight", cout, cout, 3, 3)
    add(prefix + "out_layers.3.bias", cout)
    if cin != cout:
        add(prefix + "skip_connection.weight", cout, cin, 1, 1)
        add(prefix + "skip_connection.bias", cout)


def unet_transformer(prefix, c):
    add(prefix + "norm.weight", c)
    add(prefix + "norm.bias", c)
    add(prefix + "proj_in.weight", c, c, 1, 1)
    add(prefix + "proj_in.bias", c)
    b = prefix + "transformer_blocks.0."
    for attn, kv in (("attn1", c), ("attn2", 768)):
        add(b + attn + ".to_q.weight", c, c)
        add(b + attn + ".to_k.weight", c, kv)
        add(b + attn + ".to_v.weight", c, kv)
        add(b + attn + ".to_out.0.weight", c, c)
        add(b + attn + ".to_out.0.bias", c)
    add(b + "ff.net.0.proj.weight", 8 * c, c)
    add(b + "ff.net.0.proj.bias", 8 * c)
    add(b + "ff.net.2.weight", c, 4 * c)
    add(b + "ff.net.2.bias", c)
    for n in ("norm1", "norm2", "norm3"):
        add(b + n + ".weight", c)
        add(b + n + ".bias", c)
    add(prefix + "proj_out.weight", c, c, 1, 1)
    add(prefix + "proj_out.bias", c)


def unet():
    p = "model.diffusion_model."
    add(p + "time_embed.0.weight", 1280, 320)
    add(p + "time_embed.0.bias", 1280)
    add(p + "time_embed.2.weight", 1280, 1280)
    add(p + "time_embed.2.bias", 1280)
    add(p + "input_blocks.0.0.weight", 320, 4, 3, 3)
    add(p + "input_blocks.0.0.bias", 320)
    mult = [1, 2, 4, 4]
    attention = [True, True, True, False]
    skips = [320]
    ch = 320
    index = 1
    for level, m in enumerate(mult):
        out = 320 * m
        for _ in range(2):
            unet_res(p + "input_blocks.%d.0." % index, ch, out)
            ch = out
            if attention[level]:
                unet_transformer(p + "input_blocks.%d.1." % index, ch)
            skips.append(ch)
            index += 1
        if level != len(mult) - 1:
            add(p + "input_blocks.%d.0.op.weight" % index, ch, ch, 3, 3)
            add(p + "input_blocks.%d.0.op.bias" % index, ch)
            skips.append(ch)
            index += 1
    unet_res(p + "middle_block.0.", ch, ch)
    unet_transformer(p + "middle_block.1.", ch)
    unet_res(p + "middle_block.2.", ch, ch)
    index = 0
    for level in reversed(range(len(mult))):
        out = 320 * mult[level]
        for i in range(3):
            skip = skips.pop()
            unet_res(p + "output_blocks.%d.0." % index, ch + skip, out)
            ch = out
            sub = 1
            if attention[level]:
                unet_transformer(p + "output_blocks.%d.1." % index, ch)
                sub = 2
            if level != 0 and i == 2:
                add(p + "output_blocks.%d.%d.conv.weight" % (index, sub), ch, ch, 3, 3)
                add(p + "output_blocks.%d.%d.conv.bias" % (index, sub), ch)
            index += 1
    add(p + "out.0.weight", 320)
    add(p + "out.0.bias", 320)
    add(p + "out.2.weight", 4, 320, 3, 3)
    add(p + "out.2.bias", 4)


def vae_res(prefix, cin, cout):
    add(prefix + "norm1.weight", cin)
    add(prefix + "norm1.bias", cin)
    add(prefix + "conv1.weight", cout, cin, 3, 3)
    add(prefix + "conv1.bias", cout)
    add(prefix + "norm2.weight", cout)
    add(prefix + "norm2.bias", cout)
    add(prefix + "conv2.weight", cout, cout, 3, 3)
    add(prefix + "conv2.bias", cout)
    if cin != cout:
        add(prefix + "nin_shortcut.weight", cout, cin, 1, 1)
        add(prefix + "nin_shortcut.bias", cout)


def vae_attn(prefix, c):
    add(prefix + "norm.weight", c)
    add(prefix + "norm.bias", c)
    for n in ("q", "k", "v", "proj_out"):
        add(prefix + n + ".weight", c, c, 1, 1)
        add(prefix + n + ".bias", c)


def vae():
    p = "first_stage_model."
    mult = [1, 2, 4, 4]
    add(p + "encoder.conv_in.weight", 128, 3, 3, 3)
    add(p + "encoder.conv_in.bias", 128)
    ch = 128
    for level, m in enumerate(mult):
        out = 128 * m
        for b in range(2):
            vae_res(p + "encoder.down.%d.block.%d." % (level, b), ch, out)
            ch = out
        if level != len(mult) - 1:
            add(p + "encoder.down.%d.downsample.conv.weight" % level, ch, ch, 3, 3)
            add(p + "encoder.down.%d.downsample.conv.bias" % level, ch)
    vae_res(p + "encoder.mid.block_1.", ch, ch)
    vae_attn(p + "encoder.mid.attn_1.", ch)
    vae_res(p + "encoder.mid.block_2.", ch, ch)
    add(p + "encoder.norm_out.weight", ch)
    add(p + "encoder.norm_out.bias", ch)
    add(p + "encoder.conv_out.weight", 8, ch, 3, 3)
    add(p + "encoder.conv_out.bias", 8)
    add(p + "quant_conv.weight", 8, 8, 1, 1)
    add(p + "quant_conv.bias", 8)
    add(p + "post_quant_conv.weight", 4, 4, 1, 1)
    add(p + "post_quant_conv.bias", 4)
    ch = 512
    add(p + "decoder.conv_in.weight", ch, 4, 3, 3)
    add(p + "decoder.conv_in.bias", ch)
    vae_res(p + "decoder.mid.block_1.", ch, ch)
    vae_attn(p + "decoder.mid.attn_1.", ch)
    vae_res(p + "decoder.mid.block_2.", ch, ch)
    for level in reversed(range(len(mult))):
        out = 128 * mult[level]
        for b in range(3):
            vae_res(p + "decoder.up.%d.block.%d." % (level, b), ch, out)
            ch = out
        if level != 0:
            add(p + "decoder.up.%d.upsample.conv.weight" % level, ch, ch, 3, 3)
            add(p + "decoder.up.%d.upsample.conv.bias" % level, ch)
    add(p + "decoder.norm_out.weight", ch)
    add(p + "decoder.norm_out.bias", ch)
    add(p + "decoder.conv_out.weight", 3, ch, 3, 3)
    add(p + "decoder.conv_out.bias", 3)


def clip():
    p = "cond_stage_model.transformer.text_model."
    add(p + "embeddings.token_embedding.weight", 49408, 768)
    add(p + "embeddings.position_embedding.weight", 77, 768)
    for i in range(12):
        l = p + "encoder.layers.%d." % i
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            add(l + "self_attn." + n + ".weight", 768, 768)
            add(l + "self_attn." + n + ".bias", 768)
        for n in ("layer_norm1", "layer_norm2"):
            add(l + n + ".weight", 768)
            add(l + n + ".bias", 768)
        add(l + "mlp.fc1.weight", 3072, 768)
        add(l + "mlp.fc1.bias", 3072)
        add(l + "mlp.fc2.weight", 768, 3072)
        add(l + "mlp.fc2.bias", 768)
    add(p + "final_layer_norm.weight", 768)
    add(p + "final_layer_norm.bias", 768)


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    out = sys.argv[1]
    unet()
    vae()
    clip()

    header = {}
    offset = 0
    for name, shape in tensors:
        size = int(np.prod(shape)) * 2
        header[name] = {"dtype": "F16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    rng = np.random.default_rng(1234)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for name, shape in tensors:
            if name.endswith(".bias"):
                data = np.zeros(shape, dtype=np.float16)
            elif len(shape) == 1:
                data = np.ones(shape, dtype=np.float16)
            else:
                data = (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(np.float16)
            f.write(data.tobytes())
    print("%d tensors, %.2f GiB" % (len(tensors), offset / 2**30))
    return 0


if __name__ == "__main__":
    sys.exit(main())
