#!/usr/bin/env python3
"""Builds a small Z-Image Turbo checkpoint with random weights, in mflux's format, plus the outputs
mflux computes from it: the native engine's `verify` compares its own outputs against them, so the
Swift port is checked module by module before a real checkpoint and a real Mac are involved.

Run it with a Python that has mflux (Engine/Fixtures/requirements.txt):
  $PY Engine/Fixtures/make_zimage_fixture.py <out-dir>

Everything is shrunk: the Qwen3 encoder to 4 layers of 128 (Z-Image reads its second-to-last
hidden state), the S3-DiT to two layers of 256, the VAE to a few dozen channels per stage (mflux's
own blocks, only narrower). Its encoder serves image to image: the references include a picture
encoded, mixed with the noise at the sigma of the step `IMAGE_STRENGTH` starts from, and the loop
from there, as mflux's `Img2Img` runs it.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.z_image.latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.text_encoder import TextEncoder
from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
from mflux.models.z_image.model.z_image_vae.common.unet_mid_block import UNetMidBlock
from mflux.models.z_image.model.z_image_vae.decoder.conv_in import ConvIn
from mflux.models.z_image.model.z_image_vae.decoder.conv_norm_out import ConvNormOut
from mflux.models.z_image.model.z_image_vae.decoder.conv_out import ConvOut
from mflux.models.z_image.model.z_image_vae.decoder.up_decoder_block import UpDecoderBlock
from mflux.models.z_image.model.z_image_vae.encoder.conv_in import ConvIn as EncoderConvIn
from mflux.models.z_image.model.z_image_vae.encoder.conv_norm_out import ConvNormOut as EncoderConvNormOut
from mflux.models.z_image.model.z_image_vae.encoder.conv_out import ConvOut as EncoderConvOut
from mflux.models.z_image.model.z_image_vae.encoder.down_encoder_block import DownEncoderBlock
from mflux.models.z_image.model.z_image_vae.vae import VAE

BITS = 4  # the catalog's 16 GB checkpoint is 4-bit (the 24 GB one, 8-bit, loads the same way)
IMAGE_STRENGTH = 0.5  # image to image: the part of the picture kept, mflux's --image-strength

CONFIG = {
    "family": "z-image-turbo",
    "variant": "fixture",
    "transformer": {
        "dim": 384,             # head_dim stays 128 (the rotary axes 32 + 48 + 48 add up to it);
        "n_layers": 2,          # the feed-forward width, dim / 3 · 8, must be a multiple of 64
        "n_refiner_layers": 1,
        "n_heads": 3,
        "cap_feat_dim": 128,    # the text encoder's hidden size
    },
    "text_encoder": {
        "vocab_size": 512,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "head_dim": 128,
    },
    "vae": {"block_out_channels": [32, 64, 64, 64]},  # first stage first; 32 groups need ≥ 32 channels
    "max_sequence_length": 512,
    "height": 128,
    "width": 128,
    "steps": 4,
    "seed": 7,
}


def randomize(module: nn.Module, seed: int, scale: float, dtype=mx.bfloat16) -> None:
    """Small normal weights (norms at 1 + noise), so activations stay in a sane range, in the
    dtype mflux converts a real checkpoint to (bf16 for every tensor, biases and norms included)."""
    key = mx.random.key(seed)
    updates = []
    for name, value in tree_flatten(module.parameters()):
        key, sub = mx.random.split(key)
        if name.endswith("inv_freq"):
            continue
        if value.ndim == 1 and ("norm" in name or name.endswith(".bias")):
            noise = 0.05 * mx.random.normal(value.shape, key=sub)
            new = (mx.ones(value.shape) + noise) if "norm" in name else noise
        else:
            new = scale * mx.random.normal(value.shape, key=sub) / (value.shape[-1] ** 0.5 if value.ndim > 1 else 1)
        updates.append((name, new.astype(dtype)))
    module.update(tree_unflatten(updates))
    mx.eval(module.parameters())


def small_vae(block_out_channels: list[int]) -> VAE:
    """mflux's VAE with its encoder and decoder rebuilt from the same blocks at a fraction of the width."""
    vae = VAE()
    encoder = vae.encoder
    encoder.conv_in = EncoderConvIn(in_channels=3, out_channels=block_out_channels[0])
    encoder.down_blocks = [
        DownEncoderBlock(
            in_channels=block_out_channels[0] if i == 0 else block_out_channels[i - 1],
            out_channels=block_out_channels[i],
            num_layers=2,
            add_downsample=i != len(block_out_channels) - 1,
        )
        for i in range(len(block_out_channels))
    ]
    encoder.mid_block = UNetMidBlock(channels=block_out_channels[-1])
    encoder.conv_norm_out = EncoderConvNormOut(num_channels=block_out_channels[-1], num_groups=32)
    encoder.conv_out = EncoderConvOut(in_channels=block_out_channels[-1], out_channels=32)
    channels = list(reversed(block_out_channels))
    decoder = vae.decoder
    decoder.conv_in = ConvIn(in_channels=16, out_channels=channels[0])
    decoder.mid_block = UNetMidBlock(channels=channels[0])
    decoder.up_blocks = [
        UpDecoderBlock(
            in_channels=channels[0] if i == 0 else channels[i - 1],
            out_channels=channels[i],
            num_layers=3,
            add_upsample=i != len(channels) - 1,
        )
        for i in range(len(channels))
    ]
    decoder.conv_norm_out = ConvNormOut(channels=block_out_channels[0])
    decoder.conv_out = ConvOut(in_channels=block_out_channels[0], out_channels=3)
    return vae


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG

    transformer = ZImageTransformer(**cfg["transformer"])
    text_encoder = TextEncoder(**cfg["text_encoder"])
    vae = small_vae(cfg["vae"]["block_out_channels"])
    randomize(transformer, seed=1, scale=1.0)
    randomize(text_encoder, seed=2, scale=1.0)
    randomize(vae, seed=3, scale=1.0)

    # Quantize exactly as mflux does for a saved checkpoint (every Linear and Embedding, group 64).
    for module in (transformer, text_encoder, vae):
        nn.quantize(module, group_size=64, bits=BITS, class_predicate=lambda path, m: hasattr(m, "to_quantized"))
    mx.eval(transformer.parameters(), text_encoder.parameters(), vae.parameters())

    # --- reference: text encoder (the second-to-last hidden state of the real tokens) --------------
    input_ids = mx.array([[5, 17, 42, 99, 7, 3, 250, 11, 61, 62, 63, 0, 0, 0, 0, 0]], dtype=mx.int32)
    attention_mask = (input_ids != 0).astype(mx.int32)
    num_valid = int(mx.sum(attention_mask[0]).item())
    cap_feats = text_encoder(input_ids, attention_mask)[0, :num_valid, :]
    mx.eval(cap_feats)

    # --- reference: the schedule Z-Image Turbo runs (linear, shifted for the size) -------------------
    config = Config(
        model_config=ModelConfig.z_image_turbo(),
        num_inference_steps=cfg["steps"],
        height=cfg["height"],
        width=cfg["width"],
        guidance=0.0,
        scheduler="linear",
    )
    sigmas = config.scheduler.sigmas

    # --- reference: one transformer pass ---------------------------------------------------------------
    latents = ZImageLatentCreator.create_noise(seed=cfg["seed"], height=cfg["height"], width=cfg["width"])
    timestep = mx.array([0.75], dtype=mx.float32)
    noise = transformer(x=latents, timestep=timestep, sigmas=sigmas, cap_feats=cap_feats)
    mx.eval(noise)

    # --- reference: the whole denoising loop ------------------------------------------------------------
    x = latents
    for t in range(cfg["steps"]):
        sigma_t = sigmas[t].reshape((1,))
        pred = transformer(x=x, timestep=mx.ones_like(sigma_t) - sigma_t, sigmas=sigmas, cap_feats=cap_feats)
        x = config.scheduler.step(noise=pred, timestep=t, latents=x)
        mx.eval(x)
    final_latents = x

    # --- reference: VAE decode ---------------------------------------------------------------------------
    unpacked = ZImageLatentCreator.unpack_latents(final_latents, cfg["height"], cfg["width"])
    decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=None)
    mx.eval(decoded)

    # --- reference: image to image (`LatentCreator.create_for_txt2img_or_img2img` and the loop) -----------
    # A picture as `encode_image` hands it over, [1, 3, H, W] float32 in [-1, 1]; its latents mixed with
    # the same noise at the sigma of the step the strength starts from (`Config.init_time_step`).
    image = mx.random.uniform(low=-1.0, high=1.0, shape=(1, 3, cfg["height"], cfg["width"]), key=mx.random.key(5))
    encoded = VAEUtil.encode(vae=vae, image=image, tiling_config=None)
    clean = ZImageLatentCreator.pack_latents(encoded, cfg["height"], cfg["width"])
    start = max(1, int(cfg["steps"] * IMAGE_STRENGTH))
    img2img_latents = LatentCreator.add_noise_by_interpolation(clean=clean, noise=latents, sigma=sigmas[start])
    x = img2img_latents
    for t in range(start, cfg["steps"]):
        sigma_t = sigmas[t].reshape((1,))
        pred = transformer(x=x, timestep=mx.ones_like(sigma_t) - sigma_t, sigmas=sigmas, cap_feats=cap_feats)
        x = config.scheduler.step(noise=pred, timestep=t, latents=x)
        mx.eval(x)
    img2img_final = x

    # --- save the checkpoint the way mflux does (component folders, shards, metadata) ------------------
    for name, module in (("transformer", transformer), ("text_encoder", text_encoder), ("vae", vae)):
        ModelSaver._save_weights(str(out), BITS, module, name)

    mx.save_safetensors(str(out / "references.safetensors"), {
        "input_ids": input_ids[:, :num_valid],
        "prompt_embeds": cap_feats,
        "latents": latents,
        "timestep": timestep,
        "noise": noise,
        "sigmas": sigmas,
        "final_latents": final_latents,
        "decoded": decoded,
        "image": image,
        "encoded": encoded,
        "img2img_latents": img2img_latents,
        "img2img_final": img2img_final,
    })
    with open(out / "fixture.json", "w") as f:
        json.dump({**cfg, "bits": BITS, "image_strength": IMAGE_STRENGTH}, f, indent=2)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"fixture written to {out} ({total / 1e6:.1f} MB)")
    print("sigmas:", [round(float(s), 5) for s in sigmas.tolist()])
    print("prompt_embeds", cap_feats.shape, cap_feats.dtype, "| noise", noise.shape, noise.dtype, "| decoded", decoded.shape,
          f"range [{float(decoded.min()):.3f}, {float(decoded.max()):.3f}]")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "zimage-fixture"))
