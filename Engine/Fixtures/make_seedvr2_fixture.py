#!/usr/bin/env python3
"""Builds a small SeedVR2 checkpoint with random weights, in the original checkpoint's format
(numz/SeedVR2_comfyUI: `seedvr2_ema_3b_fp16.safetensors` and `ema_vae_fp16.safetensors`, float16,
convolutions [O, I, kt, kh, kw]), loads it back through mflux's own weight mapping, and records
what mflux computes from it for one picture: the native engine's `verify` compares its own results
against them, stage by stage.

Run it with a Python that has mflux (Engine/Fixtures/requirements.txt):
  $PY Engine/Fixtures/make_seedvr2_fixture.py <out-dir>

The transformer keeps the 3B model's structure at a fraction of the width: 4 blocks of 256 (two
heads of 128), the first 2 with separate video and text weights, the last one video-only, the
3B's windows. The VAE keeps its blocks at 32, 32, 64 and 64 channels. The text embedding is a
random 58 × 320 one, saved with the references (the engine ships the real one). The picture is
300 × 200, upscaled 2 × to 600 × 400 (padded to 608 × 400): large enough that mflux encodes and
decodes it in tiles, and the video tokens fall into windows of several sizes.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from PIL import Image

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.seedvr2.latent_creator.seedvr2_latent_creator import SeedVR2LatentCreator
from mflux.models.seedvr2.model.seedvr2_transformer.transformer import SeedVR2Transformer
from mflux.models.seedvr2.model.seedvr2_vae.vae import SeedVR2VAE
from mflux.models.seedvr2.variants.upscale.seedvr2_util import SeedVR2Util
from mflux.models.seedvr2.weights.seedvr2_weight_definition import SeedVR2WeightDefinition
from mflux.models.seedvr2.weights.seedvr2_weight_mapping import SeedVR2WeightMapping
from mflux.utils.scale_factor import ScaleFactor

CONFIG = {
    "family": "seedvr2",
    "variant": "fixture",
    "transformer": {"vid_dim": 256, "txt_in_dim": 320, "heads": 2, "head_dim": 128, "num_layers": 4, "mm_layers": 2,
                    "rope_dim": 128},
    "vae": {"block_out_channels": [32, 32, 64, 64]},
    "picture": [300, 200],  # width, height
    "upscale": 2,
    "softness": 0.5,
    "seed": 11,
}


class FixtureDefinition:
    """SeedVR2WeightDefinition3B for the fixture's 4 blocks."""

    @staticmethod
    def get_components():
        blocks = CONFIG["transformer"]["num_layers"]
        return [
            ComponentDefinition(name="transformer", hf_subdir=".", num_blocks=blocks, loading_mode="mlx_native",
                                mapping_getter=lambda: SeedVR2WeightMapping.get_transformer_mapping(num_blocks=blocks),
                                weight_files=["seedvr2_ema_3b_fp16.safetensors"]),
            ComponentDefinition(name="vae", hf_subdir=".", num_blocks=4, loading_mode="mlx_native",
                                mapping_getter=SeedVR2WeightMapping.get_vae_mapping, weight_files=["ema_vae_fp16.safetensors"]),
        ]

    @staticmethod
    def get_tokenizers():
        return []

    @staticmethod
    def get_download_patterns():
        return ["seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors"]

    @staticmethod
    def quantization_predicate(path, module):
        return SeedVR2WeightDefinition.quantization_predicate(path, module)


def randomize(module: nn.Module, seed: int) -> None:
    """Small normal weights (norms and scales at 1 + noise), float16 as the checkpoint stores them."""
    key = mx.random.key(seed)
    updates = []
    for name, value in tree_flatten(module.parameters()):
        key, sub = mx.random.split(key)
        noise = mx.random.normal(value.shape, key=sub)
        if name.endswith("rope.freqs"):
            new = value  # the frequencies, as the checkpoint stores them
        elif value.ndim == 1 and ("norm" in name or name.endswith("scale")):
            new = mx.ones(value.shape) + 0.05 * noise
        elif value.ndim == 1:
            new = 0.05 * noise
        elif value.ndim == 5:  # convolutions [O, kt, kh, kw, I]
            new = noise / (value[0].size ** 0.5)
        else:
            new = noise / (value.shape[-1] ** 0.5)
        updates.append((name, new.astype(mx.float16)))
    module.update(tree_unflatten(updates))
    mx.eval(module.parameters())


def checkpoint_name(name: str, shared_from: int) -> str:
    """mflux's parameter name → the original checkpoint's (the inverse of SeedVR2WeightMapping)."""
    parts = name.split(".")
    if parts[0] == "blocks":
        block = int(parts[1])
        stream = "all" if block >= shared_from else None
        rest = ".".join(parts[2:])
        for layer in ("proj_qkv", "proj_out", "norm_q", "norm_k"):
            for side in ("vid", "txt"):
                prefix = f"attn.{layer}_{side}."
                if rest.startswith(prefix):
                    return f"blocks.{block}.attn.{layer}.{stream or side}.{rest[len(prefix):]}"
        if rest == "attn.rope.freqs":
            return f"blocks.{block}.attn.rope.rope.freqs"
        for side in ("vid", "txt", "all"):
            prefix = f"ada.params_{side}."
            if rest.startswith(prefix):
                return f"blocks.{block}.ada.{side}.{rest[len(prefix):]}"
        return name
    if name in ("out_shift", "out_scale"):
        return f"vid_out_ada.{name}"
    return name


def save_checkpoint(out: Path, transformer: SeedVR2Transformer, vae: SeedVR2VAE) -> None:
    shared_from = CONFIG["transformer"]["mm_layers"]
    tensors = {}
    for name, value in tree_flatten(transformer.parameters()):
        key = checkpoint_name(name, shared_from)
        if key in tensors:  # a shared block's text layers are its video layers
            continue
        tensors[key] = value
    mx.save_safetensors(str(out / "seedvr2_ema_3b_fp16.safetensors"), tensors)
    vae_tensors = {name: value.transpose(0, 4, 1, 2, 3) if value.ndim == 5 else value
                   for name, value in tree_flatten(vae.parameters())}
    mx.save_safetensors(str(out / "ema_vae_fp16.safetensors"), vae_tensors)


def make_picture(width: int, height: int) -> Image.Image:
    """Smooth colors with sharp edges and a disc, so the resizes and the color match have something to do."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    r = 127 + 120 * np.sin(x / 13.0 + y / 29.0)
    g = 127 + 120 * np.cos(y / 11.0 - x / 37.0)
    b = np.where((x // 24 + y // 24) % 2 == 0, 40.0, 215.0)
    b[(x - 120) ** 2 + (y - 60) ** 2 < 900] = 250.0
    return Image.fromarray(np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8), "RGB")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG

    # --- the checkpoint: random modules saved in the original format, read back by mflux's loader ---------------
    t = cfg["transformer"]
    source_transformer = SeedVR2Transformer(**t)
    source_vae = SeedVR2VAE(block_out_channels=tuple(cfg["vae"]["block_out_channels"]))
    randomize(source_transformer, seed=1)
    randomize(source_vae, seed=2)
    save_checkpoint(out, source_transformer, source_vae)
    transformer = SeedVR2Transformer(**t)
    vae = SeedVR2VAE(block_out_channels=tuple(cfg["vae"]["block_out_channels"]))
    weights = WeightLoader.load(weight_definition=FixtureDefinition, model_path=str(out))
    WeightApplier.apply_and_quantize(weights=weights, quantize_arg=None, weight_definition=FixtureDefinition,
                                     models={"transformer": transformer, "vae": vae})
    for (name, loaded), (_, source) in zip(tree_flatten(transformer.parameters()), tree_flatten(source_transformer.parameters())):
        assert mx.array_equal(loaded, source), f"mflux did not read {name} back"
    text = (mx.random.normal((1, 58, t["txt_in_dim"]), key=mx.random.key(3))).astype(mx.float16)

    # --- the picture, as SeedVR2Util.preprocess_image prepares it --------------------------------------------
    picture = make_picture(*cfg["picture"])
    picture_path = out / "picture.png"
    picture.save(picture_path)
    factor = ScaleFactor(cfg["upscale"])
    processed, true_height, true_width = SeedVR2Util.preprocess_image(image_path=picture_path, resolution=factor, softness=0.0)
    softened, _, _ = SeedVR2Util.preprocess_image(image_path=picture_path, resolution=factor, softness=cfg["softness"])

    # --- encode (in tiles), noise, one transformer pass, the step, decode (in tiles) --------------------------
    tiling = TilingConfig()
    latent = VAEUtil.encode(vae=vae, image=processed, tiling_config=tiling)
    condition = SeedVR2LatentCreator.create_condition(encoded_latent=latent)
    noise = SeedVR2LatentCreator.create_noise_latents(seed=cfg["seed"], height=latent.shape[-2], width=latent.shape[-1])
    model_input = mx.concatenate([noise, condition], axis=1)
    flow = transformer(txt=text, vid=model_input, timestep=mx.array(1000.0, dtype=mx.float32))
    mx.eval(flow)
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig
    config = Config(width=true_width, height=true_height, guidance=1.0, num_inference_steps=1, scheduler="seedvr2_euler",
                    model_config=ModelConfig.seedvr2_3b())
    latents = config.scheduler.step(noise=flow, timestep=0, latents=noise)
    decoded = VAEUtil.decode(vae=vae, latent=latents, tiling_config=tiling)
    decoded = decoded[:, :, :true_height, :true_width]
    style = processed[:, :, :true_height, :true_width]
    corrected = SeedVR2Util.apply_color_correction(decoded, style)
    pixels = (np.array(mx.clip(corrected / 2 + 0.5, 0, 1).transpose(0, 2, 3, 1).astype(mx.float32))[0] * 255).round().astype(np.uint8)
    mx.eval(latent, latents, decoded, corrected)

    references = {
        "picture_rgb": mx.array(np.array(picture)),
        "text": text,
        "input": processed,
        "softened": softened,
        "latent": latent if latent.ndim == 5 else latent[:, :, None],
        "noise": noise,
        "model_input": model_input,
        "flow": flow,
        "latents": latents,
        "decoded": decoded,
        "corrected": corrected,
        "pixels": mx.array(pixels),
    }
    mx.save_safetensors(str(out / "references.safetensors"), references)
    meta = dict(cfg)
    meta["output"] = [true_width, true_height]
    (out / "fixture.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}: {true_width} × {true_height} from {cfg['picture'][0]} × {cfg['picture'][1]}, latent {latent.shape}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fixtures/seedvr2"))
