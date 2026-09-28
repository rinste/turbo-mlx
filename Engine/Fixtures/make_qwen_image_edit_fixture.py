#!/usr/bin/env python3
"""Builds a small Qwen-Image-Edit checkpoint with random weights, in mflux's format, plus what
mflux computes from it for one picture: the native engine's `verify` compares its own results
against them, stage by stage (see make_qwen_image_fixture.py for the text-to-image half).

Run it with a Python that has mflux (Engine/Fixtures/requirements.txt):
  $PY Engine/Fixtures/make_qwen_image_edit_fixture.py <out-dir>

The picture is made here (192 × 128, the image's proportions, so mflux's resize to the image's
size keeps them as the engine does) and goes through mflux's own preprocessing: Pillow's bicubic
resizes and Qwen2.5-VL's image processor for the vision tower, Pillow's Lanczos for the VAE. The
modules are Qwen-Image's fixture ones plus a vision tower of 4 blocks of 64 (full attention in
blocks 1 and 3) and the VAE's encoder at stages of 16, 32 and 64 channels. The picture's
placeholder token is 500 instead of 151655, inside the small vocabulary.
"""

import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_qwen_image_fixture import BITS, randomize, small_text_encoder, small_vae  # noqa: E402

from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.common.config.model_config import ModelConfig  # noqa: E402
from mflux.models.common.vae.vae_util import VAEUtil  # noqa: E402
from mflux.models.common.weights.saving.model_saver import ModelSaver  # noqa: E402
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator  # noqa: E402
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder  # noqa: E402
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_transformer import VisionTransformer  # noqa: E402
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer  # noqa: E402
from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D  # noqa: E402
from mflux.models.qwen.model.qwen_vae.qwen_image_down_block_3d import QwenImageDownBlock3D  # noqa: E402
from mflux.models.qwen.model.qwen_vae.qwen_image_encoder_3d import QwenImageEncoder3D  # noqa: E402
from mflux.models.qwen.model.qwen_vae.qwen_image_mid_block_3d import QwenImageMidBlock3D  # noqa: E402
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm  # noqa: E402
from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor, smart_resize  # noqa: E402
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil  # noqa: E402
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage  # noqa: E402
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition  # noqa: E402
from mflux.utils.image_util import ImageUtil  # noqa: E402

IMAGE_TOKEN = 500
VISION_START, VISION_END = 501, 502

CONFIG = {
    "family": "qwen-image-edit",
    "variant": "fixture",
    "transformer": {"num_layers": 2, "num_attention_heads": 2, "attention_head_dim": 128, "joint_attention_dim": 256},
    "text_encoder": {
        "vocab_size": 512, "hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "intermediate_size": 512, "drop_index": 64, "image_token_id": IMAGE_TOKEN,
    },
    "vision": {"embed_dim": 64, "depth": 4, "num_heads": 2, "mlp_hidden_dim": int(64 * 2.671875), "window_size": 112,
               "fullatt_block_indexes": [1, 3]},
    "vae": {"base_dim": 16},
    "picture": [192, 128],  # width, height
    "height": 64,
    "width": 96,
    "steps": 3,
    "seed": 11,
    "guidance": 2.5,
}


def small_encoder(base_dim: int) -> QwenImageEncoder3D:
    """mflux's VAE encoder rebuilt from the same blocks at a fraction of the width."""
    d1, d2, d4 = base_dim, 2 * base_dim, 4 * base_dim
    encoder = QwenImageEncoder3D()
    encoder.conv_in = QwenImageCausalConv3D(3, d1, 3, 1, 1)
    encoder.down_blocks = [
        QwenImageDownBlock3D(d1, d1, num_res_blocks=2, downsample_mode="downsample2d"),
        QwenImageDownBlock3D(d1, d2, num_res_blocks=2, downsample_mode="downsample2d"),
        QwenImageDownBlock3D(d2, d4, num_res_blocks=2, downsample_mode="downsample3d"),
        QwenImageDownBlock3D(d4, d4, num_res_blocks=2, downsample_mode=None),
    ]
    encoder.mid_block = QwenImageMidBlock3D(d4, num_layers=1)
    encoder.norm_out = QwenImageRMSNorm(d4, images=False)
    encoder.conv_out = QwenImageCausalConv3D(d4, 32, 3, 1, 1)
    return encoder


def make_picture(width: int, height: int) -> Image.Image:
    """Smooth colors with a few sharp edges, so both resizes have something to do."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    r = 127 + 120 * np.sin(x / 13.0 + y / 29.0)
    g = 127 + 120 * np.cos(y / 11.0 - x / 37.0)
    b = np.where((x // 24 + y // 24) % 2 == 0, 40.0, 215.0)
    b[(x - 120) ** 2 + (y - 60) ** 2 < 900] = 250.0
    return Image.fromarray(np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8), "RGB")


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    width, height = cfg["width"], cfg["height"]

    transformer = QwenTransformer(**{k: v for k, v in cfg["transformer"].items()})
    text_encoder = small_text_encoder(cfg)
    s = cfg["vision"]
    text_encoder.encoder.visual = VisionTransformer(
        embed_dim=s["embed_dim"], depth=s["depth"], num_heads=s["num_heads"], mlp_ratio=2.671875,
        hidden_size=cfg["text_encoder"]["hidden_size"], window_size=s["window_size"],
        fullatt_block_indexes=s["fullatt_block_indexes"],
    )
    text_encoder.encoder.image_token_id = IMAGE_TOKEN
    vae = small_vae(cfg["vae"]["base_dim"])
    vae.encoder = small_encoder(cfg["vae"]["base_dim"])
    vae.quant_conv = QwenImageCausalConv3D(32, 32, 1, 1, 0)
    randomize(transformer, seed=1, scale=1.0)
    randomize(text_encoder, seed=2, scale=1.0)
    randomize(vae, seed=3, scale=1.0, dtype=mx.float32)
    nn.quantize(
        transformer, group_size=64, bits=BITS,
        class_predicate=lambda path, m: QwenWeightDefinition.quantization_predicate(path, m, BITS),
    )
    mx.eval(transformer.parameters(), text_encoder.parameters(), vae.parameters())

    # --- the picture, saved as mflux reads it --------------------------------------------------------------
    picture = make_picture(*cfg["picture"])
    picture_path = out / "picture.png"
    picture.save(picture_path)
    picture = ImageUtil.load_image(picture_path).convert("RGB")
    picture_rgb = np.array(picture)

    # --- reference: the vision tower's input (tokenize_with_image, then QwenImageProcessor) ------------------
    ratio = picture.width / picture.height
    condition_width = math.sqrt(384 * 384 * ratio)
    condition_height = condition_width / ratio
    condition = picture.resize((int(round(condition_width / 32) * 32), int(round(condition_height / 32) * 32)), Image.BICUBIC)
    processor = QwenImageProcessor()
    resized_h, resized_w = smart_resize(condition.height, condition.width, factor=28,
                                        min_pixels=processor.min_pixels, max_pixels=processor.max_pixels)
    vl_image = condition.resize((resized_w, resized_h), Image.BICUBIC) if (condition.height, condition.width) != (resized_h, resized_w) else condition
    pixel_values, grid = processor.preprocess([condition])
    image_grid_thw = mx.array(grid)
    pixel_values = mx.array(pixel_values)

    # --- reference: the vision tower, then the prompt with the picture's tokens ------------------------------
    image_embeds = text_encoder.encoder.visual(pixel_values.astype(mx.float32), image_grid_thw)
    mx.eval(image_embeds)
    tokens = int(np.prod(grid[0])) // 4

    def ids(prompt_ids: list[int]) -> mx.array:
        system = [(i * 37 + 11) % 499 + 1 for i in range(64)]
        suffix = [(i * 17 + 3) % 499 + 1 for i in range(5)]
        return mx.array([system + [VISION_START] + [IMAGE_TOKEN] * tokens + [VISION_END] + prompt_ids + suffix], dtype=mx.int32)

    input_ids = ids([(i * 53 + 5) % 499 + 1 for i in range(20)])
    negative_input_ids = ids([])
    vl_encoder = QwenVisionLanguageEncoder(encoder=text_encoder.encoder)

    def encode(token_ids: mx.array) -> tuple[mx.array, mx.array]:
        embeds, mask = vl_encoder(input_ids=token_ids, attention_mask=mx.ones_like(token_ids),
                                  pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        return embeds.astype(mx.float16), mask.astype(mx.float16)  # as QwenImageEdit casts them

    prompt_embeds, prompt_mask = encode(input_ids)
    negative_embeds, negative_mask = encode(negative_input_ids)
    mx.eval(prompt_embeds, negative_embeds)

    # --- reference: the picture's latents at the image's size -----------------------------------------------
    vae_rgb = np.array(ImageUtil.scale_to_dimensions(picture, target_width=width, target_height=height))
    vae_input = ImageUtil.to_array(ImageUtil.scale_to_dimensions(picture, target_width=width, target_height=height))
    reference_latents, image_ids, cond_h, cond_w, _ = QwenEditUtil.create_image_conditioning_latents(
        vae=vae, width=width, height=height, image_paths=[str(picture_path)], tiling_config=None,
    )
    mx.eval(reference_latents)

    # --- reference: one pass and the guided loop, as QwenImageEdit.generate_image runs them --------------------
    config = Config(model_config=ModelConfig.qwen_image_edit(), num_inference_steps=cfg["steps"], height=height, width=width,
                    guidance=cfg["guidance"], scheduler="linear")
    latents = QwenLatentCreator.create_noise(seed=cfg["seed"], height=height, width=width)
    timestep = 0.75
    noise = transformer(t=timestep, config=config, hidden_states=mx.concatenate([latents, reference_latents], axis=1),
                        encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask,
                        qwen_image_ids=image_ids, cond_image_grid=(1, cond_h, cond_w))[:, : latents.shape[1]]
    mx.eval(noise)
    x = latents
    for t in range(cfg["steps"]):
        hidden = mx.concatenate([x, reference_latents], axis=1)
        positive = transformer(t=t, config=config, hidden_states=hidden, encoder_hidden_states=prompt_embeds,
                               encoder_hidden_states_mask=prompt_mask, qwen_image_ids=image_ids,
                               cond_image_grid=(1, cond_h, cond_w))[:, : x.shape[1]]
        negative = transformer(t=t, config=config, hidden_states=hidden, encoder_hidden_states=negative_embeds,
                               encoder_hidden_states_mask=negative_mask, qwen_image_ids=image_ids,
                               cond_image_grid=(1, cond_h, cond_w))[:, : x.shape[1]]
        guided = QwenImage.compute_guided_noise(positive, negative, config.guidance)
        x = config.scheduler.step(noise=guided, timestep=t, latents=x)
        mx.eval(x)
    final_latents = x
    decoded = VAEUtil.decode(vae=vae, latent=QwenLatentCreator.unpack_latents(latents=final_latents, height=height, width=width),
                             tiling_config=None)
    mx.eval(decoded)

    for name, module in (("transformer", transformer), ("text_encoder", text_encoder), ("vae", vae)):
        ModelSaver._save_weights(str(out), BITS, module, name)
    mx.save_safetensors(str(out / "references.safetensors"), {
        "picture_rgb": mx.array(picture_rgb),
        "vl_rgb": mx.array(np.array(vl_image)),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw.astype(mx.int32),
        "image_embeds": image_embeds,
        "input_ids": input_ids,
        "negative_input_ids": negative_input_ids,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_embeds,
        "vae_rgb": mx.array(vae_rgb),
        "vae_input": vae_input,
        "reference_latents": reference_latents,
        "latents": latents,
        "timestep": mx.array([timestep], dtype=mx.float32),
        "noise": noise,
        "sigmas": config.scheduler.sigmas,
        "final_latents": final_latents,
        "decoded": decoded,
    })
    with open(out / "fixture.json", "w") as f:
        json.dump({**cfg, "bits": BITS}, f, indent=2)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"fixture written to {out} ({total / 1e6:.1f} MB)")
    print("vision grid", grid.tolist(), "→", tokens, "tokens; condition", condition.size, "→", vl_image.size,
          "| prompt", prompt_embeds.shape, prompt_embeds.dtype, "| reference", reference_latents.shape,
          "| decoded", decoded.shape)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "qwen-image-edit-fixture"))
