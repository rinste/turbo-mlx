#!/usr/bin/env python3
"""Builds a small Qwen-Image checkpoint with random weights, in mflux's format, plus the outputs
mflux computes from it: the native engine's `verify` compares its own outputs against them, so the
Swift port is checked module by module before a real checkpoint and a real Mac are involved.

Run it with a Python that has mflux (Engine/Fixtures/requirements.txt):
  $PY Engine/Fixtures/make_qwen_image_fixture.py <out-dir>

Everything is shrunk: the Qwen2.5-VL language model to 2 layers of 256 (kept in bf16, unquantized,
as the catalog's checkpoint keeps it), the transformer to 2 blocks of 256, the 3D decoder to
stages of 16, 32 and 64 channels (mflux's own blocks, only narrower). The transformer is quantized
the way the 4-bit checkpoint is, with its modulation producers at 8 bits. The encoder half of the
VAE is left out, as the engine never loads it.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder_layer import QwenEncoderLayer
from mflux.models.qwen.model.qwen_text_encoder.qwen_rms_norm import QwenRMSNorm
from mflux.models.qwen.model.qwen_text_encoder.qwen_rope import QwenRotaryEmbedding
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_mid_block_3d import QwenImageMidBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm
from mflux.models.qwen.model.qwen_vae.qwen_image_up_block_3d import QwenImageUpBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

BITS = 4  # the catalog's checkpoint is 4-bit

CONFIG = {
    "family": "qwen-image",
    "variant": "fixture",
    "transformer": {
        "num_layers": 2,
        "num_attention_heads": 2,       # inner_dim 256, head_dim 128 (the rotary axes 16 + 56 + 56)
        "attention_head_dim": 128,
        "joint_attention_dim": 256,     # the text encoder's hidden size
    },
    "text_encoder": {
        "vocab_size": 512,
        "hidden_size": 256,             # heads × 128
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 512,
    },
    "vae": {"base_dim": 16},            # stages of 16, 32, 64 channels (96, 192, 384 in the model)
    "height": 64,
    "width": 64,
    "steps": 3,
    "seed": 7,
    "guidance": 4.0,
}


def randomize(module: nn.Module, seed: int, scale: float, dtype=mx.bfloat16) -> None:
    """Small normal weights (norms at 1 + noise), so activations stay in a sane range."""
    key = mx.random.key(seed)
    updates = []
    for name, value in tree_flatten(module.parameters()):
        key, sub = mx.random.split(key)
        if name.endswith("inv_freq"):
            continue
        if value.ndim == 1 and ("norm" in name or name.endswith(".bias")):
            noise = 0.05 * mx.random.normal(value.shape, key=sub)
            new = (mx.ones(value.shape) + noise) if "norm" in name else noise
        elif "norm" in name and value.ndim > 1 and value.shape[-1] == 1:
            new = mx.ones(value.shape) + 0.05 * mx.random.normal(value.shape, key=sub)  # the VAE's [C, 1, 1(, 1)] norms
        else:
            new = scale * mx.random.normal(value.shape, key=sub) / (value.shape[-1] ** 0.5 if value.ndim > 1 else 1)
        updates.append((name, new.astype(dtype)))
    module.update(tree_unflatten(updates))
    mx.eval(module.parameters())


def small_text_encoder(cfg: dict) -> QwenTextEncoder:
    """mflux's encoder with its layers rebuilt narrower (their width is fixed in the constructor)."""
    e = cfg["text_encoder"]
    encoder = QwenEncoder(vocab_size=e["vocab_size"], hidden_size=e["hidden_size"], num_hidden_layers=e["num_hidden_layers"])
    encoder.layers = [
        QwenEncoderLayer(
            hidden_size=e["hidden_size"],
            num_attention_heads=e["num_attention_heads"],
            num_key_value_heads=e["num_key_value_heads"],
            intermediate_size=e["intermediate_size"],
        )
        for _ in range(e["num_hidden_layers"])
    ]
    encoder.norm = QwenRMSNorm(e["hidden_size"], eps=1e-6)
    encoder.rotary_emb = QwenRotaryEmbedding(dim=e["hidden_size"] // e["num_attention_heads"])
    text_encoder = QwenTextEncoder()
    text_encoder.encoder = encoder
    return text_encoder


def small_vae(base_dim: int) -> QwenVAE:
    """mflux's VAE with its decoder rebuilt from the same blocks at a fraction of the width."""
    vae = QwenVAE()
    d1, d2, d4 = base_dim, 2 * base_dim, 4 * base_dim
    decoder = vae.decoder
    decoder.conv_in = QwenImageCausalConv3D(16, d4, 3, 1, 1)
    decoder.mid_block = QwenImageMidBlock3D(d4, num_layers=1)
    decoder.up_block0 = QwenImageUpBlock3D(d4, d4, num_res_blocks=2, upsample_mode="upsample3d")
    decoder.up_block1 = QwenImageUpBlock3D(d2, d4, num_res_blocks=2, upsample_mode="upsample3d")
    decoder.up_block2 = QwenImageUpBlock3D(d2, d2, num_res_blocks=2, upsample_mode="upsample2d")
    decoder.up_block3 = QwenImageUpBlock3D(d1, d1, num_res_blocks=2, upsample_mode=None)
    decoder.norm_out = QwenImageRMSNorm(d1, images=False)
    decoder.conv_out = QwenImageCausalConv3D(d1, 3, 3, 1, 1)
    vae.encoder = nn.Identity()  # never loaded by the engine; keeps the fixture small
    vae.quant_conv = nn.Identity()
    return vae


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG

    transformer = QwenTransformer(**cfg["transformer"])
    text_encoder = small_text_encoder(cfg)
    vae = small_vae(cfg["vae"]["base_dim"])
    randomize(transformer, seed=1, scale=1.0)
    randomize(text_encoder, seed=2, scale=1.0)
    randomize(vae, seed=3, scale=1.0, dtype=mx.float32)  # Qwen-Image's decoder ships in float32

    # Quantize the transformer as mflux saves it: every Linear, the modulation producers at 8 bits
    # when the level is 4. The text encoder is never quantized, the VAE has nothing to quantize.
    nn.quantize(
        transformer, group_size=64, bits=BITS,
        class_predicate=lambda path, m: QwenWeightDefinition.quantization_predicate(path, m, BITS),
    )
    mx.eval(transformer.parameters(), text_encoder.parameters(), vae.parameters())

    # --- reference: text encoder (the tokens after the template's 34, every one real) ------------------
    input_ids = mx.array([[(i * 37 + 11) % 500 + 1 for i in range(40)]], dtype=mx.int32)
    prompt_embeds, prompt_mask = text_encoder(input_ids=input_ids, attention_mask=mx.ones_like(input_ids))
    negative_input_ids = mx.array([[(i * 53 + 5) % 500 + 1 for i in range(36)]], dtype=mx.int32)
    negative_embeds, negative_mask = text_encoder(input_ids=negative_input_ids, attention_mask=mx.ones_like(negative_input_ids))
    mx.eval(prompt_embeds, negative_embeds)

    # --- reference: the schedule (linear, shifted for the size, stretched to the terminal) -------------
    config = Config(
        model_config=ModelConfig.qwen_image(),
        num_inference_steps=cfg["steps"],
        height=cfg["height"],
        width=cfg["width"],
        guidance=cfg["guidance"],
        scheduler="linear",
    )
    sigmas = config.scheduler.sigmas

    # --- reference: one transformer pass at a timestep in [0, 1] ------------------------------------------
    latents = QwenLatentCreator.create_noise(seed=cfg["seed"], height=cfg["height"], width=cfg["width"])
    timestep = 0.75
    noise = transformer(
        t=timestep, config=config, hidden_states=latents,
        encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask,
    )
    mx.eval(noise)

    # --- reference: the whole loop with classifier-free guidance ----------------------------------------
    x = latents
    for t in range(cfg["steps"]):
        positive = transformer(t=t, config=config, hidden_states=x, encoder_hidden_states=prompt_embeds, encoder_hidden_states_mask=prompt_mask)
        negative = transformer(t=t, config=config, hidden_states=x, encoder_hidden_states=negative_embeds, encoder_hidden_states_mask=negative_mask)
        guided = QwenImage.compute_guided_noise(positive, negative, config.guidance)
        x = config.scheduler.step(noise=guided, timestep=t, latents=x)
        mx.eval(x)
    final_latents = x

    # --- reference: VAE decode -----------------------------------------------------------------------------
    unpacked = QwenLatentCreator.unpack_latents(latents=final_latents, height=cfg["height"], width=cfg["width"])
    decoded = VAEUtil.decode(vae=vae, latent=unpacked, tiling_config=None)
    mx.eval(decoded)

    # --- save the checkpoint the way mflux does (component folders, shards, metadata) --------------------
    for name, module in (("transformer", transformer), ("text_encoder", text_encoder), ("vae", vae)):
        ModelSaver._save_weights(str(out), BITS, module, name)

    mx.save_safetensors(str(out / "references.safetensors"), {
        "input_ids": input_ids,
        "prompt_embeds": prompt_embeds,
        "negative_input_ids": negative_input_ids,
        "negative_prompt_embeds": negative_embeds,
        "latents": latents,
        "timestep": mx.array([timestep], dtype=mx.float32),
        "noise": noise,
        "sigmas": sigmas,
        "final_latents": final_latents,
        "decoded": decoded,
    })
    with open(out / "fixture.json", "w") as f:
        json.dump({**cfg, "bits": BITS}, f, indent=2)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"fixture written to {out} ({total / 1e6:.1f} MB)")
    print("sigmas:", [round(float(s), 5) for s in sigmas.tolist()])
    print("prompt_embeds", prompt_embeds.shape, prompt_embeds.dtype, "| noise", noise.shape, noise.dtype, "| decoded", decoded.shape,
          f"range [{float(decoded.min()):.3f}, {float(decoded.max()):.3f}]")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "qwen-image-fixture"))
