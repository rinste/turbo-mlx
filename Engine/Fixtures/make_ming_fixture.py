#!/usr/bin/env python3
"""Builds a small Ming-Image checkpoint with random weights, in mflux's format, plus the outputs
mflux computes from it: the native engine's `verify` compares its own outputs against them, so the
Swift port is checked module by module before a real checkpoint and a real Mac are involved.

Run it with the app's Python (it has mflux):
  ~/Library/Application\\ Support/TurboMLX/venv/bin/python Engine/Fixtures/make_ming_fixture.py <out-dir>

mflux keeps Ming's sizes in module-level constants, so they are overridden before anything is
built: the Ling MoE encoder becomes 3 layers of 128 with 16 experts (4 chosen from 2 of 4 groups),
the connector 2 layers of 128, the query grid 4 × 4, the S3-DiT two layers of 384, the RGBA
decoder stages of 16, 32 and 64 channels. The text encoder is quantized at 5 bits and the rest at
8, as the catalog's te5 checkpoint is; the heads stay in bf16. The image token ids are moved into
the small vocabulary.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

import mflux.models.ming_image.model.ming_text_encoder.ling_moe_encoder as ling
import mflux.models.ming_image.model.ming_text_encoder.ming_condition_encoder as condition
import mflux.models.ming_image.model.ming_text_encoder.ming_connector as connector_module
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.ming_image.latent_creator.ming_latent_creator import MingLatentCreator
from mflux.models.ming_image.model.ming_transformer.ming_transformer import MingTransformer
from mflux.models.ming_image.model.ming_vae.ming_vae import MingVAE
from mflux.models.ming_image.variants.ming_image import MingImage
from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_mid_block_3d import QwenImageMidBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm
from mflux.models.qwen.model.qwen_vae.qwen_image_up_block_3d import QwenImageUpBlock3D

BITS = 8               # the catalog's checkpoint: DiT, connector and VAE at 8 bits…
TEXT_ENCODER_BITS = 5  # …and the Ling encoder at 5 (te5)

CONFIG = {
    "family": "ming",
    "variant": "fixture",
    "encoder": {
        "hidden_size": 128,
        "num_layers": 3,
        "num_heads": 2,
        "num_kv_heads": 1,
        "vocab_size": 512,
        "dense_intermediate": 256,
        "num_experts": 16,
        "top_k": 4,
        "n_group": 4,
        "topk_group": 2,
        "moe_intermediate": 64,
    },
    "connector": {"hidden_size": 128, "num_layers": 2, "num_heads": 2, "num_kv_heads": 1, "intermediate": 256},
    "heads": {"query_grid": 4, "directvlm_layers": [1, 2, 3], "cap_feat_dim": 128, "dit_dim": 384},
    "transformer": {"dim": 384, "n_layers": 2, "n_refiner_layers": 1, "n_heads": 3, "cap_feat_dim": 128},
    "vae": {"base_dim": 16},
    "image_start_id": 509,
    "image_patch_id": 508,
    "image_end_id": 510,
    "height": 64,
    "width": 64,
    "steps": 4,
    "seed": 7,
    "guidance": 2.0,
}


def override_constants(cfg: dict) -> None:
    e, c, h = cfg["encoder"], cfg["connector"], cfg["heads"]
    ling.HIDDEN_SIZE = e["hidden_size"]
    ling.NUM_LAYERS = e["num_layers"]
    ling.NUM_HEADS = e["num_heads"]
    ling.NUM_KV_HEADS = e["num_kv_heads"]
    ling.VOCAB_SIZE = e["vocab_size"]
    ling.DENSE_INTERMEDIATE = e["dense_intermediate"]
    ling.NUM_EXPERTS = e["num_experts"]
    ling.TOP_K = e["top_k"]
    ling.N_GROUP = e["n_group"]
    ling.TOPK_GROUP = e["topk_group"]
    ling.MOE_INTERMEDIATE = e["moe_intermediate"]
    connector_module.HIDDEN_SIZE = c["hidden_size"]
    connector_module.NUM_LAYERS = c["num_layers"]
    connector_module.NUM_HEADS = c["num_heads"]
    connector_module.NUM_KV_HEADS = c["num_kv_heads"]
    connector_module.INTERMEDIATE = c["intermediate"]
    condition.HIDDEN_SIZE = e["hidden_size"]
    condition.NUM_LAYERS = e["num_layers"]
    condition.CONNECTOR_SIZE = c["hidden_size"]
    condition.QUERY_GRID = h["query_grid"]
    condition.DIRECTVLM_LAYERS = tuple(h["directvlm_layers"])
    condition.CAP_FEAT_DIM = h["cap_feat_dim"]
    condition.DIT_DIM = h["dit_dim"]
    condition.IMAGE_START_ID = cfg["image_start_id"]
    condition.IMAGE_PATCH_ID = cfg["image_patch_id"]
    condition.IMAGE_END_ID = cfg["image_end_id"]


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
        elif name.endswith("expert_bias"):
            new = 17.0 + 0.5 * mx.random.normal(value.shape, key=sub)  # the real biases sit near 17
        elif "norm" in name and value.ndim > 1 and value.shape[-1] == 1:
            new = mx.ones(value.shape) + 0.05 * mx.random.normal(value.shape, key=sub)  # the VAE's [C, 1, 1(, 1)] norms
        else:
            new = scale * mx.random.normal(value.shape, key=sub) / (value.shape[-1] ** 0.5 if value.ndim > 1 else 1)
        updates.append((name, new.astype(dtype)))
    module.update(tree_unflatten(updates))
    mx.eval(module.parameters())


def small_vae(base_dim: int) -> MingVAE:
    """mflux's RGBA VAE with its decoder rebuilt from the same blocks at a fraction of the width."""
    vae = MingVAE()
    d1, d2, d4 = base_dim, 2 * base_dim, 4 * base_dim
    decoder = vae.decoder
    decoder.conv_in = QwenImageCausalConv3D(16, d4, 3, 1, 1)
    decoder.mid_block = QwenImageMidBlock3D(d4, num_layers=1)
    decoder.up_block0 = QwenImageUpBlock3D(d4, d4, num_res_blocks=2, upsample_mode="upsample3d")
    decoder.up_block1 = QwenImageUpBlock3D(d2, d4, num_res_blocks=2, upsample_mode="upsample3d")
    decoder.up_block2 = QwenImageUpBlock3D(d2, d2, num_res_blocks=2, upsample_mode="upsample2d")
    decoder.up_block3 = QwenImageUpBlock3D(d1, d1, num_res_blocks=2, upsample_mode=None)
    decoder.norm_out = QwenImageRMSNorm(d1, images=False)
    decoder.conv_out = QwenImageCausalConv3D(d1, vae.image_channels, 3, 1, 1)
    vae.encoder = nn.Identity()  # never loaded by the engine; keeps the fixture small
    vae.quant_conv = nn.Identity()
    return vae


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG
    override_constants(cfg)

    text_encoder = ling.LingMoeEncoder()
    connector = connector_module.MingConnector()
    heads = condition.MingHeads()
    transformer = MingTransformer(**cfg["transformer"])
    vae = small_vae(cfg["vae"]["base_dim"])
    randomize(text_encoder, seed=1, scale=1.0)
    randomize(connector, seed=2, scale=1.0)
    randomize(heads, seed=3, scale=1.0)
    randomize(transformer, seed=4, scale=1.0)
    randomize(vae, seed=5, scale=1.0)

    # Quantize as mflux saves the te5 checkpoint (every Linear, Embedding and expert stack; the
    # routers have no to_quantized and the heads are skipped).
    predicate = lambda path, m: hasattr(m, "to_quantized")  # noqa: E731
    nn.quantize(text_encoder, group_size=64, bits=TEXT_ENCODER_BITS, class_predicate=predicate)
    for module in (connector, transformer, vae):
        nn.quantize(module, group_size=64, bits=BITS, class_predicate=predicate)
    mx.eval(text_encoder.parameters(), connector.parameters(), heads.parameters(), transformer.parameters(), vae.parameters())

    # --- reference: the text side (query tokens through the connector, prompt states through the head)
    prompt_ids = [(i * 41 + 7) % 500 + 1 for i in range(12)]
    cap_feats, cap_feats_2 = condition.MingConditionEncoder.encode(
        prompt_ids=prompt_ids, text_encoder=text_encoder, connector=connector, heads=heads
    )
    mx.eval(cap_feats, cap_feats_2)

    # --- reference: one transformer pass, conditional and with the zeroed condition ---------------------
    sigmas = MingImage.sigmas(cfg["steps"])
    latents = MingLatentCreator.create_noise(seed=cfg["seed"], height=cfg["height"], width=cfg["width"])
    timestep = mx.array([0.75], dtype=mx.float32)
    x = latents[0][:, None]
    noise = transformer(x, timestep, cap_feats, cap_feats_2)
    noise_unconditional = transformer(x, timestep, mx.zeros_like(cap_feats), mx.zeros_like(cap_feats_2))
    mx.eval(noise, noise_unconditional)

    # --- reference: the whole loop, guided (the last, sigma = 0 step is skipped) -----------------------
    predict = MingImage._predict(transformer)
    latents_t = latents
    for t in range(cfg["steps"]):
        sigma, sigma_next = sigmas[t], sigmas[t + 1]
        if sigma.item() > 0:
            velocity = predict(latents_t, 1.0 - sigma.reshape(1), cap_feats, cap_feats_2, cfg["guidance"])
            latents_t = latents_t + (sigma_next - sigma) * velocity.astype(mx.float32)
        mx.eval(latents_t)
    final_latents = latents_t

    # --- reference: VAE decode (RGBA, in bf16) -----------------------------------------------------------
    decoded = VAEUtil.decode(vae=vae, latent=final_latents.astype(mx.bfloat16), tiling_config=None)
    mx.eval(decoded)

    # --- save the checkpoint the way mflux does (component folders, shards, metadata) --------------------
    for name, bits, module in (
        ("mllm", TEXT_ENCODER_BITS, text_encoder),
        ("connector", BITS, connector),
        ("mlp", BITS, heads),
        ("transformer", BITS, transformer),
        ("vae", BITS, vae),
    ):
        ModelSaver._save_weights(str(out), bits, module, name)

    mx.save_safetensors(str(out / "references.safetensors"), {
        "prompt_ids": mx.array(prompt_ids, dtype=mx.int32),
        "cap_feats": cap_feats,
        "cap_feats_2": cap_feats_2,
        "latents": latents,
        "timestep": timestep,
        "noise": noise,
        "noise_unconditional": noise_unconditional,
        "sigmas": sigmas,
        "final_latents": final_latents,
        "decoded": decoded,
    })
    with open(out / "fixture.json", "w") as f:
        json.dump({**cfg, "bits": BITS, "text_encoder_bits": TEXT_ENCODER_BITS}, f, indent=2)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"fixture written to {out} ({total / 1e6:.1f} MB)")
    print("sigmas:", [round(float(s), 5) for s in sigmas.tolist()])
    print("cap_feats", cap_feats.shape, cap_feats.dtype, "| cap_feats_2", cap_feats_2.shape, "| noise", noise.shape, noise.dtype,
          "| decoded", decoded.shape, f"range [{float(decoded.min()):.3f}, {float(decoded.max()):.3f}]")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "ming-fixture"))
