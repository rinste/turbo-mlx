#!/usr/bin/env python3
"""Builds a small FLUX.2 Klein checkpoint with random weights, in mflux's format, plus the outputs
mflux computes from it: the native engine's `verify` compares its own outputs against them, so the
Swift port is checked module by module before a real checkpoint and a real Mac are involved.

Run it with the app's Python (it has mflux):
  ~/Library/Application\\ Support/TurboMLX/venv/bin/python Engine/Fixtures/make_klein_fixture.py <out-dir>

The model is tiny (a few MB) except for the VAE, whose channel counts the architecture fixes; the
text encoder keeps 28 layers because Klein reads the hidden states of layers 9, 18 and 27.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler

BITS = 4  # the catalog's checkpoints are 4-bit

CONFIG = {
    "variant": "fixture",
    "transformer": {
        "num_layers": 1,
        "num_single_layers": 2,
        "num_attention_heads": 2,        # inner_dim 256, head_dim 128
        "joint_attention_dim": 3 * 128,  # three hidden states of the text encoder below
    },
    "text_encoder": {
        "vocab_size": 512,
        "hidden_size": 128,
        "num_hidden_layers": 28,
        "num_attention_heads": 4,        # 4 * head_dim 128 = 512 wide projections
        "num_key_value_heads": 2,
        "intermediate_size": 256,
    },
    "text_encoder_out_layers": [9, 18, 27],
    "max_sequence_length": 16,
    "height": 128,
    "width": 128,
    "steps": 4,
    "seed": 7,
}


class _Float32Output(nn.Module):
    """Wraps the token embedding so everything after it runs in float32."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def __call__(self, ids: mx.array) -> mx.array:
        return self.inner(ids).astype(mx.float32)


def randomize(module: nn.Module, seed: int, scale: float) -> None:
    """Small normal weights (norms at 1 + noise), so activations stay in a sane range."""
    key = mx.random.key(seed)
    updates = []
    for name, value in tree_flatten(module.parameters()):
        key, sub = mx.random.split(key)
        if name.endswith("inv_freq") or name.endswith("running_mean") or name.endswith("running_var"):
            continue
        if value.ndim == 1 and ("norm" in name or name.endswith(".bias")):
            noise = 0.05 * mx.random.normal(value.shape, key=sub)
            new = (mx.ones(value.shape) + noise) if "norm" in name else noise
        else:
            new = scale * mx.random.normal(value.shape, key=sub) / (value.shape[-1] ** 0.5 if value.ndim > 1 else 1)
        updates.append((name, new.astype(mx.bfloat16 if value.ndim > 1 else mx.float32)))
    module.update(tree_unflatten(updates))
    mx.eval(module.parameters())


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG

    transformer = Flux2Transformer(**cfg["transformer"])
    text_encoder = Qwen3TextEncoder(**cfg["text_encoder"])
    vae = Flux2VAE()
    randomize(transformer, seed=1, scale=1.0)
    randomize(text_encoder, seed=2, scale=1.0)
    randomize(vae, seed=3, scale=1.0)
    # Realistic latent statistics for the transformer → VAE hand-over.
    vae.bn.running_mean = 0.1 * mx.random.normal((128,), key=mx.random.key(4))
    vae.bn.running_var = 1.0 + 0.1 * mx.random.uniform(shape=(128,), key=mx.random.key(5))
    mx.eval(vae.bn.running_mean, vae.bn.running_var)

    # Quantize exactly as mflux does for a saved checkpoint (every Linear and Embedding, group 64).
    for module in (transformer, text_encoder, vae):
        nn.quantize(module, group_size=64, bits=BITS, class_predicate=lambda path, m: hasattr(m, "to_quantized"))
    mx.eval(transformer.parameters(), text_encoder.parameters(), vae.parameters())

    # --- reference: text encoder -------------------------------------------------------------
    seq = cfg["max_sequence_length"]
    input_ids = mx.array([[5, 17, 42, 99, 7, 3, 250, 11, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=mx.int32)[:, :seq]
    attention_mask = (input_ids != 0).astype(mx.int32)
    prompt_embeds = text_encoder.get_prompt_embeds(
        input_ids=input_ids, attention_mask=attention_mask, hidden_state_layers=tuple(cfg["text_encoder_out_layers"])
    )
    text_ids = Flux2PromptEncoder.prepare_text_ids(prompt_embeds)
    mx.eval(prompt_embeds, text_ids)

    # The same encoder with float32 activations: two correct implementations then agree to float32
    # rounding, so `verify` can tell a wrong operation from bf16 rounding carried through 28 layers.
    embed_tokens = text_encoder.embed_tokens
    text_encoder.embed_tokens = _Float32Output(embed_tokens)
    prompt_embeds_f32 = text_encoder.get_prompt_embeds(
        input_ids=input_ids, attention_mask=attention_mask, hidden_state_layers=tuple(cfg["text_encoder_out_layers"])
    )
    text_encoder.embed_tokens = embed_tokens
    mx.eval(prompt_embeds_f32)

    # --- reference: one transformer pass --------------------------------------------------------
    latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
        seed=cfg["seed"], height=cfg["height"], width=cfg["width"], batch_size=1
    )
    timestep = mx.array([0.75], dtype=mx.bfloat16)
    noise = transformer(
        hidden_states=latents, encoder_hidden_states=prompt_embeds, timestep=timestep,
        img_ids=latent_ids, txt_ids=text_ids, guidance=None,
    )
    mx.eval(noise)

    # --- reference: the scheduler and the whole denoising loop ----------------------------------
    class _Config:  # what FlowMatchEulerDiscreteScheduler reads
        num_inference_steps = cfg["steps"]
        model_config = None

    scheduler = FlowMatchEulerDiscreteScheduler(_Config())
    # Klein's model configs set requires_sigma_shift, so mflux's Config shifts the schedule by the
    # image's token count before generating; do the same.
    scheduler.set_image_seq_len(latent_height * latent_width)
    sigmas, timesteps = scheduler.sigmas, scheduler.timesteps
    x = latents
    for t in range(cfg["steps"]):
        pred = transformer(
            hidden_states=x, encoder_hidden_states=prompt_embeds, timestep=timesteps[t],
            img_ids=latent_ids, txt_ids=text_ids, guidance=None,
        )
        x = scheduler.step(noise=pred, timestep=t, latents=x, sigmas=sigmas)
        mx.eval(x)
    final_latents = x

    # --- reference: VAE decode -----------------------------------------------------------------
    packed = final_latents.reshape(1, latent_height, latent_width, final_latents.shape[-1]).transpose(0, 3, 1, 2)
    decoded = vae.decode_packed_latents(packed)
    mx.eval(decoded)

    # --- save the checkpoint the way mflux does (component folders, shards, metadata) ------------
    for name, module in (("transformer", transformer), ("text_encoder", text_encoder), ("vae", vae)):
        ModelSaver._save_weights(str(out), BITS, module, name)

    mx.save_safetensors(str(out / "references.safetensors"), {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_f32": prompt_embeds_f32,
        "text_ids": text_ids,
        "latents": latents,
        "latent_ids": latent_ids,
        "timestep": timestep,
        "noise": noise,
        "sigmas": sigmas,
        "timesteps": timesteps,
        "final_latents": final_latents,
        "decoded": decoded,
    })
    with open(out / "fixture.json", "w") as f:
        json.dump({**cfg, "bits": BITS, "latent_height": latent_height, "latent_width": latent_width}, f, indent=2)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"fixture written to {out} ({total / 1e6:.1f} MB)")
    print("sigmas:", [round(float(s), 5) for s in sigmas.tolist()])
    print("prompt_embeds", prompt_embeds.shape, prompt_embeds.dtype, "| noise", noise.shape, "| decoded", decoded.shape,
          f"range [{float(decoded.min()):.3f}, {float(decoded.max()):.3f}]")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "klein-fixture"))
