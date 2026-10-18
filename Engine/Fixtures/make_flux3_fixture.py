#!/usr/bin/env python3
"""Builds a small FLUX 3 Image Klein checkpoint with random weights, saved in the layout of BFL's
own repository (`black-forest-labs/flux-3-image-klein`: diffusers folders, Qwen3-VL's keys with a
vision tower and more layers than are read), and what mflux computes from it once it has read it
the way it reads that repository, quantized to 8 bits on loading as `mflux-generate-flux3 -q 8`
does. The native engine's `verify` reads the same folder through its own loader, so the key
mapping, the bfloat16 cast, the quantization and every stage of the pipeline are checked at once.

Run it with a Python that has mflux (Engine/Fixtures/requirements.txt):
  $PY Engine/Fixtures/make_flux3_fixture.py <out-dir>

The model is tiny (a few MB) except for the VAE, whose channel counts the architecture fixes; the
text encoder keeps 32 layers because FLUX 3 reads the hidden states of every fourth up to 32.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import PIL.Image
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.latent_creator.flux3_reference_images import Flux3ReferenceImages
from mflux.models.flux3.model.flux3_scheduler.flux3_flow_map_scheduler import Flux3FlowMapScheduler
from mflux.models.flux3.model.flux3_text_encoder.prompt_encoder import Flux3PromptEncoder
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.models.flux3.variants.flux3_klein_core import _Flux3KleinCore
from mflux.models.flux3.weights.flux3_weight_definition import (
    Flux3KleinBaseWeightDefinition,
    Flux3KleinWeightDefinition,
)

BITS = 8  # the catalog quantizes BFL's bf16 checkpoint to 8 bits on loading

CONFIG = {
    "family": "flux3-klein",
    "variant": "fixture",
    "transformer": {
        "num_layers": 2,
        "num_single_layers": 2,
        "num_attention_heads": 2,        # inner_dim 256, head_dim 128 (the rotary's four 32-wide axes)
        "joint_attention_dim": 8 * 128,  # eight hidden states of the text encoder below
    },
    "text_encoder": {
        "vocab_size": 512,
        "hidden_size": 128,
        "num_hidden_layers": 32,
        "num_attention_heads": 4,        # 4 * head_dim 128 = 512 wide projections
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "rope_theta": 5000000.0,
    },
    "saved_text_layers": 34,             # the checkpoint has two layers more, which nobody reads
    "text_encoder_out_layers": list(Flux3PromptEncoder.OUTPUT_LAYERS),
    "pad_multiple": Flux3PromptEncoder.PAD_MULTIPLE,
    "height": 128,
    "width": 96,
    "steps": 4,
    "seed": 7,
    "guidance": 3.5,
}


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


def save_original_layout(out: Path, transformer: nn.Module, text_encoder: nn.Module, vae: nn.Module) -> None:
    """The modules' tensors under the names and in the shapes of BFL's repository, all bf16."""
    def tensors(module):
        return {name: value for name, value in tree_flatten(module.parameters()) if not name.endswith("inv_freq")}

    t = {name.replace(".attn.to_out.", ".attn.to_out.0."): value for name, value in tensors(transformer).items()}
    e = {f"model.language_model.{name}": value for name, value in tensors(text_encoder).items()}
    # What BFL's text encoder also holds, which a reader must leave out.
    e["model.visual.patch_embed.proj.weight"] = mx.zeros((64, 3, 2, 16, 16))
    e["lm_head.weight"] = mx.zeros((512, 128))
    v = {}
    for name, value in tensors(vae).items():
        name = name.replace(".to_out.", ".to_out.0.")
        v[name] = value.transpose(0, 3, 1, 2) if value.ndim == 4 else value  # MLX's OHWI → PyTorch's OIHW
    v["bn.num_batches_tracked"] = mx.array(0, dtype=mx.int32)
    for folder, file, arrays in (
        ("transformer", "diffusion_pytorch_model.safetensors", t),
        ("text_encoder", "model.safetensors", e),
        ("vae", "diffusion_pytorch_model.safetensors", v),
    ):
        (out / folder).mkdir(parents=True, exist_ok=True)
        cast = {k: a.astype(mx.bfloat16) if mx.issubdtype(a.dtype, mx.floating) else a for k, a in arrays.items()}
        mx.save_safetensors(str(out / folder / file), cast)


def load_as_mflux(out: Path, base: bool) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Reads the folder as `Flux3Initializer` reads BFL's repository, quantizing on loading."""
    cfg = CONFIG
    definition = Flux3KleinBaseWeightDefinition if base else Flux3KleinWeightDefinition
    weights = WeightLoader.load(weight_definition=definition, model_path=str(out))
    transformer = Flux3Transformer(**cfg["transformer"], flow_map=not base)
    text_encoder = Qwen3TextEncoder(**cfg["text_encoder"])
    vae = Flux2VAE()
    WeightApplier.apply_and_quantize(
        weights=weights, quantize_arg=BITS, weight_definition=definition,
        models={"vae": vae, "transformer": transformer, "text_encoder": text_encoder},
    )
    mx.eval(transformer.parameters(), text_encoder.parameters(), vae.parameters())
    return transformer, text_encoder, vae


class _Float32Output(nn.Module):
    """Wraps the token embedding so everything after it runs in float32."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def __call__(self, ids: mx.array) -> mx.array:
        return self.inner(ids).astype(mx.float32)


def picture(width: int, height: int, seed: int) -> PIL.Image.Image:
    """A smooth pattern with some noise, as 8-bit RGB."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    channels = [np.sin(3 * xx / width + 2 * yy / height + p) for p in (0.0, 2.0, 4.0)]
    values = 0.5 + 0.35 * np.stack(channels, axis=-1) + 0.1 * rng.standard_normal((height, width, 3))
    return PIL.Image.fromarray(np.clip(values * 255, 0, 255).astype(np.uint8))


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG

    generated_transformer = Flux3Transformer(**cfg["transformer"])
    generated_text_encoder = Qwen3TextEncoder(**{**cfg["text_encoder"], "num_hidden_layers": cfg["saved_text_layers"]})
    generated_vae = Flux2VAE()
    randomize(generated_transformer, seed=1, scale=1.0)
    randomize(generated_text_encoder, seed=2, scale=1.0)
    randomize(generated_vae, seed=3, scale=1.0)
    generated_vae.bn.running_mean = 0.1 * mx.random.normal((128,), key=mx.random.key(4))
    generated_vae.bn.running_var = 1.0 + 0.1 * mx.random.uniform(shape=(128,), key=mx.random.key(5))
    save_original_layout(out, generated_transformer, generated_text_encoder, generated_vae)

    transformer, text_encoder, vae = load_as_mflux(out, base=False)

    # --- reference: the prompt, as Flux3PromptEncoder pads and positions it ----------------------
    real = [5, 17, 42, 99, 7, 3, 250, 11, 64, 300, 12]
    length = -(-len(real) // cfg["pad_multiple"]) * cfg["pad_multiple"]
    input_ids = mx.array([real + [0] * (length - len(real))], dtype=mx.int32)
    attention_mask = mx.array([[1] * len(real) + [0] * (length - len(real))], dtype=mx.int32)
    positions = Flux3PromptEncoder.positions(attention_mask)
    layers = tuple(cfg["text_encoder_out_layers"])
    prompt_embeds = text_encoder.get_prompt_embeds(
        input_ids=input_ids, attention_mask=attention_mask, hidden_state_layers=layers, position_ids=positions
    ).astype(mx.bfloat16)
    text_ids = Flux3PromptEncoder.prepare_text_ids(prompt_embeds.shape[1])
    embed_tokens = text_encoder.embed_tokens
    text_encoder.embed_tokens = _Float32Output(embed_tokens)
    prompt_embeds_f32 = text_encoder.get_prompt_embeds(
        input_ids=input_ids, attention_mask=attention_mask, hidden_state_layers=layers, position_ids=positions
    )
    text_encoder.embed_tokens = embed_tokens
    # The empty prompt the Base's guidance runs against: its template alone, padded the same way.
    empty_real = [5, 17, 3]
    empty_ids = mx.array([empty_real + [0] * (cfg["pad_multiple"] - len(empty_real))], dtype=mx.int32)
    empty_mask = mx.array([[1] * len(empty_real) + [0] * (cfg["pad_multiple"] - len(empty_real))], dtype=mx.int32)
    empty_embeds = text_encoder.get_prompt_embeds(
        input_ids=empty_ids, attention_mask=empty_mask, hidden_state_layers=layers,
        position_ids=Flux3PromptEncoder.positions(empty_mask),
    ).astype(mx.bfloat16)
    mx.eval(prompt_embeds, prompt_embeds_f32, empty_embeds, positions, text_ids)

    # --- reference: noise, ids, one pass, the schedule and the loop -----------------------------
    h, w = Flux3LatentCreator.latent_shape(cfg["height"], cfg["width"])
    latents = Flux3LatentCreator.create_noise(cfg["seed"], cfg["height"], cfg["width"])
    latent_ids = Flux3LatentCreator.prepare_ids(h, w)
    scheduler = Flux3FlowMapScheduler(num_inference_steps=cfg["steps"])
    timesteps = scheduler.timesteps
    predict = _Flux3KleinCore._predict(transformer)

    def velocity(x, t, cond_tokens=None, cond_ids=None, model=None, guidance=None, uncond=None):
        fn = predict if model is None else _Flux3KleinCore._predict(model)
        flow_map = (model or transformer).flow_map
        return fn(
            latents=x, latent_ids=latent_ids, text_tokens=prompt_embeds, text_ids=text_ids,
            uncond_tokens=None if uncond is None else uncond[0], uncond_ids=None if uncond is None else uncond[1],
            cond_tokens=cond_tokens, cond_ids=cond_ids, guidance=guidance,
            timestep=timesteps[t : t + 1], target_timestep=timesteps[t + 1 : t + 2] if flow_map else None,
        )

    noise = velocity(latents, 1)
    first_noise = velocity(latents, 0)
    x = latents
    states = []
    for t in range(cfg["steps"]):
        x = scheduler.step(latents=x, velocity=velocity(x, t), step=t)
        mx.eval(x)
        states.append(x)
    final_latents = x
    # Each step's time embeddings: the image's (its time plus the target's) and the context's.
    times = {}
    for i in range(cfg["steps"]):
        t, target = timesteps[i : i + 1], timesteps[i + 1 : i + 2]
        times[f"time_image_{i}"] = transformer.timestep_embedder(t) + transformer.target_timestep_embedder(target)
        times[f"time_context_{i}"] = transformer.timestep_embedder(mx.zeros_like(t)) + transformer.target_timestep_embedder(target)
    # The last step's velocity, from the state before it: the step towards t = 0.
    last_noise = velocity(states[-2], cfg["steps"] - 1)
    decoded = Flux3LatentCreator.decode_tokens(vae, final_latents, h, w)
    mx.eval(latents, latent_ids, noise, final_latents, decoded)

    # --- reference: an edit from two pictures, then one padded picture --------------------------
    # Two pictures (each cut to multiples of 16, one scaled down past a megapixel) at layers −1, −2.
    pictures = [picture(150, 100, 1), picture(1100, 1000, 2)]
    multi, _ = Flux3ReferenceImages.prepare(pictures)

    def encode_references(references):
        tokens, ids = [], []
        for index, reference in enumerate(references):
            image = mx.array(Flux3ReferenceImages.to_array(reference))
            tokens.append(Flux3LatentCreator.encode_tokens(vae, image).astype(mx.bfloat16))
            rh, rw = Flux3LatentCreator.latent_shape(reference.height, reference.width)
            ids.append(Flux3LatentCreator.prepare_ids(rh, rw, layer=-(1 + index)))
        return mx.concatenate(tokens, axis=1), mx.concatenate(ids, axis=0)

    cond_tokens, cond_ids = encode_references(multi)
    edit_noise = velocity(latents, 1, cond_tokens, cond_ids)
    x = latents
    for t in range(cfg["steps"]):
        x = scheduler.step(latents=x, velocity=velocity(x, t, cond_tokens, cond_ids), step=t)
        mx.eval(x)
    edit_final_latents = x
    edit_decoded = Flux3LatentCreator.decode_tokens(vae, edit_final_latents, h, w)
    mx.eval(cond_tokens, cond_ids, edit_noise, edit_decoded)
    # One picture whose fitted size is not a multiple of 16: padded by reflection, its box recorded.
    single, box = Flux3ReferenceImages.prepare([picture(90, 70, 3)])

    # --- reference: the Base, guided against the empty prompt ------------------------------------
    base_transformer, _, _ = load_as_mflux(out, base=True)
    empty_text_ids = Flux3PromptEncoder.prepare_text_ids(empty_embeds.shape[1])
    base_noise = velocity(latents, 1, model=base_transformer, guidance=cfg["guidance"], uncond=(empty_embeds, empty_text_ids))
    mx.eval(base_noise)

    # The sizes pictures are prepared at: one picture (fit, canvas) and one of several (cut).
    canvas_sizes = [
        [w_, h_, *Flux3ReferenceImages.reference_canvas(w_, h_)[0], *Flux3ReferenceImages.reference_canvas(w_, h_)[1]]
        for w_, h_ in [(1024, 1024), (1000, 700), (4032, 3024), (3000, 17), (17, 3000), (90, 70), (2049, 2047), (8000, 300)]
    ]
    multi_sizes = []
    for w_, h_ in [(1024, 1024), (1000, 700), (4032, 3024), (1100, 1000), (150, 100), (1025, 1023)]:
        prepared = Flux3ReferenceImages.prepare_multi_reference(PIL.Image.new("RGB", (w_, h_)))
        multi_sizes.append([w_, h_, prepared.width, prepared.height])

    for index, image in enumerate(pictures):
        image.save(out / f"picture{index + 1}.png")
    picture(90, 70, 3).save(out / "picture3.png")

    mx.save_safetensors(str(out / "references.safetensors"), {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions": positions,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_f32": prompt_embeds_f32,
        "text_ids": text_ids,
        "empty_ids": empty_ids,
        "empty_mask": empty_mask,
        "empty_embeds": empty_embeds,
        "latents": latents,
        "latent_ids": latent_ids,
        "noise": noise,
        "first_noise": first_noise,
        "last_noise": last_noise,
        **times,
        **{f"latents_{i + 1}": state for i, state in enumerate(states)},
        "timesteps": timesteps,
        "final_latents": final_latents,
        "decoded": decoded,
        "multi1": mx.array(np.asarray(multi[0])),
        "multi2": mx.array(np.asarray(multi[1])),
        "cond_tokens": cond_tokens,
        "cond_ids": cond_ids,
        "edit_noise": edit_noise,
        "edit_final_latents": edit_final_latents,
        "edit_decoded": edit_decoded,
        "single": mx.array(np.asarray(single[0])),
        "base_noise": base_noise,
    })
    with open(out / "fixture.json", "w") as f:
        json.dump({**cfg, "bits": BITS, "latent_height": h, "latent_width": w, "single_box": list(box),
                   "canvas_sizes": canvas_sizes, "multi_sizes": multi_sizes}, f, indent=2)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"fixture written to {out} ({total / 1e6:.1f} MB)")
    print("timesteps:", [round(float(s), 5) for s in timesteps.tolist()])
    print("prompt_embeds", prompt_embeds.shape, "| noise", noise.shape, "| cond", cond_tokens.shape,
          "| decoded", decoded.shape, f"range [{float(decoded.min()):.3f}, {float(decoded.max()):.3f}]")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "flux3-fixture"))
