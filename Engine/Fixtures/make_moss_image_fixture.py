#!/usr/bin/env python3
"""Builds a small MOSS-Image checkpoint with random weights in the layout of OpenMOSS's repositories
(diffusers folders: transformer/, text_encoder/ with a Qwen3.5 vision tower nobody reads, vae/ with
BFL's small decoder, scheduler/), and what OpenMOSS's own pipeline computes from it in float32. The
native engine's `verify` reads the same folders, so the keys, the Qwen3.5 encoder (gated delta net
and gated attention), the DiT, the snapped schedule, guidance with block 22 skipped and the decode
are checked at once.

MOSS-Image has no MLX reference: its reference is `pipeline_moss_image.py`, the custom diffusers
pipeline in https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT (Apache 2.0, revision fc5de0d5),
run with transformers' `Qwen3_5Model` (Engine/Fixtures/requirements-moss.txt):
  $PY Engine/Fixtures/make_moss_image_fixture.py <pipeline_moss_image.py> <out-dir>

Everything is tiny except the VAE, whose encoder widths the engine's FLUX.2 VAE fixes; the DiT keeps
24 blocks so that block 22 exists to be skipped.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5Model

CONFIG = {
    "family": "moss-image",
    "transformer": {
        "dim": 128, "ffn_dim": 256, "num_heads": 2, "num_layers": 24,
        "text_dim": 128, "freq_dim": 256, "latent_channels": 128, "rope_base": 10000.0,
    },
    "text_encoder": {
        "vocab_size": 512, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 8,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 64, "linear_value_head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "full_attention_interval": 4,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000.0, "partial_rotary_factor": 0.25,
                            "mrope_section": [2, 2, 4], "mrope_interleaved": True},
    },
    # Lengths around the chunked delta rule's 64: one short, one over two chunks.
    "prompt_lengths": [21, 150],
    "negative_length": 9,
    "height": 96,
    "width": 64,
    "steps": 4,
    "guidance": 3.0,
    "timestep": 873.0,
    "schedule_steps": [1, 2, 4, 8, 20, 28, 50, 100],
    "seed": 7,
}


def load_reference(path: Path):
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def randomize(module: torch.nn.Module, seed: int, scale: float) -> None:
    """Small normal weights; norms near their identity (zero-centred Qwen3.5 norms near 0), the
    delta net's decay parameters in their trained range."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, value in module.named_parameters():
            noise = torch.randn(value.shape, generator=generator)
            if name.endswith("A_log"):
                value.copy_(torch.log(torch.rand(value.shape, generator=generator) * 15 + 1))
            elif name.endswith("dt_bias"):
                value.copy_(noise * 0.5)
            elif "norm" in name and value.ndim == 1:
                base = 0.0 if name.startswith(("language_model", "model.language_model")) and "linear_attn.norm" not in name else 1.0
                value.copy_(base + 0.1 * noise)
            else:
                fan_in = value.shape[1] if value.ndim > 1 else value.shape[0]
                value.copy_(noise * scale / np.sqrt(max(fan_in, 1)))


def main(reference_path: Path, out: Path) -> None:
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    reference = load_reference(reference_path)
    out.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG

    # The transformer.
    transformer = reference.MOSSImageTransformer(**cfg["transformer"]).float().eval()
    randomize(transformer, 1, 1.0)
    transformer.save_pretrained(out / "transformer", safe_serialization=True)

    # Qwen3.5 with a small vision tower, saved as the repositories have it (language_model.*, visual.*).
    text = dict(cfg["text_encoder"])
    text["layer_types"] = ["full_attention" if (i + 1) % text["full_attention_interval"] == 0 else "linear_attention"
                           for i in range(text["num_hidden_layers"])]
    encoder_config = Qwen3_5Config(
        text_config=text,
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                       "out_hidden_size": text["hidden_size"], "patch_size": 16, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "num_position_embeddings": 64},
    )
    encoder_config._attn_implementation = "sdpa"
    encoder = Qwen3_5Model(encoder_config).float().eval()
    randomize(encoder, 2, 1.0)
    encoder.save_pretrained(out / "text_encoder", safe_serialization=True)

    # FLUX.2's VAE with BFL's small decoder.
    vae = AutoencoderKLFlux2(decoder_block_out_channels=(96, 192, 384, 384)).float().eval()
    randomize(vae, 3, 0.3)  # milder than 1: a random decoder at full scale turns bf16 rounding into 2% of the image
    with torch.no_grad():
        vae.bn.running_mean.copy_(torch.randn(vae.bn.running_mean.shape) * 0.1)
        vae.bn.running_var.copy_(torch.rand(vae.bn.running_var.shape) + 0.5)
    vae.save_pretrained(out / "vae", safe_serialization=True)

    scheduler = reference.MOSSImageScheduler()
    scheduler.save_pretrained(out / "scheduler")

    refs = {}
    generator = torch.Generator().manual_seed(cfg["seed"])

    # 1. The encoder's last hidden state for random token ids (and every layer's output).
    def encode(ids, tag):
        captured = {}
        hooks = [layer.register_forward_hook(lambda m, i, o, n=n: captured.__setitem__(n, o[0] if isinstance(o, tuple) else o))
                 for n, layer in enumerate(encoder.language_model.layers)]
        with torch.no_grad():
            hidden = encoder(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        for hook in hooks:
            hook.remove()
        refs[f"{tag}_input_ids"] = ids.to(torch.int32)
        refs[f"{tag}_hidden"] = hidden
        for n, state in captured.items():
            refs[f"{tag}_layer_{n}"] = state
        return hidden

    prompts = []
    for index, length in enumerate(cfg["prompt_lengths"]):
        ids = torch.randint(0, text["vocab_size"], (1, length), generator=generator)
        prompts.append(encode(ids, f"prompt{index}"))
    negative_ids = torch.randint(0, text["vocab_size"], (1, cfg["negative_length"]), generator=generator)
    negative = encode(negative_ids, "negative")
    prompt = prompts[0]

    # 2. The schedule for several step counts.
    for steps in cfg["schedule_steps"]:
        s = reference.MOSSImageScheduler()
        s.set_timesteps(steps)
        refs[f"sigmas_{steps}"] = s.sigmas.float()
        refs[f"timesteps_{steps}"] = s.timesteps.float()

    # 3. One forward, plain and with block 22 skipped, on the reference's own conditioning.
    h, w = cfg["height"] // 16, cfg["width"] // 16
    latents = torch.randn((1, 128, h, w), generator=generator, dtype=torch.float32)
    refs["noise"] = latents
    t = torch.tensor(cfg["timestep"])
    with torch.no_grad():
        refs["forward"] = transformer(latents, t, prompt).sample
        refs["forward_long"] = transformer(latents, t, prompts[1]).sample
        refs["forward_skip"] = transformer(latents, t, negative, skip_layers=[22]).sample
        refs["rope"] = transformer.rope(h, w, prompt.shape[1], latents.device)

    # 4. The whole pipeline from that noise, with guidance (negative branch skipping 22) and without.
    pipe = reference.MOSSImagePipeline(transformer, reference.MOSSImageScheduler(), None, None, vae)
    with torch.no_grad():
        common = dict(height=cfg["height"], width=cfg["width"], num_inference_steps=cfg["steps"], latents=latents,
                      prompt_embeds=prompt, return_dict=False)
        guided = pipe(**common, guidance_scale=cfg["guidance"], negative_prompt_embeds=negative, output_type="latent")[0]
        unguided = pipe(**common, guidance_scale=1.0, output_type="latent")[0]
        image = pipe(**common, guidance_scale=cfg["guidance"], negative_prompt_embeds=negative, output_type="pt")[0]
    refs["latents"] = guided
    refs["latents_no_guidance"] = unguided
    refs["image"] = image  # [1, 3, H, W] in [0, 1]
    # The same decode with the VAE in bf16, as the engine runs it: what rounding alone moves.
    vae16 = AutoencoderKLFlux2.from_pretrained(out / "vae", torch_dtype=torch.bfloat16).eval()
    with torch.no_grad():
        z = guided.to(torch.bfloat16)
        bn = vae16.bn
        z = z * (bn.running_var + vae16.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, zh, zw = z.shape
        z = z.reshape(b, c // 4, 2, 2, zh, zw).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * zh, 2 * zw)
        decoded = vae16.decode(z, return_dict=False)[0].clamp(-1, 1).float()
    refs["image_bf16"] = (decoded / 2 + 0.5).clamp(0, 1)

    save_file({k: v.detach().contiguous().float() if v.dtype.is_floating_point else v.contiguous() for k, v in refs.items()},
              str(out / "references.safetensors"))
    with open(out / "fixture.json", "w") as f:
        json.dump({**cfg, "reference": "pipeline_moss_image.py (OpenMOSS-Team/MOSS-Image-SFT fc5de0d5)"}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
