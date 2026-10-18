#!/usr/bin/env python3
"""OpenMOSS's own MOSS-Image pipeline (`pipeline_moss_image.py`, from OpenMOSS-Team/MOSS-Image-SFT) on a
checkpoint's diffusers folders, to compare real images with the native engine. It runs as OpenMOSS
ships it — bf16 transformer and Qwen3.5 encoder, float32 VAE, CFG 3 with block 22 skipped in the
negative branch — on MPS (or the CPU), from the engine's starting noise: MLX's normal for the seed,
[1, H/16, W/16, 128], so the two images differ only by the implementations.

Run it with the Python of Engine/Fixtures/requirements-moss.txt (~30 GB of memory in bf16):
  $PY Engine/Reference/moss_image_reference.py <pipeline_moss_image.py> <snapshot> --prompt "…" \\
      --width 1024 --height 1024 --steps 50 --guidance 3 --seed 42 --output out.png [--record dir]

With `--record`, it also writes what it computed on the way to a folder `turbo-engine verify` reads
with the checkpoint's own weights: the token ids and last hidden states of the prompt and the
empty negative prompt, the noise, the first step's two velocities, the final latents, the image.
`$PY Engine/Reference/moss_image_reference.py --truth <record> <snapshot>` then adds the encoder's
float32 states (CPU, ~36 GB).
"""

import argparse
import json
import sys
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer


def add_truth(record: Path, snapshot: Path) -> None:
    """`--truth`: the prompt's and the negative prompt's last hidden states from Qwen3.5 in float32 on
    the CPU, added to a recording (`*_hidden_f32`): what both bf16 encoders are measured against."""
    from safetensors.torch import load_file

    refs = load_file(str(record / "references.safetensors"))
    encoder = AutoModel.from_pretrained(snapshot / "text_encoder", dtype=torch.float32).eval()
    with torch.no_grad():
        for tag in ("prompt", "negative"):
            ids = refs[f"{tag}_input_ids"].long()
            refs[f"{tag}_hidden_f32"] = encoder(input_ids=ids, use_cache=False).last_hidden_state.float()
    save_file({k: v.contiguous() for k, v in refs.items()}, str(record / "references.safetensors"))
    print(f"added float32 states to {record}")


def main():
    if "--truth" in sys.argv:
        _, _, record, snapshot = sys.argv
        return add_truth(Path(record), Path(snapshot))
    parser = argparse.ArgumentParser()
    parser.add_argument("pipeline", type=Path, help="pipeline_moss_image.py")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--float32", action="store_true", help="every component in float32 (the parity setting)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()

    # MPS has no float64: the timestep's sinusoid, which the pipeline computes in float64 on the
    # model's device, is computed on the CPU instead (the same numbers), then moved.
    source = args.pipeline.read_text()
    sinusoid = ("freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)\n"
                "        phase = times.double()[:, None] * freq\n"
                "        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()")
    assert sinusoid in source, "pipeline_moss_image.py changed: check the timestep sinusoid"
    source = source.replace(sinusoid, sinusoid.replace("device=times.device", "device='cpu'")
                            .replace("times.double()", "times.cpu().double()") + ".to(times.device)")
    reference = types.ModuleType("pipeline_moss_image")
    exec(compile(source, str(args.pipeline), "exec"), reference.__dict__)

    dtype = torch.float32 if args.float32 else torch.bfloat16
    root = args.snapshot
    transformer = reference.MOSSImageTransformer.from_pretrained(root / "transformer", torch_dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer")
    text_encoder = AutoModel.from_pretrained(root / "text_encoder", dtype=dtype).eval()
    vae = AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=torch.float32).eval()
    scheduler = reference.MOSSImageScheduler.from_pretrained(root / "scheduler")
    pipe = reference.MOSSImagePipeline(transformer, scheduler, text_encoder, tokenizer, vae).to(args.device)

    refs = {}
    prompt_embeds, _ = pipe.encode_prompt(args.prompt)
    negative_embeds, _ = pipe.encode_prompt(args.negative)
    for tag, text, embeds in (("prompt", args.prompt, prompt_embeds), ("negative", args.negative, negative_embeds)):
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        refs[f"{tag}_input_ids"] = torch.tensor([tokenizer(formatted)["input_ids"]], dtype=torch.int32)
        refs[f"{tag}_hidden"] = embeds.float().cpu()

    # The engine's noise: MLX's normal for the seed, channels last, in the reference's layout.
    h, w = args.height // 16, args.width // 16
    noise = np.array(mx.random.normal((1, h, w, 128), key=mx.random.key(args.seed)))
    latents = torch.from_numpy(noise).permute(0, 3, 1, 2).contiguous()
    refs["noise"] = latents.clone()

    # The first step's two forwards, as the loop calls them.
    calls = []
    forward = transformer.forward

    def recording(*a, **k):
        out = forward(*a, **k)
        if len(calls) < 2:
            calls.append(out.sample.float().cpu())
        return out

    transformer.forward = recording
    with torch.no_grad():
        final = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, height=args.height,
                     width=args.width, num_inference_steps=args.steps, guidance_scale=args.guidance,
                     latents=latents.to(args.device), output_type="latent", return_dict=False)[0]
    transformer.forward = forward
    refs["step0_cond"], refs["step0_uncond"] = calls[0], calls[1] if len(calls) > 1 else calls[0]
    refs["latents"] = final.float().cpu()

    # The pipeline's own decode, from the latents.
    with torch.no_grad():
        z = final.to(vae.dtype)
        bn, eps = vae.bn, vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, zh, zw = z.shape
        z = z.reshape(b, c // 4, 2, 2, zh, zw).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * zh, 2 * zw)
        image = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    pixels = ((image[0].float().cpu() / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(pixels).save(args.output)
    refs["image"] = torch.from_numpy(pixels.astype(np.float32))
    print(f"wrote {args.output}")

    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in refs.items()}, str(args.record / "references.safetensors"))
        meta = {"family": "moss-image", "pack": str(root.resolve()), "prompt": args.prompt, "negative": args.negative,
                "width": args.width, "height": args.height, "steps": args.steps, "guidance": args.guidance,
                "seed": args.seed, "dtype": "float32" if args.float32 else "bfloat16", "device": args.device}
        (args.record / "fixture.json").write_text(json.dumps(meta, indent=2))
        print(f"recorded {args.record}")


if __name__ == "__main__":
    main()
