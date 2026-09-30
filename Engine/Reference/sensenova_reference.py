#!/usr/bin/env python3
"""SenseTime's own SenseNova-U1.5 code on an MLX pack's weights, to compare real images with the
native engine: the pack (e.g. mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit) is dequantized
tensor by tensor into bf16, its keys and convolutions put back in the original's layout, and the
reference's `t2i_generate` runs unchanged (PyTorch on MPS, or the CPU) with the tokenizer it builds
from the pack's `vocab.json` and `merges.txt`. The starting noise is the engine's: MLX's normal
for the seed, [1, H, W, 3], so the two images differ only by the implementations.

Run it with the Python of Engine/Fixtures/requirements-sensenova.txt (it needs ~40 GB of memory
for the bf16 weights):
  $PY Engine/Reference/sensenova_reference.py <SenseNova-U1 clone> <pack> --prompt "…" \
      --width 1024 --height 1024 --steps 8 --guidance 1 --seed 42 --output out.png [--record dir]

With `--record`, it also writes what it computed on the way, as make_sensenova_fixture.py does, to
a folder `turbo-engine verify` reads with the pack's own weights: the prompt's cache, the first
step's input and velocity, the last step's velocity, the image before it becomes pixels.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image


def pack_state_dict(pack: Path, device: str) -> dict:
    """The pack's tensors in the original's layout, bf16, on `device`."""
    import json

    config = json.loads((pack / "config.json").read_text())
    quantization = config.get("quantization") or {}
    tensors = {}
    for shard in sorted(pack.glob("model-*.safetensors")):
        tensors.update(mx.load(str(shard)))
    state = {}
    for key in sorted(tensors):
        if key.endswith(".scales") or key.endswith(".biases") or key.startswith("language_model.lm_head."):
            continue
        value = tensors[key]
        base = key[: -len(".weight")]
        if key.endswith(".weight") and f"{base}.scales" in tensors:
            value = mx.dequantize(value, tensors[f"{base}.scales"], tensors[f"{base}.biases"],
                                  group_size=quantization.get("group_size", 64), bits=quantization.get("bits", 4))
        array = np.array(value.astype(mx.float32))
        if array.ndim == 4:
            array = array.transpose(0, 3, 1, 2)
        name = key.replace("_embedder.mlp.1.", "_embedder.mlp.2.")
        state[name] = torch.from_numpy(array).to(torch.bfloat16).to(device)
        del tensors[key]
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--record", type=Path)
    # An edit of this picture (`it2i_generate`), prepared as the engine does: over white, resized
    # with Lanczos to about the image's area within 512² and 2048², sides multiples of 32.
    parser.add_argument("--image", type=Path)
    # Adds this much of another unit normal to the noise: how far a small change moves the image.
    parser.add_argument("--perturb", type=float, default=0.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo / "src"))
    import sensenova_u1
    from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOChatConfig
    from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel
    from sensenova_u1.models.neo_unify.modeling_qwen3 import Qwen3RotaryEmbedding
    from transformers import AutoTokenizer

    sensenova_u1.set_attn_backend("sdpa")
    import json

    config_json = json.loads((args.pack / "config.json").read_text())
    config_json.pop("quantization", None)
    config = NEOChatConfig(**config_json)
    with torch.device("meta"):
        model = NEOChatModel(config)
    state = pack_state_dict(args.pack, args.device)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    missing = [k for k in missing if not k.startswith("language_model.lm_head.")]
    assert not missing and not unexpected, (missing, unexpected)
    # The rotary tables are buffers built at construction: build them again on the device.
    for module in model.modules():
        if isinstance(module, Qwen3RotaryEmbedding):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, args.device)
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq
    model.language_model.lm_head = torch.nn.Identity()
    model.eval()

    with tempfile.TemporaryDirectory() as original:
        for file in ["vocab.json", "merges.txt", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json"]:
            (Path(original) / file).symlink_to((args.pack / file).resolve())
        tokenizer = AutoTokenizer.from_pretrained(original)

    # The engine's noise: MLX's normal for the seed, channels last → the reference's channels first.
    noise = np.array(mx.random.normal([1, args.height, args.width, 3], key=mx.random.key(args.seed)))
    if args.perturb:
        noise = noise + args.perturb * np.array(mx.random.normal(noise.shape, key=mx.random.key(args.seed + 1)))
    noise = torch.from_numpy(noise.transpose(0, 3, 1, 2).copy())
    randn = torch.randn

    def engine_noise(*shape_args, device=None, dtype=None, **_):
        return noise.to(device=device, dtype=dtype)

    records = {}
    if args.record:
        prefix_forward = model._t2i_prefix_forward

        def recording_prefix(input_ids, indexes, attention_mask):
            cache, hidden = prefix_forward(input_ids, indexes, attention_mask)
            tag = "cond" if "cond_input_ids" not in records else "uncond"
            records[f"{tag}_input_ids"] = input_ids.to(torch.int32)
            for index, layer in enumerate(cache.layers):
                records[f"{tag}_keys_{index}"] = layer.keys
                records[f"{tag}_values_{index}"] = layer.values
            return cache, hidden

        model._t2i_prefix_forward = recording_prefix
        predict = model._t2i_predict_v
        velocities = []

        def recording_predict(input_embeds, indexes, mask, cache, t, z, *rest, **kwargs):
            velocity = predict(input_embeds, indexes, mask, cache, t, z, *rest, **kwargs)
            velocities.append((input_embeds, velocity, z))
            return velocity

        model._t2i_predict_v = recording_predict

    picture = None
    if args.image:
        from sensenova_u1.models.neo_unify.utils import smart_resize

        picture = Image.open(args.image)
        if picture.mode == "RGBA":
            background = Image.new("RGB", picture.size, (255, 255, 255))
            background.paste(picture, mask=picture.split()[3])
            picture = background
        picture = picture.convert("RGB")
        area = min(max(args.width * args.height, 512 * 512), 2048 * 2048)
        height, width = smart_resize(picture.height, picture.width, factor=32, min_pixels=area, max_pixels=area)
        if (width, height) != picture.size:
            picture = picture.resize((width, height), Image.LANCZOS)

    torch.randn = engine_noise
    try:
        with torch.inference_mode():
            if picture is None:
                image = model.t2i_generate(
                    tokenizer, args.prompt, cfg_scale=args.guidance, timestep_shift=args.shift, cfg_norm="none",
                    image_size=(args.width, args.height), num_steps=args.steps, seed=args.seed,
                )
            else:
                image = model.it2i_generate(
                    tokenizer, args.prompt, [picture], cfg_scale=args.guidance, img_cfg_scale=1.0,
                    timestep_shift=args.shift, cfg_norm="none", image_size=(args.width, args.height),
                    num_steps=args.steps, seed=args.seed,
                )
    finally:
        torch.randn = randn

    if args.record:
        import json as json_module

        from safetensors.numpy import save_file

        per_step = len(velocities) // args.steps
        records["noise"] = noise
        records["timesteps"] = model._apply_time_schedule(torch.linspace(0.0, 1.0, args.steps + 1), 0, args.shift)
        records["image_embeds"] = velocities[0][0]
        records["v_cond"] = velocities[0][1]
        if per_step == 2:
            records["v_uncond"] = velocities[1][1]
        records["v_cond_last"] = velocities[-per_step][1]
        # Every step's input, image and conditional velocity, to follow the two runs apart.
        for step in range(args.steps):
            embeds, velocity, z = velocities[step * per_step]
            records[f"step_{step}_embeds"] = embeds
            records[f"step_{step}_z"] = z
            records[f"step_{step}_v"] = velocity
        records["image"] = image
        args.record.mkdir(parents=True, exist_ok=True)
        save_file({k: np.ascontiguousarray(v.detach().float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy())
                   for k, v in records.items()}, str(args.record / "references.safetensors"))
        (args.record / "fixture.json").write_text(json_module.dumps({
            "family": "sensenova", "pack": str(args.pack), "prompt": args.prompt, "width": args.width, "height": args.height,
            "steps": args.steps, "guidance": args.guidance, "timestep_shift": args.shift, "t_eps": 0.02, "seed": args.seed,
        }, indent=2))

    # `_to_pil` of examples/t2i/inference.py.
    array = ((image.float() * 0.5 + 0.5).clamp(0, 1)).permute(0, 2, 3, 1).cpu().numpy()
    Image.fromarray((array * 255.0).round().astype(np.uint8)[0]).save(args.output)
    print("saved", args.output)


if __name__ == "__main__":
    main()
