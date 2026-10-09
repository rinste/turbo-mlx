#!/usr/bin/env python3
"""Adds LoRA checks to existing fixtures (Z-Image, FLUX.2 Klein, Qwen-Image): random LoRA files in
the spellings real ones come in, and the transformer pass mflux computes with each applied the way
the engine applies them (unbaked: each layer adds scale · (x·A)·B). `turbo-engine verify` then puts
the same files on the port and compares the pass, and checks that taking them off gives the plain
pass back.

Each family gets two files, keyed as mflux's own mapping spells them (`possible_*_patterns`):
  Z-Image      ai-toolkit (`diffusion_model.….lora_A.weight`) and Kohya (`lora_unet_…`, with alpha)
  FLUX.2       diffusers (`transformer.…`) and BFL (`double_blocks.N.img_attn.qkv`, fused: one
               up matrix for the query, key and value layers, a third each; with alpha)
  Qwen-Image   ai-toolkit and Kohya (with alpha)

Run it with a Python that has mflux (Engine/Fixtures/requirements.txt), after the fixtures:
  $PY Engine/Fixtures/make_lora_fixture.py build/fixtures/zimage build/fixtures/klein build/fixtures/qwen-image
"""

import json
import re
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from mflux.models.common.lora.mapping.lora_loader import LoRALoader

RANK = 8
SCALE = 0.75
ALPHA = 4.0  # alpha / rank = 0.5 where a file has one


def load_transformer(fixture: Path, cfg: dict):
    """The fixture's transformer as mflux builds it, quantized as it was saved, with its weights."""
    family = cfg["family"] if "family" in cfg else "flux2-klein"
    bits = cfg["bits"]
    if family == "z-image-turbo":
        from mflux.models.z_image.model.z_image_transformer.transformer import ZImageTransformer
        from mflux.models.z_image.weights.z_image_lora_mapping import ZImageLoRAMapping

        transformer = ZImageTransformer(**cfg["transformer"])
        predicate = lambda path, m: hasattr(m, "to_quantized")  # noqa: E731
        mapping = ZImageLoRAMapping
    elif family == "flux2-klein":
        from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
        from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping

        transformer = Flux2Transformer(**cfg["transformer"])
        predicate = lambda path, m: hasattr(m, "to_quantized")  # noqa: E731
        mapping = Flux2LoRAMapping
    elif family == "qwen-image":
        from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
        from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping
        from mflux.models.qwen.weights.qwen_weight_definition import QwenWeightDefinition

        transformer = QwenTransformer(**cfg["transformer"])
        predicate = lambda path, m: QwenWeightDefinition.quantization_predicate(path, m, bits)  # noqa: E731
        mapping = QwenLoRAMapping
    else:
        raise SystemExit(f"{fixture}: no LoRA check for {family}")
    nn.quantize(transformer, group_size=64, bits=bits, class_predicate=predicate)
    weights = {}
    for shard in sorted((fixture / "transformer").glob("*.safetensors")):
        weights.update(mx.load(str(shard)))
    transformer.update(tree_unflatten(list(weights.items())))
    mx.eval(transformer.parameters())
    return family, transformer, mapping


def transformer_pass(family: str, transformer, cfg: dict, refs: dict):
    """The fixture's "transformer pass", from its reference inputs."""
    if family == "z-image-turbo":
        return transformer(x=refs["latents"], timestep=refs["timestep"], sigmas=refs["sigmas"], cap_feats=refs["prompt_embeds"])
    if family == "flux2-klein":
        return transformer(
            hidden_states=refs["latents"], encoder_hidden_states=refs["prompt_embeds"], timestep=refs["timestep"],
            img_ids=refs["latent_ids"], txt_ids=refs["text_ids"], guidance=None,
        )
    from mflux.models.common.config.config import Config
    from mflux.models.common.config.model_config import ModelConfig

    config = Config(
        model_config=ModelConfig.qwen_image(), num_inference_steps=cfg["steps"], height=cfg["height"],
        width=cfg["width"], guidance=cfg["guidance"], scheduler="linear",
    )
    embeds = refs["prompt_embeds"]
    return transformer(
        t=float(refs["timestep"].item()), config=config, hidden_states=refs["latents"],
        encoder_hidden_states=embeds, encoder_hidden_states_mask=mx.ones(embeds.shape[:2], dtype=mx.int32),
    )


BFL = ("double_blocks.", "single_blocks.", "img_in", "txt_in", "time_in.", "final_layer.")

# Which of a target's down patterns each file uses (the up and alpha keys follow from it).
FORMATS = {
    "z-image-turbo": {
        "aitoolkit": lambda p: p.startswith("diffusion_model.") and p.endswith(".lora_A.weight"),
        "kohya": lambda p: p.startswith("lora_unet_") and p.endswith(".lora_down.weight"),
    },
    "flux2-klein": {
        "diffusers": lambda p: p.startswith("transformer.") and p.endswith(".lora_A.weight"),
        "bfl": lambda p: re.match(r"^(diffusion_model|base_model\.model)\.", p) is not None
        and p.split(".", 1)[1].startswith(BFL) and p.endswith(".lora_down.weight")
        or p.startswith("diffusion_model.") and p.endswith(".lin.lora_down.weight"),
    },
    "qwen-image": {
        "aitoolkit": lambda p: p.startswith("diffusion_model.") and p.endswith(".lora_A.weight"),
        "kohya": lambda p: p.startswith("lora_unet_") and p.endswith(".lora_down.weight"),
    },
}


def module_at(root, path: str):
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def dims(layer) -> tuple[int, int]:
    out, inputs = layer.weight.shape
    if isinstance(layer, nn.QuantizedLinear):
        inputs = inputs * 32 // layer.bits
    return out, inputs


def make_file(transformer, mapping, accepts, seed: int) -> dict:
    """A random LoRA keyed in one spelling: for each target that has one, A [rank, in] and B
    [out, rank] (three outs for a fused source), plus alpha where the spelling has it."""
    key = mx.random.key(seed)
    sources: dict[str, dict] = {}
    for target in mapping.get_mapping():
        down = next((p for p in target.possible_down_patterns if accepts(p)), None)
        if down is None:
            continue
        stem = re.sub(r"\.(lora_A|lora_down)\.weight$", "", down)
        up = next((p for p in target.possible_up_patterns if re.sub(r"\.(lora_B|lora_up)\.weight$", "", p) == stem), None)
        if up is None:
            continue
        alpha = f"{stem}.alpha" if f"{stem}.alpha" in target.possible_alpha_patterns and "lora_down" in down else None
        for block in range(64) if "{block}" in target.model_path else [None]:
            path = target.model_path.format(block=block) if block is not None else target.model_path
            try:
                layer = module_at(transformer, path)
            except (AttributeError, IndexError):
                break
            if not hasattr(layer, "weight"):  # absent from this model (Klein has no guidance embedder)
                break
            out, inputs = dims(layer)
            name = stem.replace("{block}", str(block)) if block is not None else stem
            entry = sources.setdefault(name, {"down": down, "up": up, "alpha": alpha, "in": inputs, "outs": [], "block": block})
            entry["outs"].append(out)
    tensors = {}
    for name, entry in sources.items():
        # A fused source feeds several targets (the query, key and value): its up matrix stacks them.
        out = sum(entry["outs"])
        key, a, b = mx.random.split(key, 3)
        fill = lambda p: p.replace("{block}", str(entry["block"])) if entry["block"] is not None else p  # noqa: E731
        tensors[fill(entry["down"])] = (mx.random.normal((RANK, entry["in"]), key=a) / entry["in"] ** 0.5).astype(mx.float16)
        tensors[fill(entry["up"])] = (mx.random.normal((out, RANK), key=b) / RANK ** 0.5).astype(mx.float16)
        if entry["alpha"]:
            tensors[fill(entry["alpha"])] = mx.array(ALPHA, dtype=mx.float32)
    return tensors


def main(fixtures: list[Path]) -> None:
    for fixture in fixtures:
        cfg = json.loads((fixture / "fixture.json").read_text())
        refs = mx.load(str(fixture / "references.safetensors"))
        out = fixture / "lora"
        out.mkdir(exist_ok=True)
        entries, references = [], {}
        family = cfg.get("family", "flux2-klein")
        for index, (name, accepts) in enumerate(FORMATS[family].items()):
            fam, transformer, mapping = load_transformer(fixture, cfg)
            tensors = make_file(transformer, mapping, accepts, seed=100 + index)
            path = out / f"{name}.safetensors"
            mx.save_safetensors(str(path), tensors)
            LoRALoader.load_and_apply_lora(mapping.get_mapping(), transformer, [str(path)], [SCALE], bake_lora=False)
            noise = transformer_pass(fam, transformer, cfg, refs)
            mx.eval(noise)
            references[f"noise_{name}"] = noise
            layers = sum(1 for k in tensors if k.endswith(("lora_A.weight", "lora_down.weight")))
            entries.append({"format": name, "file": path.name, "scale": SCALE, "keys": len(tensors), "sources": layers})
            print(f"{fixture.name}: {name}, {len(tensors)} keys, pass {noise.shape} {noise.dtype}")
        mx.save_safetensors(str(out / "references.safetensors"), references)
        (out / "lora.json").write_text(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
