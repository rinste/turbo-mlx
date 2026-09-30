#!/usr/bin/env python3
"""Builds a small SenseNova-U1.5 checkpoint with random weights, in the layout of the MLX packs the
app downloads (mlx-community/SenseNova-U1.5-8B-MoT-8step-4bit: the original's keys, convolutions
[O, kh, kw, I]; unquantized and float32 here), runs SenseTime's own `t2i_generate` on it, and
records what it computes: the native engine's `verify` compares its results against them, stage
by stage.

Run it with the Python of Engine/Fixtures/requirements-sensenova.txt:
  $PY Engine/Fixtures/make_sensenova_fixture.py <SenseNova-U1 clone> <out-dir>

The model keeps the original's structure at a fraction of the width: 2 layers of 256 (4 heads of
64, 2 key-value heads, the head split 32 for text positions and 16 + 16 for rows and columns), a
1024-token vocabulary, a 64-wide patch embedding, the pixel head as it is. The reference runs
unchanged in float32 on the CPU; the fixture only wraps its methods to record their results, and
gives it a tokenizer stub (the real tokenizer is checked by `turbo-engine verify-tokenizers`).
One image of 128 × 96 (4 × 3 tokens), 3 steps with guidance 2, so the unconditional pass runs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

CONFIG = {
    "llm": {"hidden_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64,
            "intermediate_size": 512, "num_hidden_layers": 2, "vocab_size": 1024},
    "vision_hidden_size": 64,
    "width": 128,
    "height": 96,
    "steps": 3,
    "guidance": 2.0,
    "timestep_shift": 3.0,
    "t_eps": 0.02,
    "seed": 5,
    "prompt": "a red bicycle leaning on a wall",
}


def pack_config():
    """config.json as the MLX pack has it, at the fixture's size."""
    llm = CONFIG["llm"]
    return {
        "architectures": ["NEOChatModel"],
        "model_type": "neo_chat",
        "template": "neo1_0",
        "downsample_ratio": 0.5,
        "patch_size": 16,
        "tie_word_embeddings": False,
        "llm_config": {
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "attention_bias": False, "hidden_act": "silu",
            "hidden_size": llm["hidden_size"], "intermediate_size": llm["intermediate_size"],
            "num_attention_heads": llm["num_attention_heads"], "num_key_value_heads": llm["num_key_value_heads"],
            "head_dim": llm["head_dim"], "num_hidden_layers": llm["num_hidden_layers"], "vocab_size": llm["vocab_size"],
            "rms_norm_eps": 1e-6, "rope_theta": 5000000.0, "rope_theta_hw": 10000.0,
            "max_position_embeddings": 262144, "max_position_embeddings_hw": 10000, "tie_word_embeddings": False,
            "use_sliding_window": False, "sliding_window": None, "max_window_layers": llm["num_hidden_layers"],
        },
        "vision_config": {
            "architectures": ["NEOVisionModel"], "model_type": "neo_vision", "hidden_size": CONFIG["vision_hidden_size"],
            "llm_hidden_size": llm["hidden_size"], "downsample_ratio": 0.5, "patch_size": 16, "num_channels": 3,
            "rope_theta_vision": 10000.0, "max_position_embeddings_vision": 10000,
        },
        "use_pixel_head": True, "use_adaLN": False, "fm_head_layers": 2, "fm_head_dim": 1536, "fm_head_mlp_ratio": 1,
        "concat_time_token_num": 0, "extra_num_layers_post": 0,
        "noise_scale": 1.0, "noise_scale_mode": "resolution", "noise_scale_base_image_seq_len": 64,
        "noise_scale_max_value": 16.0, "add_noise_scale_embedding": True,
        "time_schedule": "standard", "time_shift_type": "exponential", "timestep_shift": 1.0,
        "base_shift": 0.5, "max_shift": 1.15, "base_image_seq_len": 64, "max_image_seq_len": 4096,
        "t_eps": 0.05, "P_mean": -0.8, "P_std": 0.8,
    }


class StubTokenizer:
    """What `_build_t2i_text_inputs` asks of a tokenizer: ids for a text, here a fixed function of
    it (distinct lengths for the conditional and unconditional templates)."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text, return_tensors=None, **_):
        codes = [ord(c) for c in text]
        count = 7 + len(codes) % 13
        ids = [(sum(codes[i::count]) * 31 + i * 17) % self.vocab for i in range(count)]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(repo / "src"))
    import sensenova_u1
    from sensenova_u1.models.neo_unify.configuration_neo_chat import NEOChatConfig
    from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel

    sensenova_u1.set_attn_backend("sdpa")
    torch.manual_seed(0)
    config_json = pack_config()
    config = NEOChatConfig(**config_json)
    model = NEOChatModel(config).to(torch.float32).eval()

    # Random weights at scales that keep every stage informative: norms around 1, linears and
    # convolutions around 1/sqrt(fan-in), biases small.
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight") or "norm_" in name and name.endswith(".weight"):
                parameter.copy_(1 + 0.1 * torch.randn(parameter.shape, generator=generator))
            elif name.endswith(".bias"):
                parameter.copy_(0.02 * torch.randn(parameter.shape, generator=generator))
            else:
                fan_in = parameter[0].numel() if parameter.ndim > 1 else parameter.numel()
                parameter.copy_(torch.randn(parameter.shape, generator=generator) / fan_in ** 0.5)

    records = {}

    # The prefix caches (conditional first, then unconditional): the keys and values each layer
    # leaves for the image tokens.
    prefix_forward = model._t2i_prefix_forward
    prefix_calls = []

    def recording_prefix(input_ids, indexes, attention_mask):
        cache, hidden = prefix_forward(input_ids, indexes, attention_mask)
        tag = "cond" if not prefix_calls else "uncond"
        prefix_calls.append(tag)
        records[f"{tag}_input_ids"] = input_ids.to(torch.int32)
        for index, layer in enumerate(cache.layers):
            records[f"{tag}_keys_{index}"] = layer.keys
            records[f"{tag}_values_{index}"] = layer.values
        return cache, hidden

    model._t2i_prefix_forward = recording_prefix

    # The noise the loop starts from.
    randn = torch.randn

    def recording_randn(*args, **kwargs):
        noise = randn(*args, **kwargs)
        records["noise"] = noise
        return noise

    # The generation embedding of each step's image, and every velocity.
    extract = model.extract_feature
    embeds = []

    def recording_extract(pixel_values, gen_model=False, grid_hw=None):
        features = extract(pixel_values, gen_model=gen_model, grid_hw=grid_hw)
        if gen_model:
            embeds.append(features)
        return features

    model.extract_feature = recording_extract
    predict = model._t2i_predict_v
    velocities = []

    def recording_predict(input_embeds, *args, **kwargs):
        velocity = predict(input_embeds, *args, **kwargs)
        velocities.append((input_embeds, velocity))
        return velocity

    model._t2i_predict_v = recording_predict

    torch.randn = recording_randn
    try:
        image = model.t2i_generate(
            StubTokenizer(CONFIG["llm"]["vocab_size"]), CONFIG["prompt"],
            cfg_scale=CONFIG["guidance"], timestep_shift=CONFIG["timestep_shift"], cfg_norm="none",
            image_size=(CONFIG["width"], CONFIG["height"]), num_steps=CONFIG["steps"], seed=CONFIG["seed"],
            t_eps=CONFIG["t_eps"],
        )
    finally:
        torch.randn = randn

    steps = CONFIG["steps"]
    assert len(velocities) == 2 * steps and len(embeds) == steps, (len(velocities), len(embeds))
    timesteps = model._apply_time_schedule(torch.linspace(0.0, 1.0, steps + 1), 0, CONFIG["timestep_shift"])
    records["timesteps"] = timesteps
    records["vision_embeds"] = embeds[0]
    records["image_embeds"] = velocities[0][0]
    records["v_cond"] = velocities[0][1]
    records["v_uncond"] = velocities[1][1]
    records["v_cond_last"] = velocities[-2][1]
    records["image"] = image

    # The whole pipeline once more, guidance off, for the conditional pass alone.
    velocities.clear()
    embeds.clear()
    prefix_calls.clear()
    torch.randn = recording_randn
    try:
        records["image_no_guidance"] = model.t2i_generate(
            StubTokenizer(CONFIG["llm"]["vocab_size"]), CONFIG["prompt"], cfg_scale=1.0,
            timestep_shift=CONFIG["timestep_shift"], image_size=(CONFIG["width"], CONFIG["height"]),
            num_steps=steps, seed=CONFIG["seed"], t_eps=CONFIG["t_eps"],
        )
    finally:
        torch.randn = randn

    # Weights in the pack's layout: the original's keys, convolutions channels last, and the
    # embedders' second linear numbered 1 (the pack drops the SiLU between them from the count).
    weights = {}
    for name, tensor in model.state_dict().items():
        array = tensor.detach().to(torch.float32).numpy()
        if array.ndim == 4:
            array = array.transpose(0, 2, 3, 1)
        weights[name.replace("_embedder.mlp.2.", "_embedder.mlp.1.")] = np.ascontiguousarray(array)

    out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(config_json, indent=2))
    save_file({k: np.ascontiguousarray(v.detach().to(torch.float32).numpy() if v.is_floating_point() else v.numpy())
               for k, v in records.items()}, str(out / "references.safetensors"))
    fixture = {"family": "sensenova", **{k: v for k, v in CONFIG.items() if k != "llm"}}
    (out / "fixture.json").write_text(json.dumps(fixture, indent=2))
    print(f"fixture written to {out}: {len(weights)} weights, {len(records)} references")


if __name__ == "__main__":
    main()
