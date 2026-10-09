"""References for `turbo-engine verify` on LTX-2.3 and LTX-2.5 (the ltx-2 family).

Unlike the image families' fixtures (small random-weight checkpoints saved from mflux's
modules), this one runs the reference, dgrauet's ltx-2-mlx (the revision in
requirements-ltx.txt), on a real pack at a tiny size: its distilled two-stage pipeline, unchanged,
with the stages recorded on the way (hooks around its own functions), then its decoders. The pack
is read in place, so the fixture holds only the recorded tensors and names the pack it came from.

    $PY Engine/Fixtures/make_ltx_fixture.py <pack> <out> [--gemma folder] [--image picture.png]

<pack>: a dgrauet/ltx-2.3-mlx-q4 / -q8 or ltx-2.5-mlx-q4 / -q8 snapshot folder; --gemma: for 2.3,
mlx-community/gemma-3-12b-it-4bit (2.5 packs carry their own Gemma 4). With --image, the clip starts
from the picture (image-to-video) and the encoder's tokens for both stages are recorded as well. On
2.5 stage 1 runs the ancestral sampler: its noise draws are recorded too.

With --lora <fixture>/lora, the run stops at the first transformer pass instead, after writing two
random LoRA files there (blocks 0, 1 and the last, and the layers around them), one keyed as
Lightricks and ComfyUI key theirs, one as Kohya's (lora_down / lora_up, with an alpha), and the
pass with each applied as the engine applies them (each layer adds scale · (x·A)·B to its output),
plus the pass with ltx-2-mlx's own fusion of the first (the deltas baked into the weights, which
are quantized again), which `verify` shows for comparison. Run it with the arguments of an
existing fixture: `verify` takes the pass's inputs from that fixture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

import ltx_core_mlx.components.diffusion_steps as diffusion_steps
import ltx_core_mlx.text_encoders.gemma.encoders.base_encoder as base_encoder
import ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder as gemma4_encoder
import ltx_core_mlx.model.transformer.model as transformer_model
import ltx_pipelines_mlx.distilled as distilled
import ltx_pipelines_mlx.utils._orchestration as orchestration
import ltx_pipelines_mlx.utils.media_io as media_io
from ltx_pipelines_mlx.utils.blocks import DurationPredictor, PromptEncoder, seconds_to_clamped_num_frames

refs: dict[str, mx.array] = {}
# Every Gemma state, to check the connector on its own (gemma_states.safetensors, ~385 MB).
gemma_states: dict[str, mx.array] = {}


def record(name: str, value) -> None:
    if name in refs or value is None:
        return
    mx.eval(value)
    refs[name] = value


class LoRADone(Exception):
    """The LoRA references are written: the rest of the run is not needed."""


class UnbakedLoRA(nn.Module):
    """A linear layer plus scale · (x·down)·up, as the engine's LoRALinear computes it."""

    def __init__(self, base, down, up, scale):
        super().__init__()
        self.base, self.down, self.up, self.scale = base, down, up, scale
        self.weight = base.weight  # read by the adaLN deduplication's signature

    def __call__(self, x):
        return self.base(x) + self.scale * ((x @ self.down) @ self.up)


# Lightricks' (and ComfyUI's) names for the module paths ltx-2-mlx (and the port) use: the inverse
# of LTXV_LORA_COMFY_RENAMING_MAP.
COMFY_NAMES = [(".to_out.", ".to_out.0."), (".audio_ff.proj_in.", ".audio_ff.net.0.proj."),
               (".audio_ff.proj_out.", ".audio_ff.net.2."), (".ff.proj_in.", ".ff.net.0.proj."),
               (".ff.proj_out.", ".ff.net.2."), (".linear1.", ".linear_1."), (".linear2.", ".linear_2.")]


def comfy_path(path: str) -> str:
    dotted = "." + path + "."
    for ours, theirs in COMFY_NAMES:
        dotted = dotted.replace(ours, theirs)
    return dotted[1:-1]


def module_at(root, path: str):
    module = root
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def put(root, path: str, layer) -> None:
    *parents, last = path.split(".")
    parent = module_at(root, ".".join(parents)) if parents else root
    if last.isdigit():
        parent[int(last)] = layer
    else:
        setattr(parent, last, layer)


def lora_references(model, call, out: Path, scale: float = 0.8, rank: int = 16) -> None:
    """Writes the LoRA files and the passes with them (see the module's docstring)."""
    from ltx_core_mlx.loader.fuse_loras import apply_loras
    from ltx_core_mlx.loader.primitives import LoraStateDictWithStrength, StateDict
    from ltx_core_mlx.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
    from ltx_core_mlx.loader.sft_loader import SafetensorsStateDictLoader

    out.mkdir(parents=True, exist_ok=True)
    last = len(model.transformer_blocks) - 1
    blocks = ("transformer_blocks.0.", "transformer_blocks.1.", f"transformer_blocks.{last}.")
    targets = {}
    for path, layer in tree_flatten(model.leaf_modules(), is_leaf=lambda m: isinstance(m, nn.Module)):
        if not isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            continue
        if path.startswith("transformer_blocks.") and not path.startswith(blocks):
            continue
        outputs, inputs = layer.weight.shape
        if isinstance(layer, nn.QuantizedLinear):
            inputs = inputs * 32 // layer.bits
        targets[path] = (layer, inputs, outputs)

    key = mx.random.key(2026)
    comfy, kohya, matrices = {}, {}, {}
    for path, (layer, inputs, outputs) in sorted(targets.items()):
        name = comfy_path(path)
        renamed = LTXV_LORA_COMFY_RENAMING_MAP.apply_to_key(f"diffusion_model.{name}.lora_A.weight")
        assert renamed == f"{path}.lora_A.weight", (path, renamed)
        key, a_key, b_key = mx.random.split(key, 3)
        a = (mx.random.normal((rank, inputs), key=a_key) / inputs ** 0.5).astype(mx.bfloat16)
        b = (0.5 * mx.random.normal((outputs, rank), key=b_key) / rank ** 0.5).astype(mx.bfloat16)
        comfy[f"diffusion_model.{name}.lora_A.weight"] = a
        comfy[f"diffusion_model.{name}.lora_B.weight"] = b
        # Kohya's: the same delta, its up matrix doubled and an alpha of half the rank.
        underscored = "lora_unet_" + name.replace(".", "_")
        kohya[f"{underscored}.lora_down.weight"] = a
        kohya[f"{underscored}.lora_up.weight"] = (b * 2).astype(mx.bfloat16)
        kohya[f"{underscored}.alpha"] = mx.array(rank / 2, dtype=mx.float32)
        matrices[path] = (a, b)
    mx.save_safetensors(str(out / "comfy.safetensors"), comfy)
    mx.save_safetensors(str(out / "kohya.safetensors"), kohya)

    results = {}
    for form in ("comfy", "kohya"):
        for path, (a, b) in matrices.items():
            # Kohya's up matrix times alpha / rank, as the engine scales it.
            up = b if form == "comfy" else (b * 2).astype(mx.bfloat16) * ((rank / 2) / rank)
            put(model, path, UnbakedLoRA(targets[path][0], a.T, up.T, scale))
        video, audio = call()
        mx.eval(video, audio)
        results[f"video_{form}"], results[f"audio_{form}"] = video, audio
        for path, (layer, _, _) in targets.items():
            put(model, path, layer)

    # ltx-2-mlx's own way: the deltas fused into the weights, quantized again.
    plain = dict(tree_flatten(model.parameters()))
    lora_sd = SafetensorsStateDictLoader().load(str(out / "comfy.safetensors"), sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
    fused = apply_loras(StateDict(sd=plain, size=0, dtype=set()), [LoraStateDictWithStrength(lora_sd, scale)])
    model.load_weights(list(fused.sd.items()))
    video, audio = call()
    mx.eval(video, audio)
    results["video_comfy_baked"], results["audio_comfy_baked"] = video, audio
    model.load_weights(list(plain.items()))

    plain_video, plain_audio = call()
    for name, value in results.items():
        base = plain_video if name.startswith("video") else plain_audio
        change = float(mx.abs(value.astype(mx.float32) - base.astype(mx.float32)).max() / mx.abs(base).max())
        print(f"{name}: differs from the plain pass by {change:.2f} of its scale")
    mx.save_safetensors(str(out / "references.safetensors"), results)
    (out / "lora.json").write_text(json.dumps([
        {"format": "comfy", "file": "comfy.safetensors", "scale": scale, "layers": len(matrices)},
        {"format": "kohya", "file": "kohya.safetensors", "scale": scale, "layers": len(matrices)},
    ], indent=2))
    print(f"LoRA references written to {out} ({len(matrices)} layers)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack")
    parser.add_argument("out")
    parser.add_argument("--gemma", default="mlx-community/gemma-3-12b-it-4bit")
    parser.add_argument("--prompt", default="A red fox walks through fresh snow in a quiet pine forest at dawn, soft light.")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image")
    parser.add_argument("--crf", type=int, default=0,
                        help="H.264 round trip of the image before encoding (upstream: 33; 0 compares the ports on the same pixels)")
    parser.add_argument("--lora", help="write LoRA references to this folder and stop after the first pass")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Hooks: the pipeline runs unchanged, these only look. ---

    # Gemma 3 (2.3) or the pack's Gemma 4 (2.5), whichever the pipeline picks.
    def hidden_states_hook(get_all_hidden_states):
        def hook(self, token_ids, attention_mask=None):
            record("token_ids", token_ids)
            record("attention_mask", attention_mask)
            states = get_all_hidden_states(self, token_ids, attention_mask=attention_mask)
            for layer in (0, 1, 24, len(states) - 1):
                record(f"gemma_state_{layer}", states[layer])
            if not gemma_states:
                mx.eval(states)
                gemma_states.update({f"state_{i}": state for i, state in enumerate(states)})
            return states
        return hook

    base_encoder.GemmaLanguageModel.get_all_hidden_states = hidden_states_hook(base_encoder.GemmaLanguageModel.get_all_hidden_states)
    gemma4_encoder.Gemma4TextEncoder.get_all_hidden_states = hidden_states_hook(gemma4_encoder.Gemma4TextEncoder.get_all_hidden_states)

    model_call = transformer_model.LTXModel.__call__
    passes = {"count": 0}

    def model_hook(self, video_latent, audio_latent, timestep, **kwargs):
        video_v, audio_v = model_call(self, video_latent, audio_latent, timestep, **kwargs)
        index = passes["count"]
        if args.lora and index == 0:
            lora_references(self, lambda: model_call(self, video_latent, audio_latent, timestep, **kwargs), Path(args.lora))
            raise LoRADone
        passes["count"] += 1
        # Every pass's input and output, to see where two loops part.
        record(f"step{index}_video_in", video_latent)
        record(f"step{index}_audio_in", audio_latent)
        record(f"step{index}_sigma", timestep)
        record(f"step{index}_video_velocity", video_v)
        record(f"step{index}_audio_velocity", audio_v)
        if index in (0, 8):  # the first pass of each stage
            stage = 1 if index == 0 else 2
            record(f"pass{stage}_video_in", video_latent)
            record(f"pass{stage}_audio_in", audio_latent)
            record(f"pass{stage}_sigma", timestep)
            record(f"pass{stage}_video_timesteps", kwargs.get("video_timesteps"))
            record(f"pass{stage}_video_keyframes_mask", kwargs.get("video_keyframes_mask"))
            record(f"pass{stage}_video_positions", kwargs.get("video_positions"))
            record(f"pass{stage}_audio_positions", kwargs.get("audio_positions"))
            record(f"pass{stage}_video_velocity", video_v)
            record(f"pass{stage}_audio_velocity", audio_v)
        return video_v, audio_v

    transformer_model.LTXModel.__call__ = model_hook

    # Both loops (2.5's stage 1 is the ancestral one), counted together.
    loops = {"count": 0}

    def loop_hook(loop):
        def hook(**kwargs):
            stage = loops["count"] + 1
            loops["count"] += 1
            video_state, audio_state = kwargs["video_state"], kwargs["audio_state"]
            record(f"stage{stage}_video_init", video_state.latent)
            record(f"stage{stage}_audio_init", audio_state.latent)
            record(f"stage{stage}_video_clean", video_state.clean_latent)
            record(f"stage{stage}_video_mask", video_state.denoise_mask)
            record(f"stage{stage}_audio_clean", audio_state.clean_latent)
            record(f"stage{stage}_audio_mask", audio_state.denoise_mask)
            record(f"stage{stage}_video_keyframes_mask", getattr(video_state, "keyframes_mask", None))
            if "noise_seed" in kwargs:
                refs[f"stage{stage}_noise_seed"] = mx.array(kwargs["noise_seed"])
            output = loop(**kwargs)
            record(f"stage{stage}_video_out", output.video_latent)
            record(f"stage{stage}_audio_out", output.audio_latent)
            return output
        return hook

    distilled.denoise_loop = loop_hook(distilled.denoise_loop)
    distilled.euler_ancestral_denoising_loop = loop_hook(distilled.euler_ancestral_denoising_loop)

    # The ancestral noise, drawn video then audio at every step but the last.
    ancestral_step = diffusion_steps.EulerAncestralDiffusionStep.step
    draws = {"count": 0}

    def ancestral_hook(self, sample, denoised_sample, sigmas, step_index, noise=None):
        if noise is not None:
            record(f"ancestral_noise_{draws['count']}", noise)
            draws["count"] += 1
        return ancestral_step(self, sample, denoised_sample, sigmas, step_index, noise=noise)

    diffusion_steps.EulerAncestralDiffusionStep.step = ancestral_hook

    load_image = media_io.load_image_and_preprocess
    images = {"count": 0}

    def image_hook(image_path, height, width, crf=media_io.DEFAULT_IMAGE_CRF):
        tensor = load_image(image_path, height, width, crf=args.crf)
        images["count"] += 1
        record(f"stage{images['count']}_image_pixels", tensor)
        return tensor

    media_io.load_image_and_preprocess = image_hook

    combined = orchestration.combined_image_conditionings
    encodes = {"count": 0}

    def conditioning_hook(images, **kwargs):
        items = combined(images, **kwargs)
        stage = encodes["count"] + 1
        encodes["count"] += 1
        record(f"stage{stage}_image_tokens", items[0].clean_latent)
        return items

    orchestration.combined_image_conditionings = conditioning_hook

    # --- The pipeline. ---

    pipe = distilled.DistilledPipeline(model_dir=args.pack, gemma_model_id=args.gemma, low_memory=True)
    upsample = pipe._upsample_latent

    def upsample_hook(video_half, upsampler=None):
        record("upsample_in", video_half)
        result = upsample(video_half, upsampler)
        record("upsample_out", result)
        return result

    pipe._upsample_latent = upsample_hook

    encode = pipe.prompt_encoder.encode

    def encode_hook(prompt):
        video, audio = encode(prompt)
        record("video_embeds", video)
        record("audio_embeds", audio)
        return video, audio

    pipe.prompt_encoder.encode = encode_hook

    try:
        video_latent, audio_latent = pipe.generate_two_stage(
            args.prompt, args.height, args.width, args.frames, frame_rate=args.fps, seed=args.seed, image=args.image
        )
    except LoRADone:
        return
    record("video_latent", video_latent)
    record("audio_latent", audio_latent)

    # --- The decoders, untiled at this size. ---

    pipe.dit = None
    decoder = pipe.video_decoder_block.load()
    pixels = decoder.decode(video_latent)
    record("pixels", pixels)
    audio_decoder, vocoder = pipe.audio_decoder_block.load()
    mel = audio_decoder.decode(audio_latent)
    record("mel", mel)
    record("waveform", vocoder(mel))

    # The encoder and decoder again with float32 weights: in bfloat16 their convolutions follow
    # the MLX version's conv3d kernels, in float32 only the math shows.
    decoder.update(tree_map(lambda p: p.astype(mx.float32), decoder.parameters()))
    pixels_f32 = decoder.decode(video_latent.astype(mx.float32))
    mx.eval(pixels_f32)
    mx.save_safetensors(str(out / "pixels_f32.safetensors"), {"pixels_f32": pixels_f32})
    if args.image:
        encoder = pipe.image_conditioner.load()
        encoder.update(tree_map(lambda p: p.astype(mx.float32), encoder.parameters()))
        tokens = {}
        for stage in (1, 2):
            pixels = refs[f"stage{stage}_image_pixels"].astype(mx.float32)[:, :, None]
            latent = encoder.encode(pixels)
            tokens[f"stage{stage}_image_tokens_f32"] = latent.transpose(0, 2, 3, 4, 1).reshape(1, -1, 128)
        mx.eval(tokens)
        mx.save_safetensors(str(out / "image_tokens_f32.safetensors"), tokens)
    mx.save_safetensors(str(out / "gemma_states.safetensors"), gemma_states)

    # The connector alone in float32 on the recorded states: in bfloat16 its outputs follow rounding
    # (2.5's move by about 1 in 20 between the two precisions), in float32 only the math shows.
    prompt_encoder = PromptEncoder(args.pack, gemma_model_id=args.gemma)
    prompt_encoder._text_encoder = object()  # the connector only
    prompt_encoder.load()
    connector = prompt_encoder._feature_extractor
    connector.update(tree_map(lambda p: p.astype(mx.float32), connector.parameters()))
    states = [gemma_states[f"state_{i}"].astype(mx.float32) for i in range(len(gemma_states))]
    video_f32, audio_f32 = connector(states, attention_mask=refs["attention_mask"])
    mx.eval(video_f32, audio_f32)
    mx.save_safetensors(str(out / "connector_f32.safetensors"), {"video_embeds_f32": video_f32, "audio_embeds_f32": audio_f32})

    # 2.5: the DurationHead's seconds on the recorded contexts, and the seconds-to-frames rule on
    # a table of durations (clamped to 1–10 s at 24 and 25 fps).
    predictor = DurationPredictor.from_checkpoint(Path(args.pack))
    if predictor is not None:
        seconds = predictor._head(video_tokens=refs["video_embeds"], audio_tokens=refs["audio_embeds"])
        record("duration_seconds", seconds.astype(mx.float32))
        table = []
        for fps in (24.0, 25.0):
            for value in (0.2, 0.9, 1.0, 1.02, 2.5, 3.3125, 4.99, 5.0, 7.7, 9.99, 10.0, 12.0, 30.0):
                frames = seconds_to_clamped_num_frames(value, frame_rate=fps, min_frames=round(fps), max_frames=round(10 * fps))
                table.append([fps, value, frames])
        refs["duration_frames_table"] = mx.array(table, dtype=mx.float32)

    mx.save_safetensors(str(out / "references.safetensors"), refs)
    (out / "fixture.json").write_text(json.dumps({
        "family": "ltx-2",
        "pack": str(Path(args.pack).resolve()),
        "gemma": None if (Path(args.pack) / "text_encoder.safetensors").exists() else str(Path(args.gemma).resolve()),
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "seed": args.seed,
        "image": str(Path(args.image).resolve()) if args.image else None,
        "crf": args.crf,
        "mlx": mx.__version__,
    }, indent=2))
    print(f"saved {len(refs)} references to {out}")
    for name, value in sorted(refs.items()):
        print(f"  {name:28} {str(value.dtype):16} {tuple(value.shape)}")


if __name__ == "__main__":
    main()
