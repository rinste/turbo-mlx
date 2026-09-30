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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map

import ltx_core_mlx.components.diffusion_steps as diffusion_steps
import ltx_core_mlx.text_encoders.gemma.encoders.base_encoder as base_encoder
import ltx_core_mlx.text_encoders.gemma.encoders.gemma4_encoder as gemma4_encoder
import ltx_core_mlx.model.transformer.model as transformer_model
import ltx_pipelines_mlx.distilled as distilled
import ltx_pipelines_mlx.utils._orchestration as orchestration
import ltx_pipelines_mlx.utils.media_io as media_io
from ltx_pipelines_mlx.utils.blocks import PromptEncoder

refs: dict[str, mx.array] = {}
# Every Gemma state, to check the connector on its own (gemma_states.safetensors, ~385 MB).
gemma_states: dict[str, mx.array] = {}


def record(name: str, value) -> None:
    if name in refs or value is None:
        return
    mx.eval(value)
    refs[name] = value


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

    video_latent, audio_latent = pipe.generate_two_stage(
        args.prompt, args.height, args.width, args.frames, frame_rate=args.fps, seed=args.seed, image=args.image
    )
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
