"""MiniMax Music 3 as mlx-audio runs it, stage by stage, for `turbo-engine verify`.

The reference is mlx-audio's port (mlx_audio/music/models/minimax_music3, Apache 2.0, the release on PyPI as 0.6.3, the
same files as its main branch on 2026-11-01), which follows the official diffusers integration
(huggingface/diffusers 9cdd659, modular_pipelines/minimax_music3) but for the prompt's cleaning. The prompt is
checked against the official functions instead (`--prompts`), copied below from diffusers' encoders.py (Apache 2.0).

    python music3_reference.py <pack> --record <dir> [--frames 150] [--steps 30]
    python music3_reference.py <pack> --truth <dir>
    python music3_reference.py --prompts <dir>

`--record` writes <dir>/fixture.json and references.safetensors: the prompt and its ids, the prompt's last hidden
states, the first frames of the autoregressive stage step by step (the keys, the guided semantic logits, the codes,
the depth decoder's hidden states, the feedback and the next hidden states), every frame's hidden states, then the
flow on them: the first window's condition, one guided step's velocities, the window's latents, the vocoder's
waveform, and a two-window run on the recorded frames repeated to 320 (the overlap) with fewer steps. `--prompts`
writes <dir>/prompts.json: captions and lyrics with what the official functions make of them. `--truth` adds
truth.safetensors to a recording: the same stages from the recording's own inputs with every activation and every
unquantized weight in float32 (the 8-bit weights dequantized as they are), against which `verify` measures both the
port and the reference: the two MLX builds (PyPI's and mlx-swift's) round bf16 apart, so the port passes a stage when it
is no further from float32 than the reference is, give or take half.
"""

import argparse
import json
import re
import sys
from pathlib import Path


# --- The official prompt functions (diffusers 9cdd659, src/diffusers/modular_pipelines/minimax_music3/encoders.py) ---

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def _clean_caption(caption: str) -> str:
    def _rewrite_special_tag(match: re.Match) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def _normalize_lyrics(lyrics: str) -> str:
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def assemble(caption: str, lyrics: str) -> str:
    return (
        f"<|im_start|><|caption_start|>{_clean_caption(caption)}<|caption_end|>"
        f"<|lyrics_start|>{_normalize_lyrics(lyrics)}<|lyrics_end|><|im_end|><|audio_start|>"
    )


CAPTIONS = [
    "Warm acoustic pop, 96 BPM, intimate female vocal",
    "## Global Metadata\n- **Genre**: synthwave\n- *Tempo*: 118 BPM\n\n---\n## Vocal Details\n  * breathy **male** lead\n",
    "<|genre electronic dance|> <|bpm 128|> <|mood|> euphoric • bright    pads\r\nsecond line\twith tab  ",
    "***bold italic*** and **nested **bold** marks** *it*alic * spaced *",
    "Ballata italiana, voce femminile, pianoforte e archi\u0085fine",
    "   ### Heading with spaces\n+ plus bullet\n___\n\n\nend  ",
    "",
]

LYRICS = [
    "[verse]\nMorning light across the room\n[chorus]\nSing with me",
    "[Verse 1] text on the tag line is dropped\nPlain line [Chorus] inline tag\nA ^ caret ^ break\n[BRIDGE][Outro]",
    "[instrumental]",
    "  [Intro]\t \nLa luce del mattino\r\nsulle strade di città\n\n[Chorus]\nCantiamo ÀÉÎ [İSTANBUL]",
    "no tags at all\njust words",
]


def write_prompts(folder: Path):
    cases = [{"caption": c, "lyrics": l, "caption_clean": _clean_caption(c), "lyrics_normalized": _normalize_lyrics(l),
              "assembled": assemble(c, l)} for c in CAPTIONS for l in LYRICS]
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompts.json").write_text(json.dumps({"family": "minimax-music3-prompts", "cases": cases}, ensure_ascii=False, indent=1))
    print(f"wrote {len(cases)} prompt cases to {folder / 'prompts.json'}")


# --- The recording ---

CAPTION = "Warm acoustic pop, 96 BPM, intimate female vocal"
LYRICS_TEXT = "[verse]\nMorning light across the room\n[chorus]\nSing with me"


def record(pack: Path, folder: Path, frames: int, steps: int, seed: int, ar_steps: int):
    import mlx.core as mx
    from mlx_audio.music import load
    from mlx_audio.music.models.minimax_music3 import ar as AR
    from mlx_audio.music.models.minimax_music3.config import AR_CFG_SCALE, DIT_CFG_SCALE
    from mlx_audio.music.models.minimax_music3.euler import denoise_chunk
    from mlx_audio.music.models.minimax_music3.sampling import sample_top_k
    from mlx_audio.music.models.minimax_music3.fusion import fuse_frame_hiddens

    model = load(str(pack))
    config = model.config
    lm, depth = model.language_model, model.rvq_depth_decoder
    refs = {}

    text = assemble(CAPTION, LYRICS_TEXT)
    ids = model._text_ids(CAPTION, LYRICS_TEXT)
    refs["ids"] = ids

    # The prompt, then the first frames one call at a time, as generate_frame_hiddens runs them.
    key = mx.random.key(seed)
    embeddings = lm.model.embed_tokens(ids)
    hidden, cache = AR.qwen3_hidden(lm, embeddings)
    last_hidden = hidden[:, -1]
    refs["prefill_last"] = last_hidden
    for index in range(ar_steps):
        key, subkey = mx.random.split(key)
        refs[f"ar{index}_key"] = subkey
        refs[f"ar{index}_last"] = last_hidden
        logits = AR.lm_logits(lm, last_hidden).astype(mx.float32)
        token_ids = mx.arange(logits.shape[-1])
        allowed = mx.logical_or(
            mx.logical_and(token_ids >= config.audio_code_offset, token_ids < config.audio_code_offset + config.semantic_vocab_size),
            token_ids == config.audio_end_token_id)
        logits = mx.where(allowed[None, :], logits, -1e9)
        conditional, unconditional = logits[:1], logits[1:2]
        guided = unconditional + (conditional - unconditional) * AR_CFG_SCALE
        threshold = mx.min(mx.topk(conditional, 50, axis=-1), axis=-1, keepdims=True)
        guided = mx.where(conditional < threshold, -1e9, guided)
        guided = mx.where(allowed[None, :], guided, -1e9)
        refs[f"ar{index}_guided"] = guided
        sampled, next_key = sample_top_k(guided, subkey, 50)
        assert int(sampled.item()) != config.audio_end_token_id
        semantic = mx.concatenate([sampled, sampled], axis=0) - config.audio_code_offset
        codes, depth_hidden, _ = AR.generate_depth_codes(lm, depth, config, last_hidden, semantic, next_key)
        refs[f"ar{index}_codes"] = codes.astype(mx.int32)
        refs[f"ar{index}_depth_hidden"] = depth_hidden
        feedback = AR._embed_audio_frame(lm, depth, config, codes)
        refs[f"ar{index}_feedback"] = feedback
        hidden, cache = AR.qwen3_hidden(lm, feedback, cache)
        last_hidden = hidden[:, -1]
        mx.eval(last_hidden)
    refs["ar_next_last"] = last_hidden

    # Every frame, from the seed.
    frame_hiddens = AR.generate_frame_hiddens(lm, depth, config, ids, max_frames=frames, seed=seed)
    mx.eval(frame_hiddens)
    refs["frames"] = frame_hiddens

    # The flow's first window.
    condition = model.condition_encoder(frame_hiddens[:, :200])
    refs["condition"] = condition
    mx.random.seed(seed + 7)
    noise = mx.random.normal((1, config.dit_in_channels, condition.shape[1])).astype(condition.dtype)
    refs["noise"] = noise
    timestep = mx.full((1,), 0.4, dtype=noise.dtype)
    refs["velocity_cond"] = model.transformer(noise, timestep, condition)
    refs["velocity_uncond"] = model.transformer(noise, timestep, mx.zeros_like(condition))
    latents, _ = denoise_chunk(model.transformer, noise, condition, num_inference_steps=steps, guidance_scale=DIT_CFG_SCALE)
    refs["latents"] = latents
    refs["wave"] = model.vocoder(latents)

    # Two windows, with the overlap, on the frames repeated to 320.
    long_frames = mx.concatenate([frame_hiddens] * (320 // frame_hiddens.shape[1] + 1), axis=1)[:, :320]
    refs["long_frames"] = long_frames
    refs["long_audio"] = model._run_flow(long_frames, 4, seed)

    mx.eval(refs)
    folder.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(folder / "references.safetensors"), refs)
    (folder / "fixture.json").write_text(json.dumps({
        "family": "minimax-music3", "pack": str(pack), "caption": CAPTION, "lyrics": LYRICS_TEXT, "assembled": text,
        "seed": seed, "frames": frames, "steps": steps, "ar_steps": ar_steps, "long_steps": 4, "timestep": 0.4,
    }, ensure_ascii=False, indent=1))
    print(f"recorded {len(refs)} references in {folder}")


def depth_hidden_forced(lm, depth, config, last_hidden, codes):
    """The depth decoder's hidden states of the conditional row for given codes [2, codebooks] (generate_depth_codes
    without the draws)."""
    import mlx.core as mx
    sequence = [depth.projection(last_hidden)[:, None, :]]
    sequence.append(depth.projection(lm.model.embed_tokens(codes[:, 0] + config.audio_code_offset))[:, None, :])
    parts = []
    for index in range(1, config.num_codebooks):
        hidden = depth(mx.concatenate(sequence, axis=1))[:, -1]
        parts.append(hidden[:1])
        if index < config.num_codebooks - 1:
            embedding = depth.audio_embeddings(codes[:, index] + (index - 1) * config.audio_vocab_size)
            sequence.append(depth.projection(embedding)[:, None, :])
    return mx.concatenate(parts, axis=-1)


def truth(pack: Path, folder: Path):
    import mlx.core as mx
    from mlx_audio.music import load
    from mlx_audio.music.models.minimax_music3 import ar as AR
    from mlx_audio.music.models.minimax_music3.config import DIT_CFG_SCALE
    from mlx_audio.music.models.minimax_music3.euler import denoise_chunk

    fixture = json.loads((folder / "fixture.json").read_text())
    refs = mx.load(str(folder / "references.safetensors"))
    model = load(str(pack))
    config = model.config
    f32 = lambda x: x.astype(mx.float32)
    # bf16 bits of the reference (bf16) weights and inputs, computed in float32.
    model.apply(lambda p: p.astype(mx.float32) if p.dtype == mx.bfloat16 else p)
    lm, depth = model.language_model, model.rvq_depth_decoder
    out = {}
    hidden, _ = AR.qwen3_hidden(lm, lm.model.embed_tokens(refs["ids"]))
    out["prefill_last"] = hidden[:, -1]
    for index in range(fixture["ar_steps"]):
        out[f"ar{index}_depth_hidden"] = depth_hidden_forced(lm, depth, config, f32(refs[f"ar{index}_last"]), refs[f"ar{index}_codes"])
        mx.eval(out)
    condition = model.condition_encoder(f32(refs["frames"][:, :200]))
    out["condition"] = condition
    noise, cond = f32(refs["noise"]), f32(refs["condition"])
    timestep = mx.full((1,), fixture["timestep"], dtype=mx.float32)
    out["velocity_cond"] = model.transformer(noise, timestep, cond)
    out["velocity_uncond"] = model.transformer(noise, timestep, mx.zeros_like(cond))
    out["latents"], _ = denoise_chunk(model.transformer, noise, cond, num_inference_steps=fixture["steps"], guidance_scale=DIT_CFG_SCALE)
    out["wave"] = model.vocoder(f32(refs["latents"]))
    out["long_audio"] = model._run_flow(f32(refs["long_frames"]), fixture["long_steps"], fixture["seed"])
    mx.eval(out)
    mx.save_safetensors(str(folder / "truth.safetensors"), out)
    print(f"wrote {len(out)} float32 references to {folder / 'truth.safetensors'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", nargs="?")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ar-steps", type=int, default=6)
    args = parser.parse_args()
    if args.prompts:
        write_prompts(args.prompts)
    if args.record:
        if not args.pack:
            sys.exit("give the pack's folder")
        record(Path(args.pack), args.record, args.frames, args.steps, args.seed, args.ar_steps)
    if args.truth:
        if not args.pack:
            sys.exit("give the pack's folder")
        truth(Path(args.pack), args.truth)


if __name__ == "__main__":
    main()
