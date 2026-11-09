"""Qwen2.5-VL 7B Instruct describing pictures as mlx-vlm runs it, for Describe Image's `turbo-engine verify`.

The reference is mlx-vlm's port (mlx_vlm/models/qwen2_5_vl, MIT, the release on PyPI as 0.7.12, with its image processor
`Qwen3VLImageProcessor` and its `generate_step`) on the pack the app downloads (mlx-community/Qwen2.5-VL-7B-Instruct-4bit,
commit fdcc572e). It reads each picture at the describer's 256 to 1280 tokens and asks for its description with the
describer's instruction, at temperature 0.

    python qwen25vl_reference.py <pack> <dir> <picture>... [--instruction "..."] [--max-tokens 300]

writes <dir>/fixture.json and references.safetensors: for each picture, the pixel values and the grid the processor makes,
the prompt's ids (mlx-vlm's chat template, then its processor), the vision tower's tokens, the first step's logits (as
`generate_step` computes them, read through a logits processor that changes nothing), and the tokens it writes greedily
until `<|im_end|>`, with their text; and truth.safetensors, the vision tokens and the first logits again in float32. The
instruction must be the describer's (Qwen25VLDescriber.instruction): verify reads it from fixture.json and says when it
is not. The pictures are best sRGB PNGs (or PNGs without a profile), which ImageIO and Pillow decode to the same bytes,
where two JPEG decoders may differ by a level here and there.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import prepare_inputs

INSTRUCTION = (
    "Write a prompt for a text-to-image model that would re-create this picture: one paragraph that starts with the kind "
    "of picture and its main subject, then describes the setting, the composition, the lighting and the colors. If the "
    "picture contains text, copy it exactly as written, in double quotes. Write only the prompt, and do not start it with "
    "\"The picture\" or \"The image\"."
)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
STOPS = {151645, 151643}  # <|im_end|>, <|endoftext|>


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack")
    parser.add_argument("out")
    parser.add_argument("pictures", nargs="+")
    parser.add_argument("--instruction", default=INSTRUCTION)
    parser.add_argument("--max-tokens", type=int, default=300)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, processor = load(args.pack)
    processor.image_processor.min_pixels = MIN_PIXELS
    processor.image_processor.max_pixels = MAX_PIXELS
    prompt = apply_chat_template(processor, model.config, args.instruction, num_images=1)

    arrays = {}
    pictures = []
    for path in args.pictures:
        name = Path(path).stem
        inputs = prepare_inputs(processor, images=[path], prompts=prompt, add_special_tokens=False)
        ids, pixels, grid = inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"]
        vision = model.vision_tower(pixels.astype(model.vision_tower.patch_embed.proj.weight.dtype), grid)

        first = []

        def capture(tokens, logits):
            if not first:
                first.append(logits[0].astype(mx.float32))
            return logits

        tokens = []
        started = time.time()
        for result in stream_generate(model, processor, prompt, image=[path], max_tokens=args.max_tokens, temperature=0.0,
                                      logits_processors=[capture]):
            if result.token is None or result.token in STOPS:
                break
            tokens.append(result.token)
        seconds = time.time() - started
        text = processor.tokenizer.decode(tokens).strip()
        print(f"{name}: {grid.tolist()[0]}, {ids.shape[1]} prompt tokens, {len(tokens)} written in {seconds:.1f} s, "
              f"peak {mx.get_peak_memory() / 1e9:.2f} GB\n{text}\n", file=sys.stderr)

        arrays[f"{name}.pixel_values"] = pixels.astype(mx.float32)
        arrays[f"{name}.grid"] = grid.astype(mx.int32)
        arrays[f"{name}.ids"] = ids.astype(mx.int32)
        arrays[f"{name}.vision"] = vision
        arrays[f"{name}.logits"] = first[0]
        arrays[f"{name}.tokens"] = mx.array(tokens, dtype=mx.int32)
        pictures.append({"name": name, "path": str(Path(path).resolve()), "text": text})

    mx.save_safetensors(str(out / "references.safetensors"), arrays)

    # The same two stages with every activation and every unquantized weight in float32 (the 4-bit weights dequantized
    # with float32 scales), from the same inputs: what verify measures both the port and mlx-vlm against, the two
    # builds of MLX (PyPI's and mlx-swift's) rounding float16 apart.
    model.set_dtype(mx.float32)
    truths = {}
    for picture in pictures:
        name = picture["name"]
        pixels, grid = arrays[f"{name}.pixel_values"], arrays[f"{name}.grid"]
        truths[f"{name}.vision"] = model.vision_tower(pixels, grid)
        first = []

        def capture(tokens, logits):
            if not first:
                first.append(logits[0].astype(mx.float32))
            return logits

        for _ in stream_generate(model, processor, prompt, image=[picture["path"]], max_tokens=1, temperature=0.0,
                                 logits_processors=[capture]):
            pass
        truths[f"{name}.logits"] = first[0]
    mx.save_safetensors(str(out / "truth.safetensors"), truths)
    fixture = {
        "family": "qwen2.5-vl",
        "pack": str(Path(args.pack).resolve()),
        "instruction": args.instruction,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "max_tokens": args.max_tokens,
        "pictures": pictures,
    }
    (out / "fixture.json").write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
