# turbo-engine

The engine of Turbo MLX: the catalog's seven families (five image models, LTX-2 for video and the
SeedVR2 upscaler) on
[MLX Swift](https://github.com/ml-explore/mlx-swift), behind a JSON-lines protocol (below) that
the app speaks to it; the Python engine the app used to ship, `Reference/turbo_worker.py`, speaks
it too for the image families (see `docs/native-engine.md` for the why and the plan).

```
Sources/turbo-engine/         the executable: `serve` (the worker) and `verify` (parity checks)
Sources/TurboEngineCore/      protocol, mflux checkpoint loading, PNG output, the families
  Families/Family.swift       the FamilyModel protocol the engine drives, and the loader by family
  Families/Shared/            what families share: the Qwen3 prompter, the S3-DiT, the schedules,
                              tiled VAE decoding, mixture-of-experts layers, pixels, Pillow's
                              bicubic and Lanczos resizes
  Families/Klein/             FLUX.2 Klein: Qwen3 text encoder, FLUX.2 transformer, VAE decoder,
                              and the VAE encoder for the pictures an image is edited from
  Families/ZImage/            Z-Image Turbo: the Qwen3 encoder read in float32, the S3-DiT,
                              the 16-channel decoder
  Families/QwenImage/         Qwen-Image 2512: Qwen2.5-VL text encoder, the dual-stream
                              transformer, the causal 3D decoder, classifier-free guidance; and
                              Qwen-Image-Edit 2511: the vision tower and the VAE encoder for the
                              picture an image is edited from
  Families/Ming/              Ming-Image: the Ling MoE encoder, the Qwen2 connector and heads,
                              the S3-DiT in bf16, the RGBA decoder
  Families/LTX/               LTX-2.3: Gemma 3 12B and the text connector, the audio–video
                              transformer, the video VAE (tiled decode, image encoder), the ×2
                              latent upsampler, the audio VAE and the 48 kHz vocoder, the
                              distilled two-stage pipeline
  Families/SeedVR2/           SeedVR2 3B: the causal 3D VAE (tiled encode and decode), the
                              windowed transformer, one flow step, the wavelet and Lab color
                              correction; the picture's preparation with Pillow's bicubic
  Resources/                  SeedVR2's fixed text embedding, as mflux ships it
VideoOutput.swift             MP4 (H.264 + AAC) with AVAssetWriter, frames as they are decoded
Fixtures/make_*_fixture.py    build the checkpoint + references `verify` compares against
Fixtures/requirements.txt     the mflux revision they (and the reference worker) run with
Fixtures/requirements-ltx.txt the ltx-2-mlx revision the LTX-2 port follows
Reference/turbo_worker.py     mflux behind the same protocol, to compare real images
Licenses/                     the licenses of the projects the ports follow (mflux, ltx-2-mlx,
                              mlx-lm with mlx-swift-lm for the mixture-of-experts layers, Pillow
                              for its resizes), which the app's acknowledgements reproduce
```

SeedVR2's text embedding ships with the engine, in its resource bundle
(`TurboEngine_TurboEngineCore.bundle`, next to the binary or in the app's `Contents/Resources`):
the model has no text encoder, only this fixed 58 × 5120 prompt, which mflux ships the same way.

The modules mirror mflux's module tree name for name, so the checkpoints the app already
downloads (`mflux-community/flux2-klein-4b-mflux-q4` and friends) load without conversion: the
loader reads the shards, recognizes the layers stored quantized from their shapes (linears,
embeddings and Ming's stacked experts, at any of mflux's levels, mixed ones included), and puts
every tensor where its key says. SeedVR2 has no mflux checkpoint: mflux reads the original one
(`numz/SeedVR2_comfyUI`, float16) through a mapping, so its modules are named as that checkpoint
names its tensors instead, and only the VAE's convolutions are transposed on loading.

## Families

| Family | Text side | Transformer | Decoder | Guidance |
|---|---|---|---|---|
| FLUX.2 Klein | Qwen3 (hidden states of layers 9, 18, 27), padded to 512 | FLUX.2 double/single stream; an edit adds the reference picture's tokens after the image's (t = 10), only the image's come out | FLUX.2, 32 channels; the encoder for references, in their own proportions up to ~1 MP | base checkpoints: negative space |
| Z-Image Turbo | Qwen3 4B in float32, second-to-last state, real tokens only, thinking on | S3-DiT, tokens padded to 32 with learned pad tokens, float32 stream | FLUX.1-style, 16 channels, tiles with Save memory | off |
| Qwen-Image 2512 | Qwen2.5-VL 7B (bf16, unquantized), the template's 34 tokens dropped | 60 dual-stream blocks, float32 stream, modulation producers at 8 bits | Wan-derived causal 3D, 16 channels, tiles | true CFG, rescaled to the conditional norm; the unconditional pass is skipped at 1 |
| Qwen-Image-Edit 2511 | Qwen2.5-VL 7B with its vision tower: the picture at ~384 × 384 (Pillow's bicubic twice, as mflux) in 14-pixel patches, windowed attention, 2 × 2 merged; its tokens replace the placeholders of a 64-token template, everything in float32, then float16 | Qwen-Image's; the picture's latents follow the image's (frame position 1), only the image's come out | Qwen-Image's, plus the encoder for the picture at the image's size (its own proportions when they differ; Pillow's Lanczos) | as Qwen-Image, negative prompt empty |
| Ming-Image 0.1 Design | Ling-mini-2.0 MoE (256 experts, 8 routed with group-limited top-k, bf16 router as upstream), Qwen2 connector over 256 query tokens, direct-VLM head | S3-DiT in bf16, no padding, two caption streams | Qwen VAE for RGBA, one scaling factor, tiles | zeroed conditions |
| LTX-2.3 distilled | Gemma 3 12B (4-bit, all 49 hidden states, prompt left-padded to 1024), per-token RMS, two projections and two 8-block connectors with learnable registers | 48 audio–video blocks (4096 + 2048 wide, cross-modal attention both ways), block linears 4 or 8 bits, float32 activations | causal-3D conv VAE (non-causal decoder, 32 × 32 × 8, 128 channels), tiled over frames and pixels; audio VAE + BigVGAN vocoder with bandwidth extension to 48 kHz | none (distilled); two stages: 8 steps at half size, ×2 latent upsampler, 3 steps |
| SeedVR2 3B (upscaler) | none: a fixed 58 × 5120 embedding | 32 blocks of 2560 (10 with separate video and text weights, then shared), the video's 2 × 2 patches attending in windows (shifted every other block) with the whole text, rotary frequencies read from the checkpoint, float16 weights and float32 activations; input: noise, the picture's latent and a mask of ones; one Euler step from t = 1000 | causal 3D VAE, 16 channels, encode and decode in 512-pixel tiles; then the picture's low frequencies under the result's detail (five-level wavelet) and its Lab a/b (and 20% of L) histograms | none |

Every family keeps a prompt cache and encodes the queued prompts while its text encoder is
resident. With *Save memory*, Qwen-Image and Ming-Image release the text side once the prompts are
encoded and reload only it when a new prompt arrives (after releasing the transformer, so the two
are never co-resident); the other two keep everything loaded. Qwen-Image-Edit encodes its prompt
with the picture, so at the start of each image rather than ahead for the queue, and caches the
last few prompt–picture pairs; with *Save memory* it releases its text side the same way.

## Building

Requires Xcode 26 (mlx-swift 0.31.6 asks for a Swift 6.3 toolchain) and a Mac with Apple silicon.
`swift build` alone does not compile mlx-swift's Metal shaders on macOS; use Xcode or `xcodebuild`
(from `Engine/`, with `-skipPackagePluginValidation` for mlx-swift's package plug-in).

The app's build does it: its *Embed turbo-engine* phase runs `scripts/embed-engine.sh`, which
builds the engine in Release (`scripts/build-engine.sh`, derived data in `build/engine`) and puts
it in the bundle, `turbo-engine` in `Contents/MacOS` and the resource bundles it loads
(mlx-swift's Metal library among them) in `Contents/Resources`, signed like the app and with
`turbo-engine.entitlements`, which make it inherit the app's sandbox (a sandboxed app can start
no other helper, and none from outside its bundle). Start the app: the engine status shows
"turbo-engine 0.2" and every model runs on it. The engine is rebuilt only when `Engine/`
changed; when it fails to build, so does the app.

For the checks below:

```bash
scripts/build-engine.sh            # builds Release into build/bin/turbo-engine
```

To work on the engine in Xcode, open `Engine/Package.swift` and run the `turbo-engine` scheme
with the arguments below.

## Checking LTX-2 against ltx-2-mlx

LTX-2 has no mflux port: its reference is dgrauet's [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)
(MIT, the revision in `Fixtures/requirements-ltx.txt`), and its fixture is not a small random
checkpoint but a tiny run of the reference's distilled pipeline on a real pack (`dgrauet/ltx-2.3-mlx-q4`
or `-q8`, with `mlx-community/gemma-3-12b-it-4bit`), recorded stage by stage by hooks around its
own functions:

```bash
$PY Engine/Fixtures/make_ltx_fixture.py <pack> <gemma> /tmp/fixtures/ltx          # text to video
$PY Engine/Fixtures/make_ltx_fixture.py <pack> <gemma> /tmp/fixtures/ltx-i2v --image picture.png
build/bin/turbo-engine verify /tmp/fixtures/ltx
```

`verify` feeds each stage the reference's own inputs: the tokens (identical), Gemma's states, the
connector on the reference's 49 states (bit-identical), the noise and the positions (identical),
every transformer pass of both stages (within 1e-4), the upsampler, the image encoder, the
decoders and the vocoder. Three stages are shown but do not decide, for the reasons the output
gives: Gemma's deeper states and the contexts they lead to carry bfloat16 rounding through 48
layers; the first stage's loop moves a bfloat16 latent by about one unit of its last place at each
early step, so two correct loops end a percent apart; and the video encoder and decoder, whose
bfloat16 convolutions follow MLX's conv3d, which changed between mlx-swift's MLX (0.31) and the
reference's (0.32.2). The encoder and decoder are therefore also compared in float32, where they
match within 1e-4 (the generator writes those references too, from the reference's modules with
upcast weights, next to Gemma's 49 states for the connector's check). Both packs pass, the 4-bit
one from a prompt and from an image.

The whole pipeline, same prompt and seed through the app's protocol, 768 × 512 × 49 frames with
the 4-bit pack: the same clip as the reference (PSNR 29–35 dB per frame, 32.6 on average, through
H.264), the same loudness, in 97 s against 97 s, peaking at 15 GB with Save memory. A 5-second
clip (121 frames) takes 231 s and peaks at 17 GB; with the 8-bit pack and no Save memory, 294 s and
38 GB (41 GB for the whole process: a clip holds MLX's buffer cache to 4 GB, which otherwise kept
15 GB of passes already done).

## Checking a port against mflux

Before trusting it with a real checkpoint, compare each family with mflux on a small one:

```bash
# 0. A Python with mflux, once (any 3.10+; uv is the quickest way to one)
uv venv --python 3.12 ~/.venvs/mflux
uv pip install --python ~/.venvs/mflux/bin/python -r Engine/Fixtures/requirements.txt
PY=~/.venvs/mflux/bin/python

# 1. A tiny checkpoint with random weights, and what mflux computes from it
$PY Engine/Fixtures/make_klein_fixture.py      /tmp/fixtures/klein
$PY Engine/Fixtures/make_zimage_fixture.py     /tmp/fixtures/zimage
$PY Engine/Fixtures/make_qwen_image_fixture.py /tmp/fixtures/qwen-image
$PY Engine/Fixtures/make_qwen_image_edit_fixture.py /tmp/fixtures/qwen-image-edit
$PY Engine/Fixtures/make_ming_fixture.py       /tmp/fixtures/ming
$PY Engine/Fixtures/make_seedvr2_fixture.py    /tmp/fixtures/seedvr2

# 2. The same computations in Swift (the fixture names its family)
build/bin/turbo-engine verify /tmp/fixtures/klein
build/bin/turbo-engine verify /tmp/fixtures/zimage
build/bin/turbo-engine verify /tmp/fixtures/qwen-image
build/bin/turbo-engine verify /tmp/fixtures/qwen-image-edit
build/bin/turbo-engine verify /tmp/fixtures/ming
build/bin/turbo-engine verify /tmp/fixtures/seedvr2
```

`verify` reports, stage by stage, the largest difference relative to the reference's scale (and the
RMS one): the text side, the initial noise for the fixture's seed, one transformer pass (for Ming
also the unconditional one), the schedule, the whole denoising loop (guided where the family uses
guidance), and the decode; for Klein also an edit from a reference picture (the VAE encoder, the
picture's tokens and ids, one pass and the loop with them, its decode, and the sizes pictures of
several shapes are encoded at); for Qwen-Image-Edit the picture's resizes byte for byte, its
patches, the vision tower, the prompt with the picture's tokens, the picture's latents, a pass and
the loop over the image and the picture; for SeedVR2 (whose fixture is saved in the original
checkpoint's format and read back through mflux's own mapping first) the picture's preparation
byte for byte with and without softness, the tiled encode, the noise, the transformer's input and
one pass (bit-identical), the step, the tiled decode, the color correction, and the whole upscale
from the picture, shown with its PSNR (about 49 dB: the random weights and the histogram match
spread the VAE's bf16 rounding over a few pixels). Anything above 3% fails. Identical math lands well below that; a
wrong reshape, a swapped rotary pair or a missing cast shows up as a large error at the first stage
it touches.

Two things to know when reading the numbers. Klein's text encoder is also run with float32
activations, and that check decides for it: in bf16 its layers carry the rounding of MLX's
kernels, which differ between mlx-swift's MLX (0.31) and the Python one, and the fixture's random
weights amplify it (shown, marked "·"). Ming's router picks experts from bf16 scores, so a rounding
difference between the two MLX versions can flip a choice; the fixture's random weights make that
unlikely, and it would show as a large error on the caption features alone. Ming's text side and
the start of its decode run in bf16 as well, as in mflux, and land at 2–3% on the fixture where
the other families' stages stay under 1%.

The fixture generators are ordinary mflux code and run wherever mflux imports (they were exercised
on a Linux CPU build of MLX while the ports were written); `verify` needs the Mac.

Then the real thing, which the fixtures cannot stand in for: they are saved straight from mflux's
modules, and a published checkpoint can store a tensor in another shape (the Qwen VAE's norms
are flat in the catalog's checkpoints, for one). Start `build/bin/turbo-engine serve` and
`$PY Engine/Reference/turbo_worker.py serve`, send both the same `generate` line with the same
checkpoint, prompt, seed and size, and compare the two PNGs: the same image, with small local
differences from the two MLX versions' rounding (PSNR 27–35 dB at 512 × 512 for the four
families).

A Klein edit (`params.image`, the reference picture's path) is compared the same way against
mflux's `Flux2KleinEdit`, which the Python worker does not run. Give both an sRGB picture: the
engine draws the reference into sRGB, while mflux takes a Display P3 file's values as they are.
At 1024 × 640 the two land 25 dB apart. Against mflux run in float32, the engine is 2 dB behind
mflux in both cases: an edit 25 dB (mflux 27), text-to-image 28.5 dB (mflux 30). An edit carries
more of the bf16 rounding in both engines, and the engine's encoder is no worse than mflux's:
both are 3% (RMS) from a float32 encode.

SeedVR2 (`params.image` and `params.upscale`, the factor, with `params.softness`) is compared
against mflux's `SeedVR2(model_path=…).generate_image(resolution=ScaleFactor(…))`, which the
Python worker does not run either, on the original checkpoint's snapshot: 688 × 384 to
1376 × 768 lands 56.7 dB from mflux (at most 7 levels apart), 672 × 880 to 1344 × 1760 62 dB
(at most 4), in 18 s and 34 s against mflux's 37 s and 51 s, peaking at 10.4 GB against 18 GB.

## Protocol

Commands on stdin, one JSON object per line: `generate` (id, model, params), `load` (model),
`cancel` (id), `unload`, `shutdown`. Events on stdout: `ready`, `phase`, `progress`, `done`
(path, seed, size, seconds, peak_memory, timings per phase), `failed`, `cancelled`,
`model_loaded`, `load_failed` (a `load` that did not), `unloaded`. Everything else goes to stderr, which the app shows as the engine log.
