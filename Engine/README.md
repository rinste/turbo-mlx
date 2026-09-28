# turbo-engine

The engine of Turbo MLX: the catalog's four families on [MLX Swift](https://github.com/ml-explore/mlx-swift),
behind a JSON-lines protocol (below) that the app speaks to it; the Python engine the app used to
ship, `Reference/turbo_worker.py`, speaks it too (see `docs/native-engine.md` for the why and the
plan).

```
Sources/turbo-engine/         the executable: `serve` (the worker) and `verify` (parity checks)
Sources/TurboEngineCore/      protocol, mflux checkpoint loading, PNG output, the families
  Families/Family.swift       the FamilyModel protocol the engine drives, and the loader by family
  Families/Shared/            what families share: the Qwen3 prompter, the S3-DiT, the schedules,
                              tiled VAE decoding, mixture-of-experts layers, pixels
  Families/Klein/             FLUX.2 Klein: Qwen3 text encoder, FLUX.2 transformer, VAE decoder
  Families/ZImage/            Z-Image Turbo: the Qwen3 encoder read in float32, the S3-DiT,
                              the 16-channel decoder
  Families/QwenImage/         Qwen-Image 2512: Qwen2.5-VL text encoder, the dual-stream
                              transformer, the causal 3D decoder, classifier-free guidance
  Families/Ming/              Ming-Image: the Ling MoE encoder, the Qwen2 connector and heads,
                              the S3-DiT in bf16, the RGBA decoder
Fixtures/make_*_fixture.py    build the checkpoint + references `verify` compares against
Fixtures/requirements.txt     the mflux revision they (and the reference worker) run with
Reference/turbo_worker.py     mflux behind the same protocol, to compare real images
```

The modules mirror mflux's module tree name for name, so the checkpoints the app already
downloads (`mflux-community/flux2-klein-4b-mflux-q4` and friends) load without conversion: the
loader reads the shards, recognizes the layers stored quantized from their shapes (linears,
embeddings and Ming's stacked experts, at any of mflux's levels, mixed ones included), and puts
every tensor where its key says.

## Families

| Family | Text side | Transformer | Decoder | Guidance |
|---|---|---|---|---|
| FLUX.2 Klein | Qwen3 (hidden states of layers 9, 18, 27), padded to 512 | FLUX.2 double/single stream | FLUX.2, 32 channels | base checkpoints: negative space |
| Z-Image Turbo | Qwen3 4B in float32, second-to-last state, real tokens only, thinking on | S3-DiT, tokens padded to 32 with learned pad tokens, float32 stream | FLUX.1-style, 16 channels, tiles with Save memory | off |
| Qwen-Image 2512 | Qwen2.5-VL 7B (bf16, unquantized), the template's 34 tokens dropped | 60 dual-stream blocks, float32 stream, modulation producers at 8 bits | Wan-derived causal 3D, 16 channels, tiles | true CFG, rescaled to the conditional norm; the unconditional pass is skipped at 1 |
| Ming-Image 0.1 Design | Ling-mini-2.0 MoE (256 experts, 8 routed with group-limited top-k, bf16 router as upstream), Qwen2 connector over 256 query tokens, direct-VLM head | S3-DiT in bf16, no padding, two caption streams | Qwen VAE for RGBA, one scaling factor, tiles | zeroed conditions |

Every family keeps a prompt cache and encodes the queued prompts while its text encoder is
resident. With *Save memory*, Qwen-Image and Ming-Image release the text side once the prompts are
encoded and reload only it when a new prompt arrives (after releasing the transformer, so the two
are never co-resident); the other two keep everything loaded.

## Building

Requires Xcode 26 (mlx-swift 0.31.6 asks for a Swift 6.3 toolchain) and a Mac with Apple silicon.
`swift build` alone does not compile mlx-swift's Metal shaders on macOS; use Xcode or `xcodebuild`
(from `Engine/`, with `-skipPackagePluginValidation` for mlx-swift's package plug-in).

The app's build does it: its *Embed turbo-engine* phase runs `scripts/embed-engine.sh`, which
builds the engine in Release (`scripts/build-engine.sh`, derived data in `build/engine`) and puts
it in the bundle, `turbo-engine` in `Contents/MacOS` and the resource bundles it loads
(mlx-swift's Metal library among them) in `Contents/Resources`, signed like the app. Start the
app: the engine status shows "turbo-engine 0.2" and every model runs on it. The engine is rebuilt
only when `Engine/` changed; when it fails to build, so does the app.

For the checks below, or to run the app on another build of the engine:

```bash
scripts/build-engine.sh            # builds Release into build/bin/turbo-engine
```

The app runs the engine `TURBO_ENGINE` names before its own (`open --env
TURBO_ENGINE=$PWD/build/bin/turbo-engine "path/to/Turbo MLX.app"`). To work on the engine in
Xcode, open `Engine/Package.swift` and run the `turbo-engine` scheme with the arguments below.

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
$PY Engine/Fixtures/make_ming_fixture.py       /tmp/fixtures/ming

# 2. The same computations in Swift (the fixture names its family)
build/bin/turbo-engine verify /tmp/fixtures/klein
build/bin/turbo-engine verify /tmp/fixtures/zimage
build/bin/turbo-engine verify /tmp/fixtures/qwen-image
build/bin/turbo-engine verify /tmp/fixtures/ming
```

`verify` reports, stage by stage, the largest difference relative to the reference's scale (and
the RMS one): the text side, the initial noise for the fixture's seed, one transformer pass (for
Ming also the unconditional one), the schedule, the whole denoising loop (guided where the family
uses guidance), and the decode. Anything above 3% fails. Identical math lands well below that; a
wrong reshape, a swapped rotary pair or a missing cast shows up as a large error at the first
stage it touches.

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

## Protocol

Commands on stdin, one JSON object per line: `generate` (id, model, params), `load` (model),
`cancel` (id), `unload`, `shutdown`. Events on stdout: `ready`, `phase`, `progress`, `done`
(path, seed, size, seconds, peak_memory, timings per phase), `failed`, `cancelled`,
`model_loaded`, `unloaded`. Everything else goes to stderr, which the app shows as the engine log.
