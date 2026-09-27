# turbo-engine

The native engine of Turbo MLX: FLUX.2 Klein on [MLX Swift](https://github.com/ml-explore/mlx-swift),
speaking the same JSON-lines protocol as `turbo_worker.py`, so the app runs either without knowing
the difference (see `docs/native-engine.md` for the why and the plan).

```
Sources/turbo-engine/         the executable: `serve` (the worker) and `verify` (parity check)
Sources/TurboEngineCore/      protocol, mflux checkpoint loading, PNG output, the families
  Families/Klein/             Qwen3 text encoder, FLUX.2 transformer, VAE decoder, scheduler, prompt
Fixtures/make_klein_fixture.py  builds the checkpoint + references `verify` compares against
```

The modules mirror mflux's module tree name for name, so the checkpoints the app already
downloads (`mflux-community/flux2-klein-4b-mflux-q4` and friends) load without conversion: the
loader reads the shards, recognizes the layers stored quantized from their shapes, and puts every
tensor where its key says.

## Building

Requires Xcode 26 (mlx-swift 0.31.6 asks for a Swift 6.3 toolchain) and a Mac with Apple silicon.
`swift build` alone does not compile mlx-swift's Metal shaders on macOS; use Xcode or `xcodebuild`:

```bash
scripts/build-engine.sh            # builds Release and installs the binary for the app
```

The script puts `turbo-engine` in `~/Library/Application Support/TurboMLX/bin/`, where the app
looks for it (after `TURBO_ENGINE` and the app bundle). Start the app: the engine status shows
"turbo-engine 0.1" once a FLUX.2 Klein model is selected, and Klein images run natively. The
Python engine keeps serving the other families.

To work on the engine in Xcode, open `Engine/Package.swift` and run the `turbo-engine` scheme
with the arguments below.

## Checking the port against mflux

Before trusting it with a real checkpoint, compare it with mflux on a small one:

```bash
# 1. A tiny Klein checkpoint with random weights, and what mflux computes from it
~/Library/Application\ Support/TurboMLX/venv/bin/python Engine/Fixtures/make_klein_fixture.py /tmp/klein-fixture

# 2. The same computation in Swift
turbo-engine verify /tmp/klein-fixture
```

`verify` reports, stage by stage, the largest difference relative to the reference's scale: the
text encoder's prompt embeddings, the initial noise and ids for the fixture's seed, one
transformer pass, the scheduler's sigmas, the whole denoising loop, and the VAE decode. Anything
above 3% fails. Identical math on the same kernels lands well below that; a wrong reshape, a
swapped rotary pair or a missing cast shows up as a large error at the first stage it touches.

Then the real thing: generate the same prompt and seed with the app on the Python engine and on
the native one (rename the engine binary to switch), and compare the two PNGs.

## Protocol

Commands on stdin, one JSON object per line: `generate` (id, model, params), `load` (model),
`cancel` (id), `unload`, `shutdown`. Events on stdout: `ready`, `phase`, `progress`, `done`
(path, seed, size, seconds, peak_memory, timings per phase), `failed`, `cancelled`,
`model_loaded`, `unloaded`. Everything else goes to stderr, which the app shows as the engine log.
