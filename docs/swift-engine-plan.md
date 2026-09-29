# A 100% Swift engine: efficient, with few dependencies, and a sealed build

*Plan, September 2026. Written from a reading of the repository at `main` f13474e (version 1.3,
build 4); no code changed with it. The performance figures come from `README.md`,
`docs/native-engine.md` and `docs/generation-performance.md` (M1 Max); the observations on the
code refer to the files of the engine in `Engine/Sources/`.*

## Status

29 September 2026: Phase 0's first item, 1c, 1f and Phase 5 are done, the code side of Phase 2
(the sealed build) is in the tree, and every result is marked as made with generative AI; the rest
has not started.

- **Phase 5, the dependency diet.** The engine tokenizes on its own (`Tokenizer/BPETokenizer.swift`:
  byte-level BPE for Qwen2/3 and Ling, SentencePiece-style BPE with byte fallback for Gemma 3,
  Qwen3's chat template as the fixed string it renders for one user turn), and swift-transformers
  left the build with its eight packages: the engine resolves mlx-swift and the swift-numerics it
  brings, nothing else. The gate: `turbo-engine verify-tokenizers` compares every family's pipeline
  with `transformers` on 64 prompts (`Fixtures/tokenizers.json`, 32,464 ids): identical. Two
  Foundation traps were in the way: Swift's `String` merges canonically equivalent keys (Gemma's
  vocabulary has ";" and the Greek question mark as two tokens), so tokens are matched by bytes,
  and `JSONSerialization` drops a leading U+FEFF, so `tokenizer.json` has a parser of its own.
  Every fixture passes and real images and clips are bit for bit those of before; loading a Qwen
  tokenizer takes 0.27 s instead of 0.74.

- **1f, the memory a job needs.** The app refuses a generation this Mac cannot hold
  (`TurboMLX/Models/MemoryEstimate.swift`, the `notEnoughMemory` blocker): its peak is predicted
  from 48 measurements of the catalog's models (every image model at 512, 1024 and 2048 px or
  1536 px, with and without Save memory; SeedVR2 up to 16.8 MP; LTX from 1 to 10 s, 768 × 512 and
  1024 × 576, both packs), as the larger of the prompt's encoding and a line in the megapixels
  (for a clip, in megapixels and megapixels × latent frames, capped by the decode budget the
  engine gives a smaller Mac). The fit is within 0.5 GB of every measurement but Ming-Image with
  Save memory, kept at the encoding's 11.7 GB. A job is refused above the Mac's memory less 2.5 GB,
  and the message says what would make it fit: Save memory, a smaller size, a shorter clip, a
  smaller scale, another model. And the engine wires what a job uses for its length (mlx-swift's
  `WiredMemoryTicket`, as large as the GPU's recommended working set, as the Python worker did):
  the system's wired memory rose to 29 GB during a Klein job and fell back after it, at the same
  speed, with MLX's peak lower (10.1 against 12.7 GB at 1024 px), so the estimate errs on the side
  of starting.
- **Marking.** The engine writes IPTC's Digital Source Type into every PNG's XMP and into an XMP
  box of every MP4 (`Provenance.swift`): "created" from a prompt, "edited" from a picture unless
  the picture says it was itself generated. It is what Article 50 of the EU AI Act asks of
  generated media in machine-readable form.

- **Phase 2, what is done.** A second target, *TurboMLX Sealed* (a build that does not update
  itself and has no sandbox exception), on the same synchronized folder: `SEALED` compiles Sparkle
  out (`AppUpdater`, the Check for Updates menu, the Settings section) and the target does not
  link it; `TurboMLX-Sealed.entitlements` has the sandbox, network client, user-selected
  read-write and app-scope bookmarks, no temporary exception; `TurboMLX-Sealed-Info.plist` drops
  Sparkle's keys and answers export compliance (`ITSAppUsesNonExemptEncryption` false); its
  products go to `Release-Sealed`; the engine is
  signed with the identifier `<bundle id>.turbo-engine` (`TURBO_MLX_CHANNEL`). Models live in
  `ModelFolder`: by default `~/.cache/huggingface` in the GitHub build and the container's
  `Application Support/TurboMLX/Models` (out of Time Machine) in the sealed one, and in both a
  folder the user picks in Settings → Models, kept as a security-scoped bookmark opened before the
  engine starts (a Hugging Face home or a hub itself; the engine restarts to inherit it). The
  token: `HF_TOKEN`, else the keychain (Settings → Models), else, GitHub build only, the CLI's
  file. `Resources/PrivacyInfo.xcprivacy` declares no tracking, no data collected, and the reasons
  for UserDefaults (CA92.1), file dates (C617.1, 3B52.1) and disk space (E174.1, 85F4.1).
  A release script for the sealed build, kept out of the repository, archives it, checks it (no
  Sparkle, no temporary exception, the engine inherits the sandbox, the manifest is there),
  exports the installer package and uploads it. Checked: both targets build; the sealed build,
  run sandboxed under a test bundle ID, starts its engine, makes its models folder in the
  container, excluded from backups, with no sandbox denial; the GitHub build is unchanged by
  default.
- **Phase 2, what is left.** For the account holder: the certificates, identifier,
  provisioning profile, record and API key the sealed channel needs (the release script's header
  lists them), then a beta build on a clean Mac and the submission. In the code: a visual check
  of the two new Settings sections, of a folder chosen through the open panel with the engine
  reading from it, and of the refusal's message.

- **mlx-swift 0.32.2**, whose MLX core is the one mflux's venv and ltx-2-mlx run, replaced 0.31.6
  (core 0.31.1). Two deprecated calls changed (`asData(noCopy:)`), nothing else in the code. Every
  fixture passes, and what two MLX versions used to round differently now matches the references
  exactly: Klein's bf16 text encoder (3–4% before), Ming's text side, passes and loop (1–3%), LTX's
  video and audio decoders in bf16 (25% and 0.6%). One stage moved the other way, within
  tolerance: LTX's text connector, bit-identical before, is 1.5% off. Real images came closer to
  mflux's (PSNR at 512², same prompt and seed): Klein 29 → 39 dB, Z-Image 35 → 39 dB (4-bit) and
  38 → 44 dB (8-bit), Qwen-Image 25 → 36 dB, Qwen-Image Edit 60 dB, Ming 32 → 33 dB, SeedVR2 61 dB.
- **What the new core cost, and 1c.** MLX 0.32.2 runs a short 3D convolution (PR 3785) as one 2D
  convolution per kernel frame, all queued at once, each free to use Winograd with buffers for as
  many frames as three quarters of the GPU's working set hold (`winograd_batch_step`). Faster, but
  a still image's causal VAE did three convolutions where one suffices, and a clip's decode held
  Winograd buffers for whole tiles three times over: SeedVR2's first upscale took twice as long
  and peaked 6 GB higher, a 5-second LTX clip with *Save memory* peaked at 30 GB instead of 16.
  Two changes, both in the engine: a single frame goes through one 2D convolution (1c: the
  kernel's last frame for the Qwen-Image VAE, whose padding frames are zeros, which is bit for bit
  what MLX computes; the kernel's frames summed for SeedVR2, whose padding repeats the frame: 60.6
  dB from mflux against 61.4); and `turbo-engine` sets `MLX_CONV_WINOGRAD_TILE_BATCH=1` at launch,
  one frame per Winograd step, which only a clip has more than one of.
- **Before and after**, M1 Max 64 GB, the same requests through the protocol, one process per
  engine (mflux 0.20 on MLX 0.32.2 for comparison; its 1024² figures are §1.4's):

  | Model | Request | mlx-swift 0.31.6 | 0.32.2 | mflux |
  |---|---|---|---|---|
  | FLUX.2 Klein 4B q4 | 512², 4 steps | 11.2 s · 6.3 GB | 9.4 s · 6.3 GB | 11.5 s · 6.2 GB |
  | FLUX.2 Klein 4B q4 | 1024², 4 steps | 33.9 s · 12.7 GB | 28.6 s · 12.7 GB | 29.9 s · 10.6 GB |
  | FLUX.2 Klein 4B q4 | edit, 640 × 512 | 20.9 s · 6.6 GB | 17.4 s · 6.6 GB | |
  | Z-Image Turbo q4 | 512², 9 steps | 21.2 s · 7.4 GB | 20.0 s · 7.4 GB | 24.6 s · 7.3 GB |
  | Z-Image Turbo q4, Save memory | 1024², 9 steps | 95.0 s · 7.7 GB | 95.7 s · 7.7 GB | 106 s · 7.8 GB |
  | Z-Image Turbo q8 | 512², 9 steps | 23.4 s · 12.1 GB | 22.6 s · 12.1 GB | 23.5 s · 12.0 GB |
  | Qwen-Image 2512 q4, Save memory | 512², 10 steps, CFG | 62.8 s · 15.1 GB | 55.3 s · 16.0 GB | 65.5 s · 16.6 GB |
  | Qwen-Image Edit 2511 q4, Save memory | 320 × 256, 3 steps | 18.6 s · 16.3 GB | 17.4 s · 16.3 GB | 19.9 s · 28.8 GB |
  | Ming-Image te5, Save memory | 512², 12 steps | 43.9 s · 11.7 GB | 35.2 s · 11.7 GB | 40.4 s · 11.7 GB |
  | SeedVR2 3B | 640 × 512 → 1280 × 1024 | 19.7 s · 9.8 GB | 10.4 s · 10.7 GB | 32.8 s · 17.2 GB |
  | SeedVR2 3B | 768² → 3072² | 135.9 s · 13.2 GB | 79.5 s · 13.2 GB | |
  | LTX-2.3 q4 | 768 × 512, 25 frames | 67.9 s · 24.8 GB | 60.4 s · 26.7 GB | |
  | LTX-2.3 q4, Save memory | 768 × 512, 5 s | 227.4 s · 15.9 GB | 205.7 s · 18.2 GB | |
  | LTX-2.3 q8 | 768 × 512, 5 s | 294 s · 38 GB | 218.5 s · 37.4 GB | |

  Without the two changes 0.32.2 gave SeedVR2 38.1 s · 15.6 GB and the 5-second clip
  221.6 s · 30.2 GB. The engine is now ahead of mflux on every family, so §1.4's gaps and most of
  §8's targets are behind us; Klein's peak at 1024² is not.
- **A lesson for the next update:** an MLX release can change the memory a model needs without
  changing a pixel (here through a heuristic sized on the Mac's whole working set). Updating
  mlx-swift means re-measuring the peaks, not only running `verify`: the `bench` of Phase 0 is
  the tool for it.

## Short answer

**Yes, and it already is, for the part that matters.** The native engine exists, is Swift on
mlx-swift, runs every family of the catalog (FLUX.2 Klein, Z-Image Turbo, Qwen-Image 2512,
Qwen-Image Edit 2511, Ming-Image, LTX-2.3, and since 3378198 the SeedVR2 upscaler), loads the mflux checkpoints without conversion, is
checked stage by stage against mflux and ltx-2-mlx, and the app carries no Python: no
interpreter, no venv, no `uv`. The bundle went from 77 to 41 MB.

What is left, to "detach completely" and to be efficient, lies on four axes:

1. **Speed.** *Updated 29 September:* with mlx-swift 0.32.2 and the single-frame decoders
   (Status) the engine is faster than mflux on every family: Klein −18 % at 512² and −4 % at
   1024², Z-Image −4 to −19 %, Qwen-Image −16 %, Qwen-Image Edit −13 %, Ming −13 %, SeedVR2 3×.
   When this plan was written it was slower on Klein (+12 % at 1024², +40 % at 512²) and Ming
   (+15 %). What is left in the code (no `compile`, RoPE tables rebuilt at every step, a RoPE
   rotation spread over a dozen kernels) is now headroom below mflux rather than a gap, and
   Klein's higher peak at 1024² remains. All of it fixable in Swift, with the existing `verify`
   as the safety net.
2. **Python.** It only remains outside the app: the fixture generators (`Engine/Fixtures/*.py`),
   the reference worker (`Engine/Reference/turbo_worker.py`) and, indirectly, the catalog's
   checkpoints, which are conversions made by others with Python tools. The workflow can be
   brought to a point where Python is only needed to regenerate a reference when mflux
   changes, and a Swift converter makes the app independent of other people's checkpoints too.
3. **Dependencies.** The engine has two direct ones (mlx-swift, swift-transformers) but
   swift-transformers brings eight more. An in-house tokenizer (BPE over `tokenizer.json`)
   leaves the engine with mlx-swift alone, which is Apple's and cannot be replaced. The app has
   only Sparkle, which the sealed build leaves out.
4. **A sealed build**, which does not update itself and has no sandbox exception, as some
   distribution channels require. Feasible. Five things block it today: the "temporary
   exception" entitlement for `~/.cache/huggingface`, Sparkle (and its two mach-lookup
   exceptions), the Hugging Face token read from disk, the missing privacy manifest, and a
   memory handling that lets generations start when they do not fit in RAM. None of them needs
   a change of architecture: the sandboxed child process (`turbo-engine` in `Contents/MacOS`,
   `inherit` entitlement) is the form Apple prescribes for a helper.

Recommended order: measure and update MLX (days) → efficiency without changing pixels (1–2
weeks) → the sealed build (1–2 weeks, app side, can run in parallel) → speed that changes pixels, as
options (bf16, Lightning, live preview) → verification without Python → dependency diet →
converter. Every phase leaves the app shippable.

## 1. Where things stand

### 1.1 What already runs in Swift

| Piece | Where | Notes |
|---|---|---|
| JSON-lines protocol, child process, cancel, timings | `Engine/Sources/TurboEngineCore/{Engine,Protocol}.swift`, `Sources/turbo-engine/Server.swift` | the same protocol as the old Python worker; `cancel` read on a thread of its own |
| mflux checkpoint loading (shards, index, bits and group size inferred from the shapes, stacked experts) | `Checkpoint.swift` | no conversion |
| Seven families (SeedVR2 since 3378198), module for module after mflux / ltx-2-mlx | `Families/` (~7 000 lines) | text encoders, DiTs, VAEs, schedulers, tiling, MoE, Pillow's resizes bit for bit |
| Tokenizers | swift-transformers (`Qwen3Prompter.swift`) | Qwen2/3 BPE, Gemma; chat templates through swift-jinja with a fixed-string fallback |
| Output | `ImageOutput.swift` (PNG through ImageIO), `VideoOutput.swift` (MP4, H.264 + AAC, through AVAssetWriter) | frames go to the writer as the decoder produces them |
| Verification | `turbo-engine verify <fixture>` | every stage against mflux, 3 % tolerance; LTX against ltx-2-mlx on a real pack |
| Downloads, HF cache, sandbox, updates | the app (`Services/`) | all Swift; Sparkle for the updates |

### 1.2 Where Python remains

| What | Where | For | When it is really needed |
|---|---|---|---|
| `turbo_worker.py` | `Engine/Reference/` | mflux behind the same protocol, to compare real images (PSNR) | whenever a new comparison with mflux is wanted |
| `make_{klein,zimage,qwen_image,qwen_image_edit,ming}_fixture.py` | `Engine/Fixtures/` | tiny random-weight checkpoints plus the references `verify` compares | only when a reference changes (a new family, a new mflux revision). The fixtures they produce are safetensors + JSON, which Swift reads on its own |
| `make_ltx_fixture.py` | `Engine/Fixtures/` | runs ltx-2-mlx on a real pack and records its stages | same |
| `requirements*.txt` | `Engine/Fixtures/` | the mflux and ltx-2-mlx pins | same |
| The `mflux-community/*`, `joeynyc/*`, `dgrauet/*` checkpoints | Hugging Face (the catalog) | conversions and quantizations made by others with Python tools | every new model depends on someone converting it |
| Nothing | app, build, runtime | | never |

A contributor who does not touch the references already needs no Python, but this is not
organized: the fixtures have to be generated locally in a venv.

### 1.3 Dependencies

| Target | Direct | Transitive | License | For |
|---|---|---|---|---|
| App | Sparkle 2.10 | | MIT | updates from the GitHub feed |
| Engine | mlx-swift 0.32.2 (0.31.6 until 29 September) | (vendors MLX's C++ core and Metal kernels) | MIT (Apple) | everything; not replaceable |
| Engine | swift-transformers 1.3.4 (`Tokenizers`), until Phase 5 | swift-jinja, swift-huggingface, swift-collections, swift-crypto, swift-asn1, swift-numerics, yyjson, EventSource | Apache 2.0 | the tokenizers only; its `Hub` module (network, downloads) is dead weight in an app that downloads by itself |

Ten packages resolved for the engine (eleven with mlx-swift 0.31.6, which also pulled
swift-argument-parser), one of them used for a single thing.

### 1.4 Measured performance (M1 Max, native against mflux 0.20 / ltx-2-mlx)

*As of the plan's writing, with mlx-swift 0.31.6; Status has the figures since 0.32.2.*

| Model | Size | Native | Reference | Peak, native / ref. |
|---|---|---|---|---|
| FLUX.2 Klein 4B q4 | 1024² | 33.4 s | 29.9 s | 12.7 / 10.6 GB |
| FLUX.2 Klein 4B q4 | 512² | 15.0 s | 10.7 s | |
| Z-Image Turbo q8 | 512² | 22.6 s | 24.4 s | |
| Z-Image Turbo q4, Save memory | 1024² | 92 s | 106 s | 8.2 / 7.8 GB |
| Qwen-Image 2512 q4 | 512² | 121 s | 121 s | |
| Ming-Image te5 | 512² | 43.6 s | 37.8 s | |
| Qwen-Image Edit q4 | 672 × 880, 20 steps | ~15 min (the Mac was busy building) | 12.4 min | |
| LTX-2.3 q4, Save memory | 768 × 512 × 49 | 97 s | 97 s | 15 GB |

Image parity: PSNR 27–35 dB at 512² (the differences are the bf16 rounding of two MLX
versions), 30 dB for Z-Image q4 at 1024². The picture agrees with the documents' thesis: the
language does not matter; kernels, precision, steps and the MLX version do.

## 2. Where the engine loses time and memory (read in the code)

In order of what it is worth.

1. **No `compile`.** There is no call to `compile` in `Engine/Sources`; mflux compiles the whole
   step for FLUX.2 and Z-Image. Without it every step rebuilds the lazy graph (thousands of
   operations) and the elementwise chains (modulation `(1 + scale) · norm(x) + shift`, gates,
   SwiGLU, RoPE) come out as separate kernels. It weighs most where the GPU work is small:
   exactly Klein's profile at 512² (+40 %). mlx-swift has `compile(inputs:outputs:shapeless:)`.
   Caveat: mflux does not compile on M1/M2 below Max; to be measured, not assumed.
2. **RoPE tables rebuilt at every pass.** `Flux2Transformer.callAsFunction` recomputes
   `posEmbed(imageIds)` and `posEmbed(textIds)` on every call (`Flux2Transformer.swift`, line
   ~352); `QwenImageTransformer` rebuilds `rope(grids:textLength:)` at every pass;
   `LTXTransformer` builds four `LTXRope` per pass. The ids change neither between steps nor
   between the two passes of CFG: a cache per shape computes them once per generation. The
   S3-DiT (Z-Image, Ming) already keeps its tables precomputed (`S3DiT.swift`,
   `S3DiTRopeEmbedder`).
3. **The RoPE rotation in a dozen kernels.** `Flux2Rope.apply` (`Flux2Transformer.swift`, lines
   104–116) and `S3DiTRope.apply` (`S3DiT.swift`, lines 76–92): a cast to float32, a reshape to
   pairs, two strided slices, four mul/adds, `stacked`, a reshape, a cast. For q and for k, in
   every block (Klein: 25 blocks, 50 applications per pass). Candidates: tables already in the
   interleaved full-width layout, so the rotation is `x·cos + swap(x)·sin`; `compile`, which
   fuses the elementwise part; or a custom kernel through `MLXFast.metalKernel`. The Qwen3 and
   Qwen2.5 text encoders (`rotateHalf`) can use the fused `MLXFast.RoPE` as Gemma already does
   (`Gemma3TextEncoder.swift`, lines 110–112).
4. **MLX core 0.31.1 against 0.32.2.** *Done on 29 September (see Status).* The engine is on mlx-swift 0.31.6 (core 0.31.1); mflux is
   on 0.32.2. **mlx-swift 0.32.2 exists and vendors that very core 0.32.2** (checked in the tag's
   submodule): it brings the small-depth conv3d 3× faster (MLX PR 3785, which touches the
   Qwen/Ming decoders and LTX's VAE), the kernels for M5's neural accelerators and the Metal
   completion handler that rethrows a GPU reset instead of aborting. `Package.swift` asks for
   `from: "0.31.6"`, so it is a `Package.resolved` bump. To know: in 0.32.2 the synchronous
   `Memory.withWiredLimit` is deprecated and does nothing; the wired limit goes through a
   ticket-based, asynchronous `WiredMemoryManager` (`Source/MLX/WiredMemory.swift`).
5. **Conv3d on a single frame.** *Done on 29 September for Qwen-Image, Ming and SeedVR2 (see
   Status): with MLX 0.32.2 it also cost memory.* `QwenCausalConv3D` (`QwenImageVAE.swift`, lines 12–30) runs a
   real 3D convolution on one frame preceded by two frames of zeros: the result is identical to
   a Conv2d with the kernel's last temporal slice (`weight[:, 2]`), at a third of the
   multiplications. Today it also goes through core 0.31's slow path. A still-image decoder
   built with Conv2d (weights sliced at load time) is exact and cheaper: the decode is worth
   6.7 s/MP on Qwen-Image and 3.1 s/MP on Ming (`TimeEstimate.swift`), more at 1536–2048.
6. **Explicit attention in the Qwen/Ming VAE mid block** (`QwenImageVAE.swift`, lines 83–105):
   `matmul` + `softmax` over every pixel of the latent frame. At 1024² (128 × 128 tokens) the
   score matrix is 16384² × 4 bytes = 1 GB in float32 per frame, the reason *Save memory* has
   to tile. `MLXFast.scaledDotProductAttention` never materializes it and is faster; mflux
   runs the explicit version, so parity has to be rechecked (only the rounding differs).
7. **Float32 residual stream in Z-Image and Qwen-Image.** Faithful to mflux (`S3DiT.swift`,
   `keepsWeightsPrecision = false`; `QwenImagePipeline.initialLatents` in float32; the comment
   at the top of `QwenImageTransformer.swift`). mflux issue 761 measures 1.4× on the loop in
   bf16, and on M5 float32 bypasses the neural accelerators. It is the single largest lever the
   engine controls without waiting for upstream, but it changes the pixels slightly: behind a
   PSNR gate and a visual check on text, as an option or as the default after review.
8. **GQA through `repeated`** in the Qwen3 and Qwen2.5 text encoders (`Qwen3TextEncoder.swift`
   lines 89–93, `Qwen25TextEncoder.swift` lines 49–53): the fused SDPA handles grouped heads
   itself (Gemma already does). Only counts in prompt encoding: small.
9. **Memory: no wired limit, no predicted peak.** *Done on 29 September (1f, see Status).* The Python worker had `wired_memory()`; the
   engine has none. Without it macOS can page the weights out under pressure and a 10 s step
   takes 60 (`docs/generation-performance.md`, §7). And today the app lets a Qwen-Image at
   2048² start on a 16 GB Mac: the engine dies of a Metal OOM (fatal, in either language) and
   the app reports it. The catalog's measured peaks scale with the pixels: a generation can be
   refused before it starts. A review of the sealed build would need this too.
10. **Klein's higher peak** (12.7 against 10.6 GB): hypotheses to measure with
    `Memory.snapshot()` per phase: the default `cacheLimit` (unchanged outside Save memory),
    the float32 temporaries of the RoPE, the absence of `compile`.
11. **Synchronization per step.** `eval(latents)` then `progress` in every loop: right for
    progress and cancel. An `asyncEval` on the next step's graph would overlap CPU and GPU:
    1–2 %, low priority.
12. **Output.** The engine writes the PNG (ImageIO) and the app decodes it again; the `save`
    timing shows it, under a second. Handing over raw pixels (a memory-mapped file) with the
    PNG written in the background would remove it. Low priority.

What is already done well and should stay: cancellation between steps and tiles; the prompts
of the queue encoded ahead while the encoder is resident; the deterministic release of the
encoder in Save memory (no `gc.collect`); Ming's MoE with expert-sorted gathered quantized
matmuls (`SwitchLayers.swift`, as mlx-lm); LTX with an `eval` every 8 blocks and its tiled
decode at `cacheLimit = 0`; the unconditional pass skipped at guidance 1 in Qwen-Image (which
mflux does not do).

## 3. "100% Swift": the three levels that remain

### 3.1 Verification without Python

The fixtures are already static artifacts (safetensors + `fixture.json`): the only thing that
needs Python is producing them. Freeze them:

- Publish the fixtures as versioned artifacts (assets of a GitHub release, or a Hugging Face
  dataset repository such as `rinste/turbo-mlx-fixtures`): the image families weigh tens of
  MB, LTX about 0.5 GB (`gemma_states.safetensors` alone is 385 MB). Each fixture carries the
  mflux / ltx-2-mlx revision that generated it.
- `turbo-engine verify --all` downloads (or reads from the cache) and checks every family in
  one command; `turbo-engine compare a.png b.png` computes the PSNR instead of the comparison
  by hand.
- Frozen mflux reference images per (model@revision, prompt, seed, size), next to the fixtures:
  the "real image" comparison no longer needs mflux to run.
- A tokenizer parity corpus (prompt → ids, JSON), generated once with Python's `tokenizers`:
  it serves Phase 5 and guards against regressions of swift-transformers.
- CI: a GitHub Actions job on a macOS arm64 runner that builds the engine and runs `verify` on
  the small fixtures (they are tiny models, seconds of GPU). Whether Metal is available in the
  runners' virtualized GPU is to be checked with a trial job. There is no CI today.

Result: Python is only for whoever regenerates a reference because mflux or ltx-2-mlx changed.
Everybody else (and the release) never installs it.

### 3.2 Checkpoints without Python: `turbo-engine convert`

mflux's format is plain and the engine already reads all of it (`Checkpoint.swift`: shards
≤ 2 GB with `model.safetensors.index.json`, `quantization_level` and `mflux_version` in the
metadata, `weight`/`scales`/`biases` tensors in the affine scheme with groups of 64). Writing
it from Swift is symmetric:

- read the original safetensors (diffusers / Hugging Face layout, bf16) of the transformer,
  the text encoder and the VAE;
- map the keys onto the module tree (which already mirrors mflux: it is the map mflux applies
  the other way round);
- quantize with MLX's `quantize(model:groupSize:bits:mode:)`, the same quantizer as mflux's
  `nn.quantize`, the same predicate (Linear and Embedding), the same bits;
- write the shards, the index and the metadata.

A strong check at no cost: convert FLUX.2 Klein 4B from the original weights and compare it
tensor for tensor with `mflux-community/flux2-klein-4b-mflux-q4`: same bf16 source, same
quantizer → equality expected (at most rounding differences between MLX versions, which
`verify` and the PSNR measure). Then:

- merging a LoRA (Lightning for Qwen-Image) in Swift: a matmul and a requantization;
- "Add Model…" accepting an original repository, quantized on first launch (a 2–3× larger
  download: optional, not the default);
- possibly the project's own checkpoints published on Hugging Face, so the catalog depends
  neither on mflux-community nor on conversions with other people's choices (Ming te5 is a
  third party's conversion).

Order: Klein and Z-Image (keys close to diffusers), then Qwen-Image, last Ming (MoE, connector,
a conversion with choices of its own) and LTX (dgrauet's packs have a layout of their own).

### 3.3 Dependencies: an in-house tokenizer, Sparkle only in the GitHub build

swift-transformers is used for one thing: `AutoTokenizer.from(modelFolder:)` and
`applyChatTemplate`. An in-house tokenizer that reads `tokenizer.json`:

- byte-level BPE (GPT-2 style) for Qwen2/Qwen3 and for Ling (Ming): the pre-tokenization
  regex (ICU / `NSRegularExpression` supports `\p{L}`, `\p{N}` and inline flags), the
  byte-to-unicode table, merges ordered by rank, added and special tokens;
- SentencePiece-style BPE with byte fallback for Gemma 3 (`Replace(" ", "▁")` normalizer,
  `<bos>`), as Hugging Face's "fast" conversion describes it;
- the chat templates as fixed strings: the code already has them as fallbacks
  (`Qwen3Prompter.templated`, `TemplatePrompter`); swift-jinja becomes unnecessary.

Cost: about a week, with the parity corpus (multilingual, emoji, spaces and newlines, long
prompts) at 100 % identity as the gate. Benefit: the engine depends on mlx-swift alone, −10
packages, a shorter build, less supply-chain surface, and no networking library in the process
that talks to the GPU. It is the least urgent item for the end user and the most consistent
with "few external dependencies"; if the week is not worth it, a pinned swift-transformers is
acceptable.

Sparkle stays in the GitHub channel and leaves the sealed one (§5).

## 4. Architecture: a separate process, confirmed

`docs/native-engine.md` argues it and the code confirms it: a Metal OOM is fatal for the
process; an engine of 6–22 billion parameters near the memory ceiling will meet it; today the
engine dies, the app survives and shows the log. In-process would lose all that for a gain the
code does not justify (the PNG round trip is under a second). The sandboxed child process is
also the form Apple prescribes for a helper in a sealed app (see §5).

A possible refinement, not a necessary one: an XPC service (`Contents/XPCServices`) with a typed
protocol and a lifecycle managed by launchd. It can come later, without touching the families.

A note on the wired limit (item 9 of §2): with mlx-swift 0.32.2 the ticket is asynchronous and
the engine is synchronous on one thread (`Server.run` → `Engine.generate`). Either the ticket
is held on a `Task` for the length of the generation, or the C API is called through `Cmlx`.
To decide when updating.

## 5. The sealed build: what changes

| Blocker | Today | What to do |
|---|---|---|
| The `temporary-exception.files.home-relative-path.read-write` entitlement for `~/.cache` (`TurboMLX.entitlements`), which a sealed build cannot have. *Done (29 Sep): `ModelFolder`, see Status.* | the models live in `~/.cache/huggingface/hub`, shared with mflux | Models in the container by default (`…/Application Support/TurboMLX/Models/hub`, the same hub layout: `ModelLocator` and `HubDownloader` do not change, only `huggingFaceHome` does; exclude the folder from Time Machine). An optional shared folder the user picks with `NSOpenPanel` + a security-scoped bookmark (`FolderAccess` exists already; it needs `files.user-selected.read-write`), which covers whoever wants to share with mflux: the user can pick `~/.cache/huggingface` itself. Migration: on first launch, offer to move or to reuse the existing downloads. |
| The Hugging Face token read from `~/.cache/huggingface/token` and from the environment (`HubDownloader.storedToken`). *Done (29 Sep): `HuggingFaceToken`.* | | A field in Settings, stored in the Keychain; `HF_TOKEN` stays as an override for development. |
| Sparkle: a sealed build does not update itself; two `temporary-exception.mach-lookup` entitlements; `SUFeedURL`/`SUPublicEDKey` in `TurboMLX-Info.plist`. *Done (29 Sep): the "TurboMLX Sealed" target.* | `AppUpdater.swift`, the "Check for Updates…" menu, Settings → About | A second Xcode target ("TurboMLX Sealed") on the same synchronized folders, without the Sparkle product linked, with entitlements and Info.plist of its own and `SWIFT_ACTIVE_COMPILATION_CONDITIONS = SEALED`; `#if !SEALED` around `AppUpdater`, the menu and the Settings section. The GitHub channel stays as it is; the sealed build gets a release script of its own. |
| The `turbo-engine` helper | `Contents/MacOS`, entitlements `app-sandbox` + `inherit` (`Engine/turbo-engine.entitlements`) | Already compliant: Apple asks that a helper have only those two entitlements. Signed with the channel's certificate and the app's profile from the Xcode archive. To be checked early with a beta build. |
| Privacy manifest | missing. *Done (29 Sep).* | `PrivacyInfo.xcprivacy` with the "required reason APIs" in use: `UserDefaults` (CA92.1), file dates (`ModelLocator`: `.contentModificationDateKey`; `QwenImageEditPipeline.pictureKey`: `attributesOfItem` → C617.1 / 3B52.1), disk space (`DownloadCenter.checkSpace`: `.volumeAvailableCapacityForImportantUsage` → E174.1 / 85F4.1). No tracking, no data collection. Codes to confirm against Apple's list at the time. |
| Memory and review | the engine dies when memory runs out; the app says so | A predicted peak before starting (the catalog's peaks × pixels, per family) → a refusal with a clear message; the requirements stated on the product page; the 16 GB model (Z-Image q4) is there already. A reviewer on an 8–16 GB Mac must be able to generate something on first launch. |
| Architecture | arm64 | Fine: Apple-silicon-only apps are accepted (Intel Macs do not see them). |
| 5–38 GB downloads | | They are data, not code (the rules are about executable code). Apps that download models into their container already exist. The space needed is already shown. |
| Generated content | | Describe the use on the product page; the models' licenses are already in About (the LTX-2 Community license has use restrictions that bind the outputs too: to be said). |
| Build number | `CURRENT_PROJECT_VERSION` grows for Sparkle | Holds for the sealed channel as well. |

What does not change: the sandbox and the hardened runtime (already on),
`LSMinimumSystemVersion` 15, the container migration (already done), `TURBO_MLX_HOME` for
trying the first launch.

## 6. The plan, by phase

Every phase has a gate and leaves the app shippable. The estimates are of work, with a Mac at
hand to measure.

### Phase 0 — Measure and update (days)

- mlx-swift → 0.32.2 (`Package.resolved`); rebuild; `verify` on the six fixtures; re-measure
  the table of §1.4. The cheapest item with the most likely return (decoders, M5). *Done (29
  September): the results are under Status.*
- `turbo-engine bench`: the catalog × 3 sizes × fixed prompts and seeds, seconds per phase and
  peak, in a table in `docs/`. The performance document has asked for it since September.
- One Metal System Trace (Instruments) per family: the top kernels and the gaps between them
  decide Phase 1.

Gate: `verify` green on every family; a before/after table.

### Phase 1 — Efficiency without changing the pixels (1–2 weeks)

a. RoPE tables per shape, computed once per generation (Klein, Qwen-Image, LTX).
b. A fused RoPE rotation or `compile` of the step (Klein, Z-Image); measured on M1/M2 base too.
c. The Qwen/Ming single-frame decoder with Conv2d (exact). *Done (29 September), SeedVR2's too.*
d. Fused SDPA in the VAE mid block.
e. GQA without `repeated` in the text encoders.
f. A wired limit for the length of the generation + predicted peak and refusal. *Done (29 September).*
g. `asyncEval` on the next step (measure; keep only if it shows).

Gate: `verify` ≤ 3 % per stage; PSNR against the reference images no worse than today
(27–35 dB); times in the bench. Expected: Klein 1024² from 33 to ≤ 30 s (parity with mflux),
Klein 512² −20–30 %, Qwen/Ming decode −30–50 %, Ming 512² towards mflux's 38 s, a lower Klein
peak.

### Phase 2 — The sealed build (1–2 weeks, app side; in parallel with Phase 1)

The whole table of §5: the models in the container + the optional shared folder + migration;
the token in the Keychain; a target without Sparkle with entitlements and Info.plist of its
own; `PrivacyInfo.xcprivacy`; the predicted peak and its messages; a release script for the
sealed channel; a beta build; submission. Gate: the beta build installs and generates on a clean
16 GB Mac; review answers.

### Phase 3 — Speed that changes the pixels, as options (1–2 weeks)

- A bf16 stream for Z-Image and Qwen-Image: ~1.4× on the loop, more on M5; a "Fast" option or
  the default after a comparison on text-heavy prompts.
- Qwen-Image 2512 Lightning (4 steps, no CFG) as a catalog entry of its own: ~10× on the
  slowest model (the engine already skips the unconditional pass at guidance 1). It needs a
  pre-merged checkpoint: from Phase 6, or once with mflux.
- TeaCache / First-Block Cache for Qwen-Image at 20 steps (1.5–2×; nothing on the 4–9-step
  models).
- Live preview: a linear latents → RGB projection every N steps, a `preview` event with a small
  PNG on the protocol, shown by the app; stopping a wrong image at step 2 of 9 is the largest
  gain for whoever iterates.

Gate: PSNR + a visual review on text-heavy prompts; the options that change the model are
catalog entries, not replacements.

### Phase 4 — Verification without Python (days)

§3.1: fixtures and reference images published, `verify --all`, `compare`, the tokenizer corpus,
a trial CI job. Gate: a clean clone without Python builds and verifies.

### Phase 5 — Dependency diet (1 week)

§3.3: the in-house tokenizer, swift-transformers and its nine transitive packages gone,
`make-acknowledgements` updated. Gate: the parity corpus at 100 %, `verify` green.

### Phase 6 — A Swift converter (2–3 weeks)

§3.2: `turbo-engine convert`, the tensor-for-tensor check against Klein's mflux checkpoint,
then Z-Image and Qwen-Image; LoRA merging; "Add Model…" from original repositories. Gate:
`verify` and PSNR against the corresponding mflux checkpoint.

## 7. What not to expect, and the risks

- **Swift instead of Python in the loop does not give a 2×.** The same Metal kernels, the same
  core. The 2–10× come from the steps (Lightning), the precision (bf16), the MLX version on
  M5. Phase 1 closes the gap and removes the overhead; it does not change the order of
  magnitude.
- **Every numerical optimization costs a verification.** The gate exists (`verify` at 3 %, PSNR
  on real images) and must be used for every item; a DiT that "runs" with a wrong epsilon
  produces plausible, wrong images.
- **mlx-swift is pre-1.0.** The wired-memory API already changed between 0.31.6 and 0.32.2. Pin
  by version, update only with the bench and `verify` at hand.
- **`compile` on M1/M2 base.** mflux avoids it there: measure before enabling it everywhere.
- **The sealed channel's review has unknowns** (generative content, the helper, the large
  downloads). A beta build at the end of Phase 2 finds them out at the lowest cost. The GitHub
  channel depends on none of them.
- **The in-house tokenizer** can diverge on exotic text: the corpus has to be broad, and the old
  tokenizer stays available until the corpus is at 100 %.
- **The converter** meets original layouts that differ from family to family: start with Klein,
  where the comparison with the mflux checkpoint is immediate.

## 8. Metrics and targets

| Metric | Today | Target | From |
|---|---|---|---|
| Klein 4B q4, 1024² | 33.4 s (28.6 s on 29 Sep) | ≤ 30 s | Phases 0–1 |
| Klein 4B q4, 512² | 15.0 s (9.4 s on 29 Sep) | ≤ 11 s | Phase 1 (overhead) |
| Z-Image q4, 1024², Save memory | 92 s | 65–70 s | Phase 3 (bf16) |
| Qwen-Image q4, 1024², Save memory | 589 s | 60–90 s with Lightning; ~450 s with bf16 | Phases 3 and 6 |
| Ming te5, 1024 × 576 | 76 s (mflux) | ≤ 76 s | Phase 1 |
| Qwen / Ming decode | 6.7 / 3.1 s per MP (about half on 29 Sep, 1c) | −30–50 % | Phase 1 (c, d) |
| Klein peak, 1024² | 12.7 GB | ≤ 10.6 GB | Phases 0–1 |
| Engine packages | 11 (10 since mlx-swift 0.32.2; 2 since Phase 5: mlx-swift and its swift-numerics) | 1 (mlx-swift) | Phase 5 |
| Python in the workflow | fixtures and reference, locally | only to regenerate a reference | Phase 4 |
| Sealed channel | none (a target since 29 Sep) | a sealed target + a beta build | Phase 2 |

The time targets are expectations to measure with Phase 0's bench, not promises: Phase 0
itself may move them.
