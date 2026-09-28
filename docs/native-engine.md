# A native engine: MLX Swift instead of the Python worker

*Proposal, September 2026. Written for a decision; facts checked against the sources linked,
mostly GitHub, on 27 September 2026.*

## Status

27 September 2026: Phase 0 and the first cut of Phase 1 are in the tree. The app downloads
models itself (`Services/HubDownloader.swift`), routes each family to an engine
(`BackendController.engineKind(for:)`), and its data model, history and viewer know about video.
`Engine/` holds the native engine: FLUX.2 Klein (4B and 9B geometry) on MLX Swift, loading the
mflux checkpoints unchanged, behind the JSON protocol, with a `verify` mode that compares every
stage with mflux on a small random-weight checkpoint (`Engine/Fixtures/make_klein_fixture.py`).
It was written without a Mac at hand; checked on an M1 Max (64 GB) the same day:

- **Build.** It compiled as written (Xcode 26.6, mlx-swift 0.31.6, whose MLX core is 0.31.1);
  `xcodebuild` needs `-skipPackagePluginValidation` for mlx-swift's plug-in
  (`scripts/build-engine.sh` passes it).
- **`verify`** passes. Noise, ids, sigmas and timesteps match exactly; one transformer pass, the
  denoising loop and the VAE are within 0.8%. The text encoder is 3–4% off in bf16 but agrees to
  about 1e-6 in float32: the math is the same and the gap is the rounding of two MLX versions'
  kernels, which the fixture's random weights amplify. Two differences from mflux turned up and
  were fixed: FLUX.2's schedule is shifted by the image's token count (`requires_sigma_shift`),
  and the pixels are denormalized in the decoder's dtype.
- **Same prompt and seed**, through the app's protocol, with `flux2-klein-4b-mflux-q4`: the same
  image as mflux (composition, text, lighting), with small local differences (PSNR 23–27 dB at
  1024 × 1024 and 1536 × 864), what 4 steps make of bf16 rounding that differs between kernels.
  At 1024 × 1024 the native engine took 33.4 s against 29.9 s and peaked at 12.7 GB against
  10.6 GB: the older MLX core is the first suspect.
- Klein's base checkpoints stay on mflux: the native engine has no classifier-free guidance yet.

Later the same day, Phase 2 went into the tree, again without a Mac at hand: Z-Image Turbo,
Qwen-Image 2512 and Ming-Image on the native engine (`Engine/Sources/TurboEngineCore/Families/`),
each loading the catalog's mflux checkpoint unchanged, with a fixture generator and a `verify`
stage of its own, and Klein's base checkpoints with their classifier-free guidance. The app routes
every built-in family to `turbo-engine` when the binary is there; the Python engine is only
started without it. Checked on the M1 Max on 28 September:

- **Build.** One error: the S3-DiT read its patch size off the transformer instead of its config.
- **Loading.** Two problems, fixed in `WeightLoading` and `QwenImageVAE`. MLXNN replaces the
  modules of a list only when it is given the list's first element, so the S3-DiT's
  `cap_embedder` (an RMSNorm, then a quantized linear) kept Z-Image and Ming from loading; the
  loader now completes such lists. And the catalog's checkpoints store the Qwen VAE's norms flat
  ([C], mflux's `reshape_gamma_to_1d`) where the fixtures, saved straight from mflux's modules,
  keep [C, 1, 1, 1]: the port takes both, as mflux does. Only a real checkpoint shows that one.
- **`verify`** passes for the four families. Z-Image and Qwen-Image are within 0.6% at every
  stage (Qwen-Image within 1e-5); Ming's text side (2.7%) and decode (2.4%) run in bf16 and carry
  the two MLX versions' rounding. One difference from mflux turned up there and was fixed: the
  VAE's attention multiplies the scores by a float32 scale, which carries Ming's bf16 decode on
  in float32 from the mid block, where the port had stayed in bf16 (3.6% off).
- **Same prompt and seed**, through the app's protocol at 512 × 512: the same image as mflux for
  every family (composition, text, alpha), PSNR 35 dB for Z-Image q8, 31 dB for Ming, 29 dB for
  Qwen-Image, 27 dB for Klein. Native against mflux: Z-Image 22.6 s against 24.4 s, Qwen-Image
  121 s against 121 s, Ming 43.6 s against 37.8 s, Klein 15.0 s against 10.7 s; peaks within
  1.5 GB of mflux's, lower for Ming and Qwen-Image.
- **Z-Image Turbo 4-bit**, the catalog's model for 16 GB Macs, checked later that day (at the
  commit the catalog now names): the same prompt and seed at 1024 × 1024 with Save memory give the
  same image as mflux, PSNR 30.1 dB, in 92 s against 106 s, peaking at 8.2 GB against 7.8 GB.
- **Save memory at 1024 × 1024**, native: Ming-Image peaks at 12.5 GB (151 s), Qwen-Image at
  16.4 GB (589 s), against the 14.7 and 21.3 GB measured with mflux; no seams between the tiles.

The app's build embeds the engine in the bundle (the *Embed turbo-engine* phase,
`scripts/embed-engine.sh`, signed with the app), and on 28 September Python left the app, the
first half of Phase 3: `uv`, `setup_backend.sh`, the install UI and the Python settings are gone,
the build fails without the engine, and the app went from 77 to 41 MB. `turbo_worker.py` moved to
`Engine/Reference/`, outside the app, as the reference real images are compared with (the check
that found the flat VAE norms), next to the fixture generators; the mflux revision both run with
is in `Engine/Fixtures/requirements.txt`.

The second half followed the same day: no Mac App Store, and the App Sandbox on, with the shared
hub cache. The app reaches `~/.cache/huggingface` through a temporary-exception entitlement for
that path, which the App Store would refuse but which needs no prompt and works where the folder
does not exist yet; the engine is signed to inherit the sandbox, so it reads the models there
too. Local model folders are kept with security-scoped bookmarks, opened before the engine
starts. `ShellEnvironment`'s shell probe is gone: an `HF_HOME` elsewhere is no longer followed,
and the token comes from the file the Hugging Face CLI saves. A container migration manifest
moved the history and the settings into the container on the first sandboxed launch (checked:
the 42 images and the selected model came along, and the engine loaded Ming-Image from the
shared cache with no sandbox denial). Builds are signed with the Developer ID certificate, local
ones too, so the container keeps recognizing the app from one build to the next. Phase 3 is
done.

Phase 4 began on 28 September with LTX-2.3 distilled (`Engine/Sources/TurboEngineCore/Families/LTX/`).
The reference is not a Swift port but dgrauet's Python [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)
(MIT), whose pre-quantized packs (`dgrauet/ltx-2.3-mlx-q4` and `-q8`, one safetensors file per
component) the catalog now downloads, with Gemma 3 12B from `mlx-community/gemma-3-12b-it-4bit`
as a companion checkpoint shared by both. The port mirrors its module tree, so the packs load
unchanged, and runs its `DistilledPipeline`: Gemma's 49 states through the text connector, eight
steps at half size, the ×2 latent upsampler, three steps at full size, the video decoder (tiled
over frames and pixels when the clip would not fit half the memory), the audio decoder and the
BigVGAN vocoder with bandwidth extension, then an MP4 written with `AVAssetWriter` (H.264 and
48 kHz AAC) and a poster. A reference image pins the first frame in both stages. The fixture is a
tiny run of the reference's own pipeline on the real pack, recorded by hooks (Engine/README.md):
every transformer pass matches within 1e-4 given the reference's inputs, the connector is
bit-identical, the encoder and decoder match in float32; at 768 × 512 × 49 frames the whole clip
through the app's protocol is the same as the reference's (PSNR 32.6 dB on average through H.264),
in the same 97 s, peaking at 15 GB with *Save memory* (a 5-second clip: 231 s, 17 GB). Two
known differences: the reference first
compresses a reference image with H.264 (CRF 33) to match its training clips, which the port does
not (VideoToolbox's round trip shifts the colors); and mlx-swift's conv3d, older than the
reference's, rounds the VAE's bfloat16 convolutions differently (about 40 dB between the two).

## The question

Turbo MLX is a Swift app, but the images are generated by a Python child process (mflux on MLX)
that the app installs on first launch. Could the app use [MLX Swift](https://opensource.apple.com/projects/mlx/)
directly? Speed and efficiency are the point of the app, the models are a curated set, and the
app must be ready for video models such as LTX.

## Short answer

Yes, and it is the right direction for this app, but not for the reason one might expect.

- **Python is not what makes generation slow.** MLX Swift and Python MLX are two thin bindings
  over the same C++ core and the same Metal kernels. The denoising loop is GPU-bound; the
  language around it costs well under 1% of a step (see below). A native engine does not turn
  Z-Image's 100 s into 50 s. Speed comes from the model, the quantization, the step count and
  the MLX version, and those are the same in both languages.
- **What a native engine removes is everything around the loop:** the 1 GB engine install on
  first launch, uv, the private Python and the venv, the worker start-up, the login-shell
  environment probe, the PNG round trip, and Python's garbage collector standing between "free
  the text encoder" and the memory actually being freed. The app becomes one signed bundle that
  works the moment it is opened, and becomes eligible for the Mac App Store (the current design
  downloads and executes code, which App Store guideline 2.5.2 forbids).
- **For video it matters more than for images.** A clip is hundreds of frames plus, with LTX-2,
  an audio track: they should go from MLX arrays to an `AVAssetWriter` in the same process, not
  through files or pipes, and memory has to be steered closely enough (tiled and chunked
  decoding, caches dropped on pressure) that a process the app controls is the only comfortable
  option.
- **The cost is the models themselves.** mflux implements each family (tokenizer, text encoder,
  transformer, VAE, scheduler, weight loading); a native engine has to have each one in Swift
  and prove it produces the same images. A year ago that meant writing them. Today there is an
  MIT or Apache 2.0 Swift port of every family in the catalog and of LTX-2.3/2.5 (see "What
  exists to reuse"), so the work is selection, weight compatibility, verification and
  packaging. It is still weeks per family, not days, which is why the move has to be
  incremental: the Python engine stays as a fallback until the last built-in family runs
  natively, and the app stays shippable at every step.

## Where the time and memory go today

Measured on an M1 Max with mflux 0.20, 1024 × 1024 unless noted (README):

| Model | Steps | Whole image | ≈ per step |
|---|---|---|---|
| FLUX.2 Klein 4B, 4-bit | 4 | ~30 s | ~6 s |
| Z-Image Turbo, 4-bit | 9 | ~100 s | ~10 s |
| Ming-Image 0.1, 8-bit (1024 × 576) | 12 | 76 s | ~5.5 s |

A step is one transformer pass (two with real CFG) over a few thousand latent tokens: a chain
of large matmuls and attentions that MLX runs as Metal kernels. What Python does per step is
build the lazy graph for the next pass and call `eval`: on the order of a thousand small calls,
roughly 10 ms. The worker also forces `mx.eval(latents)` at every step so the reported progress
is truthful, which gives up the CPU/GPU overlap MLX would otherwise provide, another few
milliseconds. Total: well under 0.5% of a 5–10 s step. The same holds for Swift, whose calls go
through the same C API. (The per-step `eval` is not waste, by the way: the video ports below
found that Metal's watchdog kills command buffers that run too long, so evaluating on a cadence
is required anyway.)

What is actually slow or heavy, and what would change:

| Where | Today | Native |
|---|---|---|
| First launch | Download ~1 GB (Python, MLX, mflux), a few minutes, before anything else; a model can only be downloaded once the engine exists | Nothing to install: the model download starts right away |
| Worker start | Python start + `import mflux` (a few seconds) on every launch, restart and force stop | Instant |
| Model load | Reading safetensors from disk, I/O bound | Same files, same time |
| Text encoder | One forward pass of a 4–16 B model; with *Save memory*, the whole model is unloaded and reloaded so the encoder and the transformer are never co-resident (`turbo_worker.py`, `_generate`) | Same pass; the encoder can be dropped deterministically (ARC) without `gc.collect()` + `clear_cache()` and without reloading the transformer |
| Denoising | GPU | Same kernels, same time |
| VAE decode | GPU; tiled in low-memory mode | Same; tiles can report progress and be cancelled between tiles |
| Saving and display | PIL encodes a PNG, the app decodes it again | The decoded array becomes a `CGImage` directly; the PNG is written in the background |
| Cancel | At the next denoising step; a second press kills the Python process and the loaded model | Between any two evaluations, tiles included; nothing is lost |
| Live preview | None: a preview per step would mean a PNG through the pipe | Cheap: decode a small preview from the latents every few steps in-process (the Swift FLUX.2 port already exposes a per-step image callback) |
| Memory | Python process (a few hundred MB) plus MLX; freeing depends on Python's GC | MLX only; deterministic freeing; can react to macOS memory-pressure notifications |
| Failures | A crash kills the worker; the app survives and shows the log | Same, provided the engine keeps its own process (see "The architecture"); in-process, a Metal out-of-memory is fatal |
| Distribution | Hardened runtime, no sandbox, code downloaded at run time: not App Store eligible | Sandbox possible, App Store possible |
| Dependencies | `turbo_worker.py` imports private mflux modules (`ConfigResolution.resolve_restricted`, `QwenPromptEncoder`…) and is pinned to one commit | The pipeline is ours; new models are Swift work, not a Python adapter |

## What MLX Swift offers today

Verified on `Package.swift`, the `Source/` tree, tags and release notes of
[ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift).

- **Version and platforms.** 0.31.6 (2 July 2026), releases roughly monthly. macOS 14+, so the
  app's macOS 15 target is fine. `swift-tools-version` 6.3: a current Xcode is needed. The
  Metal shaders are compiled by a build step that only works from Xcode or `xcodebuild`, not
  `swift build`; for an Xcode project this changes nothing.
- **Everything the image and video models need is there.** `conv2d` and `conv3d` (plus their
  transposed forms and `Conv3d` / `ConvTransposed3d` modules) for 2D and causal 3D VAEs;
  fused scaled-dot-product attention in the main module, with grouped-query shapes and array
  or causal masks; RMSNorm, LayerNorm, GELU/SiLU, embeddings; `compile` for fusing the small
  ops of a block; FFT and `complex64` for an audio vocoder; custom Metal kernels through
  `MLXFast.metalKernel` if a hot spot ever needs one.
- **Quantization is the same as Python's.** `QuantizedLinear`,
  `quantize(model:groupSize:bits:mode:)` and the `quantized` / `dequantized` / `quantizedMM`
  ops, with 2, 3, 4, 5, 6 and 8 bits and group sizes 32/64/128 in affine mode, plus `mxfp4`,
  `mxfp8` and `nvfp4`. An affine 4-bit weight quantized by mflux is the tensor triple
  (`weight`, `scales`, `biases`) that `QuantizedLinear` consumes; only the parameter names
  differ.
- **Files.** `loadArrays(url:)` and `loadArraysAndMetadata(url:)` read safetensors and their
  metadata header: the same files the app downloads today.
- **Memory control** is finer than what the worker reaches through mflux: `Memory.cacheLimit`,
  `Memory.memoryLimit`, `Memory.snapshot()` (active, cache, peak), `Memory.clearCache()`,
  wired limits, `GPU.maxRecommendedWorkingSetBytes()`. `peakMemory` in the history keeps
  working.
- **Errors.** Since 0.21.3, `withError { }` and `setErrorHandler` turn most MLX failures into
  Swift errors; a Metal allocation failure still brings the process down in practice (the same
  is true of Python: see the mlx-lm issues on Metal OOM). MLX 0.32.2 changed the Metal
  completion handler so that a GPU reset is rethrown instead of aborting; the Zephra app pins an
  unreleased mlx-swift revision to get it, catches Metal faults process-wide and still relaunches
  on a lost device. A separate engine process is the simpler guarantee; see "The architecture".
- **One lag to know about.** MLX Swift vendors a copy of the MLX core and updates it a little
  behind Python. MLX 0.32.2 (August 2026) made small-depth 3D convolutions 3× faster
  ([PR 3785](https://github.com/ml-explore/mlx/pull/3785), after
  [issue 3625](https://github.com/ml-explore/mlx/issues/3625)); MLX Swift 0.31.6 predates it, so
  the Swift video ports decompose 3D convolutions into per-frame 2D ones themselves. Image
  models are unaffected.

## What exists to reuse

The Swift side of the MLX ecosystem changed in the last twelve months. Beyond Apple's own
libraries there is now at least one open-source Swift port of every family in the catalog and
of LTX-2 (verified on GitHub; dates are last commits):

| What | Where | Notes |
|---|---|---|
| Qwen3, Gemma 3, Mistral 3, Qwen2.5-VL, Qwen3-VL and many more LLMs/VLMs | [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) 3.31.4 (Apple, MIT) | The text encoders of Z-Image, FLUX.2 Klein and Qwen-Image are models this library already runs; a diffusion pipeline needs their hidden states rather than logits, a small change. No T5 (needed only by the old LTX-Video 0.9 and FLUX.1) and no Ling MoE (Ming-Image). |
| Tokenizers | [swift-transformers](https://github.com/huggingface/swift-transformers) 1.3.4 (Apache 2.0) | Loads `tokenizer.json` for Qwen2/3 (BPE), Gemma, T5 (Unigram). Caveat: SentencePiece's precompiled normalization is approximated. |
| SD 2.1, SDXL Turbo | mlx-swift-examples `StableDiffusion` (Apple, MIT) | Apple's reference for a diffusion pipeline in Swift; not a model the app wants. |
| FLUX.2 Klein 4B/9B, FLUX.2 dev | [VincentGourbin/flux-2-swift-mlx](https://github.com/VincentGourbin/flux-2-swift-mlx) (MIT, v2.1.0, 12 Sep 2026, macOS 15, pins mlx-swift 0.31.6) · [mzbac/flux2.swift](https://github.com/mzbac/flux2.swift) (Apache 2.0, Feb 2026, 5 commits) | The first is a real library (`Flux2Pipeline`: load with progress, per-step progress and preview callbacks, LoRA) behind an App Store product; it loads the original diffusers weights and quantizes at start-up (int4/8-bit, exportable), not mflux checkpoints; no cancellation API. Klein 4B transformer: 7.4 GB bf16, 2.1 GB int4; ~26–30 s at 1024² on an M2 Ultra. |
| Z-Image Turbo | [nanguoyu/z-image-swift-mlx](https://github.com/nanguoyu/z-image-swift-mlx) + [swift-diffusion-core](https://github.com/nanguoyu/swift-diffusion-core) (Apache 2.0, 23 Sep 2026) · [mzbac/zimage.swift](https://github.com/mzbac/zimage.swift) (MIT, Dec 2025) | The first reads mflux's numbered-shard 4-bit format directly (its own Qwen3 4B encoder; no cancellation); the core is still a scaffold, pinned to `main`. The second has progress stages and per-step cancellation (`Task.checkCancellation`), loads diffusers weights or its own 8-bit copy; M2 Ultra at 1024²: bf16 ~21 GB / 46 s, 8-bit ~7.5 GB / 44 s; its pins are stale. |
| Qwen-Image family | [xocialize/qwen-image-edit-swift](https://github.com/xocialize/qwen-image-edit-swift) (MIT, macOS 26; Edit 2511 and NVIDIA's Qwen-Image-Flash) · `qwen-image21-swift` (Qwen-Image 2.1, a different 7 B architecture with research-only weights) · [mzbac/qwen.image.swift](https://github.com/mzbac/qwen.image.swift) (GPLv3: not usable here) | The Edit port's text-to-image path mirrors diffusers' `QwenImagePipeline`, and its weight keys are the same for Qwen-Image, Edit 2511 and Flash, so it should load the 2512 checkpoint, but nothing ships or tests that: to confirm first. Qwen2.5-VL from the same author's `qwen25vl-mlx-swift`. Flash 8-bit: 22 GB resident, 30 GB peak, 20 s for 4 steps at 1024². |
| Ming-Image 0.1 Design / Layer | [xocialize/ming-image-swift](https://github.com/xocialize/ming-image-swift) (MIT, v0.2.0, 27 Sep 2026, macOS 26) | Its own Ling MoE encoder and Qwen2 connector; loads its own conversions (`mlx-community/Ming-Image-0.1-Design-{bf16,8bit,4bit}`, group 64), not the te5 checkpoint in the catalog. M5 Max peaks: bf16 ~50 GB, 8-bit ~28 GB, 4-bit ~20 GB (no encoder release, apparently: mflux with *Save memory* peaks at 14.7 GB); ~40 s at 1024², 12 steps. Part of `mlx-engine-swift` (one package per model, a memory governor, cooperative cancellation). |
| LTX-2.3 / LTX-2.5, video with audio | [VincentGourbin/ltx-video-swift-mlx](https://github.com/VincentGourbin/ltx-video-swift-mlx) (MIT, v0.4.2, 10 Sep 2026) · [xocialize/ltx-2-mlx-swift](https://github.com/xocialize/ltx-2-mlx-swift) (Apache 2.0, 19 Sep 2026, needs the macOS 27 SDK) | Text- and image-to-video with audio, retakes, lip sync; bf16 to 4-bit; 3D-conv VAE with temporal tiling and a vocoder in Swift; `actor LTXPipeline` with progress, no cancellation; tracks `mlx-swift-lm` `main`. The second is parity-gated against the Python port (cosine ≥ 0.999). |
| Wan 2.2 TI2V-5B, VACE | xocialize `ti2v-5b-mlx-swift`, `vace-mlx-swift` (Apache 2.0) | The smaller video option; listed by the same author, not examined in depth. |
| mflux-format loaders | nanguoyu's Z-Image port; mere-run's loader (Klein and Z-Image mflux conversions); [osaurus-ai/vmlx-swift](https://github.com/osaurus-ai/vmlx-swift) (`MFluxQuant.swift` for its FLUX.1, FLUX.2 and Qwen ports) | Three independent Swift codebases already read the `weight` / `scales` / `biases` layout mflux writes: the key-map approach below is proven. |
| Whole apps | [sawfwair/mere-run](https://github.com/sawfwair/mere-run) (MIT, macOS 15: Klein, Z-Image, Qwen Image Edit, LTX 2.3/2.5, Wan 2.2) · [jamesbrink/Zephra](https://github.com/jamesbrink/Zephra) (MIT, macOS 15) | Proof that a native Swift app with this catalog, video included, ships today. mere-run embeds its own ports and runs inference in a child process behind an admission queue; Zephra vendors the ports above as kits, runs in-process on a pinned mlx-swift `main`, and relaunches on a lost GPU. |

Two things follow. First, the cost estimate changes: the families do not have to be written
from scratch; the work is choosing a port per family, making it load the checkpoints the app
already ships (or changing the catalog to the checkpoints the port expects), proving parity
with mflux, and packaging. Second, most of these projects are months old, some with a handful
of commits; they are references and starting points, not dependencies to take on blindly. Take
their model code (a few files per family) into the engine target behind our own small
interface, not their orchestration frameworks, which change fast. Licenses matter: MIT and
Apache 2.0 ports can be vendored into a proprietary app, GPLv3 ones (mzbac's FLUX.1 and
Qwen-Image ports) cannot.

mflux itself (MIT) stays the reference implementation: its pipelines are the oracle the Swift
engine is verified against, and its code is the map when a port and the paper disagree.

## The catalog, family by family

What each built-in family is made of (from the model repositories and mflux's weight
definitions), which is what a Swift engine has to load and run:

| Family | Transformer | Text encoder | VAE | Swift reference |
|---|---|---|---|---|
| FLUX.2 Klein 4B ([black-forest-labs/flux2](https://github.com/black-forest-labs/flux2)) | FLUX-style MMDiT, 5 double-stream + 20 single-stream blocks, ~4 B | Qwen3 (by hidden size, the 4 B model for Klein 4B and the 8 B one for Klein 9B) | FLUX.2 autoencoder, 32 latent channels (128 after the 2 × 2 pixel shuffle) | flux-2-swift-mlx (model code), vmlx-swift and mere-run (mflux weights) |
| Z-Image Turbo ([Tongyi-MAI/Z-Image](https://github.com/Tongyi-MAI/Z-Image)) | Single-stream "S3-DiT", 6 B | Qwen3 4B | FLUX.1-style 16-channel 2D VAE | z-image-swift-mlx (mflux weights), zimage.swift (cancellation) |
| Qwen-Image 2512 ([QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image)) | Dual-stream MMDiT, 20 B | Qwen2.5-VL 7B (text path) | Wan 2.1-derived 16-channel causal 3D VAE | qwen-image-edit-swift (same weight keys), vmlx-swift |
| Ming-Image 0.1 Design ([mflux PR 765](https://github.com/mflux-community/mflux/pull/765)) | Z-Image-style single-stream DiT, 30 layers, 6.15 B | Ling-mini-2.0, a ~16 B mixture-of-experts LLM (1.4 B active), plus a Qwen2-1.5B connector producing 256 query tokens | Qwen-Image VAE retrained for RGBA | ming-image-swift |

Order of difficulty, lowest first: Klein (small, dense, closest reference), Z-Image (shares
the Qwen3 encoder with Klein), Qwen-Image (20 B and a VLM encoder, but a well-trodden
architecture; its 3D VAE is the first piece of video machinery), Ming-Image (a MoE text
encoder that no Apple library implements, a connector stage, RGBA decoding). Note that
Ming-Image is the app's default model on Macs with 24 GB or more: until it is native, most
users still see the engine install on first launch, unless the default changes.

Klein 9B, which "Add Model…" can add, is licensed for non-commercial use only; Klein 4B, the
built-in one, is Apache 2.0.

## The same model files

The catalog's checkpoints are in mflux's format, and the app should keep using them: they are
pre-quantized (Klein 4B is a 4.6 GB download; the original bf16 transformer plus Qwen3 4B would
be more than three times that), users already have them on disk, and mflux, other tools and this app
share one cache.

The format is plain and readable from Swift (mflux
[`model_saver.py`](https://github.com/mflux-community/mflux/blob/main/src/mflux/models/common/weights/saving/model_saver.py)
and `weight_applier.py`): one folder per component (`transformer/`, `text_encoder/`, `vae/`…),
safetensors shards of at most 2 GB named `0.safetensors`, `1.safetensors`… with a
`model.safetensors.index.json` (`ModelLocator` already reads it), and the metadata
`quantization_level` (bits) and `mflux_version` in every shard. The tensors are
`tree_flatten(model.parameters())` of mflux's own module tree, quantized with `nn.quantize`:
each linear layer becomes `weight` (packed uint32), `scales` and `biases` in MLX's affine
scheme, group size 64 by default. The group size is not recorded; mflux infers bits and group
size from the tensor shapes and tolerates layers left in bf16, and the Swift loader does the
same. In MLX Swift terms: `loadArraysAndMetadata(url:)` per shard, a per-family map from mflux
parameter paths to the Swift module's paths (mechanical: the module trees mirror each other
when the port follows mflux's structure), `quantize(model:groupSize:bits:)` with the same
predicate, then `update(parameters:)`. No conversion step, no second download. Three Swift
codebases already do exactly this (nanguoyu's Z-Image port, mere-run, vmlx-swift's
`MFluxQuant.swift`), so it is a known quantity, not a research item.

The alternative most Swift ports chose, the original diffusers layout quantized on first load
and cached (Zephra, flux-2-swift-mlx's export), doubles the first download and adds a
quantization pass to the first run; it can be offered later through "Add Model…" if wanted.
Nothing in the plan needs it.

## The architecture

### Keep the worker, change its language

The app already has the right shape: a UI process and an engine process that speak a small,
explicit protocol (`WorkerCommand` / `WorkerEvent`: `generate`, `cancel`, `unload`, `shutdown`;
`ready`, `phase`, `progress`, `done`, `failed`, `model_loaded`). What is wrong with it is not the
separation but what is on the other side of the pipe.

The proposal is to replace `turbo_worker.py` with a Swift executable, **`turbo-engine`**, built
by the same Xcode project (a second target, embedded in `Contents/MacOS` exactly as `uv` is
today, signed with the app) and linking MLX Swift through Swift Package Manager. It speaks the
same JSON lines, so `BackendController`, `LineProcess`, `AppModel.handle(_:)` and every view
keep working unchanged, and the two engines can coexist: the app routes a job to the native
engine when its family is supported natively and to the Python worker otherwise. The Python
installer, `uv` and `setup_backend.sh` are deleted the day the last built-in family is native.

Why a separate process rather than `import MLX` in the app:

- MLX signals errors with C++ exceptions. MLX Swift turns most of them into Swift errors, but
  a Metal allocation failure (`[metal::malloc] Resource limit exceeded`) still ends the process
  in practice. An app whose purpose is running 6–22 B models near the memory ceiling will hit
  it (a 2048 × 2048 image on a 16 GB Mac). Today the worker dies and the app shows "The image
  engine stopped" with the log; that resilience is worth keeping.
- The UI process stays small and the engine's memory is accounted separately, so macOS can
  reclaim the engine under pressure without touching the window, the queue or the history.
- Force stop keeps working exactly as now: kill and restart.

An XPC service would be the more idiomatic packaging of the same idea (typed protocol,
launchd-managed lifetime). It is a refinement for later: the JSON protocol is a dozen message
kinds, already handled robustly, and keeping it is what makes the transition gradual. mere-run,
the most complete of the native apps, made the same choice: its macOS app launches its CLI as a
child process and puts an admission queue with memory checks in front of it.

One thing the ports mostly lack is cancellation (only zimage.swift checks for it per step; the
FLUX.2 and LTX pipelines have progress callbacks but no way to stop). The app relies on `cancel`
between steps today and will rely on it between tiles and chunks for video, so the engine's
family interface has to thread a cancellation check through every loop it borrows.

Inside `turbo-engine`, one `ImageFamily` (later `MediaFamily`) type per family, the Swift twin
of the `FAMILIES` adapters in `turbo_worker.py`: load, encode prompt, denoise with a step
callback, decode, save. Text encoders come from mlx-swift-lm where it has them (Qwen3,
Qwen2.5-VL), exposing hidden states. The engine runs everything off the main actor
(the project sets `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`; the engine target should not).

### What changes in the app

- **`BackendController`** grows a second engine (`turbo-engine` next to the Python worker),
  chooses per job, and loses the install/update logic once Python is gone. `Status.notInstalled`
  and `.installing` disappear with it, as do "Install Image Engine" and the setup view.
- **`DownloadCenter`** stops depending on the engine: a `URLSession` downloader that talks to
  the Hugging Face API (`/api/models/{repo}/tree/main?recursive=true` for sizes, `/resolve/`
  with a `Range` header to resume) and writes the hub cache layout `ModelLocator` already reads
  (`models--org--name/blobs`, `snapshots/<commit>/…`, `refs/main`). Existing downloads stay
  visible, mflux and other tools still share them, and a model can be downloaded before, or
  without, any engine. swift-transformers' `HubApi` does the downloading well but uses its own
  flat layout, so either write the ~300 lines or teach `ModelLocator` a second layout. This
  piece has no MLX dependency and can land first.
- **`ShellEnvironment`** exists to give Python the user's `PATH`, `HF_HOME` and `HF_TOKEN`. The
  native engine needs only the last two: read them from the environment and from Settings, and
  drop the login-shell probe when Python goes.
- **`ModelLocator`, `HistoryStore`, `AppModel`, the views:** unchanged by the engine swap. They
  change for video (below), which is independent of the engine's language.

### Proving it

A native family is done when, for the catalog's checkpoints and a fixed set of prompts, seeds
and sizes, `turbo-engine` produces the same image as mflux (PSNR above a threshold, not
bit-identity: kernel scheduling differs; xocialize gates its LTX port on cosine similarity
≥ 0.999 against the Python port, a good precedent) in the same or less time and memory.
`turbo-engine verify` runs that comparison against a folder of mflux reference images on the
developer's Mac (there is no GPU in CI). Keep mflux as the oracle during development: dumping
intermediate tensors from both sides is how a wrong reshape or an off-by-one in rotary
embeddings gets found.

## Ready for video

Most of what a video model needs from the app has nothing to do with LTX or with MLX Swift and
should be in place before the first video family, ideally before the first native image
family, so nothing is designed image-only twice.

### Domain model

- **`GenerationRequest`** gains the video fields as optionals that are `nil` for images:
  `frames` (8k + 1 for LTX), `fps`, and later an input image for image-to-video. The protocol
  gains the same parameters.
- **A result is media, not an image.** `HistoryItem.fileName` becomes a file plus a kind:
  `.image` (PNG, as now) or `.video` (MP4/MOV with a poster PNG for thumbnails, Quick Look and
  the strip). `GenerationJob.Phase` gains `.encodingVideo`, and progress carries a unit (steps,
  tiles, frames) so the VAE decode of 121 frames is not a spinner.
- **`ModelFamily`** gains a media kind and the video geometry rules (sides multiple of 32,
  frame count 8k + 1, fps range), the way it already carries steps, guidance and components.
- **Settings:** a duration and frame-rate control that appears for video families; disk-space
  checks that know a clip is not 2 MB; memory tiers per resolution and length, since the
  same model fits or does not depending on both.

### Engine and output

- The engine writes the clip with `AVAssetWriter` (HEVC or H.264, from `CVPixelBuffer`s filled
  straight from the decoded arrays) and a poster frame; LTX-2's audio (a separate branch of the
  same transformer, decoded by an audio VAE and a vocoder) becomes the file's audio track through
  a second writer input. Nothing goes through PNG-per-frame or a pipe. This is the strongest
  argument for the engine being Swift: on the Python side it means ffmpeg or imageio, another
  dependency to ship.
- Video VAEs decode in temporal chunks and spatial tiles; each chunk is a natural progress and
  cancellation point, and an `eval` point for Metal's watchdog.

### Viewer

- `OutputPanel` shows a `VideoPlayer` (AVKit) for `.video` items where it shows `ZoomableImage`
  today; the strip shows the poster with a duration badge; Copy copies the file; Share and Show
  in Finder work as they are.

### LTX, as it stands in September 2026

From [Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video) and
[Lightricks/LTX-2](https://github.com/Lightricks/LTX-2):

| Model | Released | Transformer | Text encoder | Output | Weights (bf16) |
|---|---|---|---|---|---|
| LTX-Video 0.9.8 | July 2025 | 2 B distilled, 13 B dev/distilled | T5-XXL | video | 2 B: ~4 GB + T5 ~9.5 GB; 13 B: ~26 GB + T5 |
| LTX-2 | January 2026 | 19 B (14 B video + 5 B audio), 48 blocks | Gemma 3 12B (~25 GB) | video + audio, up to 4K/50 fps | ~43 GB (27 GB fp8) |
| LTX-2.3 | March 2026 | 22 B, dev and distilled 1.1 | Gemma 3 12B | video + audio | ~30 GB packaged |
| LTX-2.5 | August 2026 | 22 B, NVFP4 available | fine-tuned Gemma 4 12B | video + audio, 4K | ~66–70 GB |

All of them use a causal 3D VAE with 32 × 32 spatial and 8 × temporal compression and 128
latent channels (frames must be 8k + 1, up to 481 in the Swift port, i.e. 20 s at 24 fps; sides
multiples of 32, 64 in the Swift port; 768 × 512, 1024 × 576 and 832 × 480 are the recommended
sizes). For LTX-2.3 the Swift port runs Gemma 3 12B from a 4-bit QAT checkpoint of about
7.5 GB; LTX-2.5's fine-tuned Gemma 4 is a 26 GB bf16 download that is not quantized in place,
which is why 2.5 needs 38 GB even with a 4-bit transformer. The LTX-2 family is under the
LTX-2 / LTX-2.x Community License: free, including commercially, below US$10 M yearly revenue,
with use restrictions that also bind the generated outputs, so the app must show the license as
it does for image models.

Measured on Apple silicon by the Swift ports (their READMEs):

| Port | Mac | Setting | Time | Peak memory |
|---|---|---|---|---|
| xocialize/ltx-2-mlx-swift, LTX-2.3 4-bit | M5 Max | 512 × 288, 121 frames | ~64 s | 15.4 GB |
| same, 8-bit | M5 Max | 704 × 512, 161 frames | ~188 s | 37.5 GB |
| same, bf16 | M5 Max | 704 × 512, 481 frames | ~960 s | 72.7 GB |
| VincentGourbin/ltx-video-swift-mlx, LTX-2.3 4-bit | M3 Max 96 GB | 1024 × 576, 241 frames, image-to-video with audio | 1294 s | 38.4 GB |
| same, 8-bit | M3 Max 96 GB | same | 1458 s | 44.6 GB |
| dgrauet/ltx-2-mlx (Python), LTX-2.5 distilled 8-bit | M5 Pro 64 GB | 704 × 448, 49 frames | ~40 s | 20.9 GB |

What this means for the catalog: LTX-2.3 distilled at 4-bit is the first video family, listed
under 32 GB for short, small clips and 64 GB for 704 × 512 and longer; LTX-2.5 is a 64 GB+
model. Wan 2.2 TI2V-5B (Apache 2.0; xocialize lists a Swift port, not examined here) is the candidate
for 24 GB Macs. The
only LTX that would fit 16 GB is the 2 B distilled LTX-Video 0.9.8, which has no Swift port and
needs a T5-XXL encoder that no Apple library implements; it is not worth it unless 16 GB video
becomes a requirement. Same pattern as today's images: the memory tier in the model's name,
*Save memory* dropping the Gemma encoder after the prompt is read, tiles and chunks below.

Two constraints from the ports. VincentGourbin's README asks for macOS 26.3 while its
`Package.swift` declares macOS 15, and no reason is documented: to be settled by building it on
macOS 15; xocialize's needs the macOS 27 SDK outright. Both track `mlx-swift-lm` `main` rather
than a release. The app targets macOS 15 today; the video phase may raise that, or ship video
only on newer systems.

## Plan

Each phase leaves the app shippable.

**Phase 0 — Prepare (no visible change).** Media kinds in the domain model and the history
(`fileName` + kind, poster), progress units, engine routing in `BackendController`, the native
downloader (models downloadable without an engine). No MLX yet; this is the groundwork for
video and for the native engine alike.

**Phase 1 — First native family.** Add `mlx-swift`, `mlx-swift-lm` (Qwen3) and
`swift-transformers` (tokenizers) to a new `turbo-engine` target. Implement FLUX.2 Klein 4B
first: the smallest transformer in the catalog, 4 steps, Apache 2.0, the closest Swift
reference (flux-2-swift-mlx, MIT, macOS 15, actively maintained), loading the app's mflux
checkpoint. Route Klein to the native engine, everything else to Python. Ship when `verify`
passes and Klein users see no install step.

**Phase 2 — The rest of the catalog.** Z-Image Turbo (shares the Qwen3 encoder; two Swift
references), then Qwen-Image 2512 (Qwen2.5-VL encoder from mlx-swift-lm, the Edit 2511 port
for the transformer, the Wan-derived 3D VAE: the first video building block, tested on still
images), then Ming-Image (its MoE encoder and connector from ming-image-swift). Each family
switches to native as it passes `verify`. If video is the priority, LTX can be pulled in
between Z-Image and Qwen-Image: after Phase 1 the engine, the protocol and the verification
harness exist, and the heavy image families can stay on Python meanwhile.

**Phase 3 — Remove Python.** Delete `uv`, `setup_backend.sh`, `turbo_worker.py`, the install
UI, `ShellEnvironment`'s shell probe. Turn on the App Sandbox (the shared hub cache through a
security-scoped bookmark, or the app container: user's choice). Decide about the Mac App Store.

**Phase 4 — Video.** LTX-2.3 distilled in `turbo-engine` (from the two Swift ports, gated on
parity with the Python port), the viewer and export work from "Ready for video", memory tiers
by resolution and length in the catalog, Wan 2.2 TI2V-5B for 24 GB Macs after it.

## Risks and open points

- **Numerical parity is the real work.** A DiT port that runs but drifts by a wrong epsilon,
  timestep shift or attention scale produces plausible, wrong images. `verify` against mflux
  and tensor dumps are not optional.
- **The ports are young.** Handfuls of commits, one-person projects, tracking unreleased
  branches of mlx-swift-lm. Vendor their model code behind our interface; do not take their
  frameworks as dependencies.
- **Ming-Image is the default model and the hardest port.** Until it is native, the install
  step remains for most users; the alternative is a different default for a while.
- **MLX Swift is pre-1.0 and lags the core by a few weeks.** Pin the version; the engine's
  surface (matmul, attention, conv, quantized linear, safetensors) is the stable part. The 3D
  convolution speed-up arrives with the next core bump; until then the video VAE decodes per
  frame, as the Python ports do.
- **Metal out-of-memory is fatal**, in either language: hence the helper process, and memory
  tiers that are honest.
- **Metal's watchdog** kills long command buffers: evaluate per step, per tile, per chunk, as
  the worker already does per step.
- **Model additions become Swift work.** Today a new mflux family is a Python adapter; natively
  it is a port. This matches a curated catalog; "Add Model…" keeps accepting checkpoints of
  supported families only.
- **Video may raise the deployment target** (macOS 26 by the ports' READMEs, unconfirmed) and
  does raise the memory floor (32 GB for LTX-2.3). Both should be stated in the catalog, as the
  image tiers are.
- **App size:** MLX Swift's compiled core and Metal library add tens of MB; `uv` (37 MB)
  leaves. Roughly neutral.
