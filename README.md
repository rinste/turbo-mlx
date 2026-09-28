# Turbo MLX

A macOS app (SwiftUI) that generates images, and videos with sound, on the Mac's GPU with MLX.
It is built to be distributed: the people who install it never open Terminal and need neither Python nor Xcode. The
engine is part of the app, written in Swift on MLX; on first launch the user picks a model, the
app downloads it, and that is all the setup there is.

![Turbo MLX: settings on the left, the image with its zoom controls and metadata on the right, the history below](docs/screenshot.webp)

- **Left column:** the prompt as blocks (Subject and Style to start with, each one a piece of the
  final text; renamable, resizable, and put in order by dragging), aspect ratio, steps and
  guidance; under *Advanced*, the resolution or a custom size, seed, number of images, transparent
  or white background, memory saving. For a video model, a *Reference Image* above the prompt
  (dropped from Finder or from the history, chosen from a file or among the generated images: the
  clip's first frame) and the clip's duration and frame rate. At the bottom, on a darker tray, the
  model (image and video models in two groups, with download and status) above the main button,
  which does what the current state needs: *Download and Generate* → *Generate*, with how long it
  should take (judged from this Mac's earlier generations with the model, e.g. *Generate (~4 min ·
  ⌘↩)*).
- **Right column:** the image (on a checkerboard when transparent) or the clip, with its prompt (a
  long one scrolls) and metadata, the running generation with its model, per-step progress and time left,
  and the history with the queue. Zoom: pinch, the mouse wheel, double-click, ⌘-scroll, the − % +
  controls or the View menu (⌘+, ⌘-, ⌘0 actual size, ⌘9 fit); two-finger scroll (⌥-wheel with a
  mouse) or drag to move around. Quick Look (space), drag and drop, copy. *Reuse Prompt and Settings* (the
  button under the image, or ⌘R) puts the image's prompt, in its blocks, its model and its
  settings back in the left column.

## Models

| Model | Memory | Download | Peak at 1024 px | Notes |
|---|---|---|---|---|
| [Ming-Image 0.1 Design](https://huggingface.co/joeynyc/Ming-Image-0.1-Design-mflux-q8-te5) | 24 GB | 19.3 GB | 14.7 GB | graphic design and typography, PNG with alpha · 12 steps · MIT |
| [Z-Image Turbo](https://huggingface.co/mflux-community/z-image-turbo-mflux-q4) | 16 GB | 5.9 GB | 8.5 GB | photorealism, text in the image · 4-bit · 9 steps · Apache 2.0 |
| [Z-Image Turbo](https://huggingface.co/mflux-community/z-image-turbo-mflux-q8) | 24 GB | 11 GB | ~13.6 GB | the 8-bit version, closer to the original |
| [FLUX.2 Klein 4B](https://huggingface.co/mflux-community/flux2-klein-4b-mflux-q4) | 24 GB | 4.6 GB | 14.1 GB | the fastest: 4 steps · Apache 2.0 |
| [Qwen-Image 2512](https://huggingface.co/mflux-community/qwen-image-2512-mflux-q4) | 32 GB | 27.6 GB | 21.3 GB | 20B, rich scenes and long text · 4-bit · 20 steps · Apache 2.0 |
| [LTX-2.3](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) | 32 GB | 28.5 GB | 17 GB (768 × 512, 5 s) | video with sound, from a prompt or from an image as the first frame · 22B distilled, 4-bit · 8 + 3 steps · LTX-2 Community |
| [LTX-2.3](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8) | 64 GB | 37.8 GB | 38 GB (768 × 512, 5 s) | the 8-bit version, closer to the original |

Measured on an M1 Max with mflux 0.20, the Python reference the native engine is checked against
(the 8-bit Z-Image peak adds its larger weights to the measured 4-bit one). A model is listed
under the smallest common Mac memory size it peaks under 80% of. Times at 1024 × 1024: FLUX.2
Klein ~30 s, Z-Image Turbo 4-bit ~100 s; Ming-Image takes 76 s at 1024 × 576. The model stays
loaded between images. LTX-2.3 is measured with the native engine against dgrauet's
[ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), with *Save memory*: a 5-second 768 × 512 clip
with sound takes about 4 minutes (231 s), a 3-second one from an image 137 s; the 8-bit version,
without *Save memory* as on a 64 GB Mac, takes 5 minutes (294 s) for the same 5 seconds.

Ming-Image comes in one version (te5): at 1024 px its memory peak is set by the DiT, which is the
same in every conversion, and te6/te8 give the same images as te5 with more memory. Qwen-Image
3.0 has no public weights (it only runs on Alibaba's platform), so Qwen-Image 2512 is the latest
Qwen model here; Qwen-Image 2.1 is newer but licensed for non-commercial use only.

Peaks are with *Save memory* on, the default below 64 GB; without it Ming-Image peaks at ~35 GB
and Qwen-Image at ~43 GB.
More mflux checkpoints of the same families can be added from
**⋯ → Add Model…** (Hugging Face repository or local folder; for Klein you can pick the 4B/9B/base
variant).

## Development

Requirements: an Apple silicon Mac, macOS 15 or later, Xcode 26.

```bash
open TurboMLX.xcodeproj
```

Then ⌘R in Xcode. The app's build also builds the native engine from `Engine/` and embeds it
(the *Embed turbo-engine* phase, `scripts/embed-engine.sh`): the first build takes a few minutes
for MLX's C++ core and Metal kernels, the next ones seconds, since the engine is only rebuilt when
`Engine/` changed. The app cannot generate without it, so when the engine does not build, neither
does the app.

Builds are signed with the team's Developer ID certificate, local ones too: the sandbox ties the
app's container to its signature, and an ad hoc one changes with every build. Without the
certificate, override `CODE_SIGN_IDENTITY=-`, knowing macOS may then keep a new build out of the
data an older one left.

To try the first-run experience without touching your real history, point the app at an empty
data folder in its container (`~` is the container's home there, hence the quotes):

```bash
open -n --env TURBO_MLX_HOME='~/first-run' "path/to/Turbo MLX.app"
```

## How it works

```
SwiftUI ── JSON lines (stdin/stdout) ──▶ turbo-engine serve ──▶ MLX Swift (GPU)
```

- **Engine.** `Engine/` is a Swift package that runs all four families (FLUX.2 Klein,
  Z-Image Turbo, Qwen-Image 2512, Ming-Image) on MLX Swift from mflux's checkpoints. The app's
  build embeds its `turbo-engine` binary in the bundle (`Contents/MacOS`, signed with the app),
  and every model runs there: nothing is installed on first launch, the engine starts in an
  instant, and no Python process sits between the app and the GPU. The engine keeps the
  selected model in memory, loads it as soon as a prompt is being written, encodes the prompts
  of the queued images while the text encoder is in memory, reports phases, per-step progress
  and the seconds each phase took (shown when hovering the time of an image), and stops a
  generation at the next step. See `Engine/README.md` and
  [docs/native-engine.md](docs/native-engine.md).
- **mflux is the reference.** The ports follow [mflux](https://github.com/mflux-community/mflux)
  (Python + MLX) module for module: `Engine/Fixtures` builds the references `verify` compares
  each stage against, and `Engine/Reference/turbo_worker.py`, the Python engine the app used to
  ship, runs a real checkpoint through mflux behind the same protocol, to compare the images.
  Neither is part of the app.
- **Save memory.** Frees the text encoder of Ming-Image and Qwen-Image once the prompt is read
  (a new prompt reloads only the text side, with the transformer released first, so the two are
  never in memory together), decodes the image in tiles where the VAE allows it (not FLUX.2,
  whose tiles would show seams), and keeps MLX's buffer cache small.
- **Downloads.** The app downloads models itself (`Services/HubDownloader.swift`) into the
  Hugging Face cache, in the same layout huggingface_hub uses, so mflux and other tools share
  them; an interrupted download resumes. No engine is needed to download. A gated or private
  repository uses the token the Hugging Face CLI saved (`hf auth login`).
- **Sandbox.** The app runs in the App Sandbox (`TurboMLX.entitlements`): its history and
  settings live in its container, and outside it the app reaches only `~/.cache/huggingface`,
  the folder the Hugging Face CLI and mflux use by default, through an entitlement for that path
  (an `HF_HOME` elsewhere is not followed). The engine inherits the sandbox. A folder picked for
  a local model is kept with a security-scoped bookmark (`Services/FolderAccess.swift`), opened
  before the engine starts so it can read it too. The first sandboxed launch moved the history
  and the settings from their old places into the container
  (`Resources/container-migration.plist`). Direct distribution only: the entitlement for
  `~/.cache/huggingface` is a temporary exception, which the Mac App Store does not accept.

| What | Where |
|---|---|
| History (PNGs + `history.json`) | `~/Library/Containers/com.stefanorinaldo.TurboMLX/Data/Library/Application Support/TurboMLX/History` |
| Models | `~/.cache/huggingface/hub` |

```
TurboMLX/
  App/        entry point, AppModel (queue, engine events, selection, persistence)
  Models/     catalog and families, settings, jobs, history items
  Services/   the engine process, downloads, Hugging Face cache, history
  Views/      left column, output, history, settings, log
Engine/       the engine (Swift package: turbo-engine, TurboEngineCore), its fixtures and the
              mflux reference worker
scripts/      release.sh, embed-engine.sh, build-engine.sh, ExportOptions.plist
TurboMLX.entitlements   the app's sandbox (the engine's: Engine/turbo-engine.entitlements)
```

The project uses Xcode's synchronized folders: files added under `TurboMLX/` join the target on
their own.

**Adding a model family:** a case in `ModelFamily` (`Models/ModelCatalog.swift`: download
patterns, components, steps, guidance), a `FamilyModel` under `Engine/Sources/TurboEngineCore/Families/`
with its fixture and `verify` stage (see `Engine/README.md`), and an adapter in `FAMILIES` in
`Engine/Reference/turbo_worker.py` to compare real images with mflux.

**Where this is going:** [docs/native-engine.md](docs/native-engine.md) is the case for the
native engine, what is checked so far, and what comes next.
[docs/generation-performance.md](docs/generation-performance.md) ranks the ways to make generation
faster, with the measurements that decide each one.

## Distribution

The app is Apple silicon only and uses the hardened runtime; it must be signed with a Developer ID
and notarized.

**From Xcode:** Product → Archive → Distribute App → Direct Distribution (Xcode signs, notarizes
and staples).

**From the command line** (also builds the DMG): store the notarization credentials once (an
app-specific password, kept in the keychain), then run the script.

```bash
xcrun notarytool store-credentials turbo-mlx --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID
```

```bash
scripts/release.sh
```

The DMG ends up in `build/release/`. The script checks that the exported app carries
`turbo-engine`, signed with the Developer ID and allowed to inherit the app's sandbox: without it
the app cannot generate.

## Troubleshooting

- **Engine Log** (⌥⌘L): what the engine prints, warnings included.
- Settings (⌘,) → Engine: the engine's versions and executable, restart.
