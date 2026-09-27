# Turbo MLX

A macOS app (SwiftUI) that generates images on the Mac's GPU with MLX. It is built to be
distributed: the people who install it never open Terminal and need neither Python nor Xcode. The
engine is part of the app, written in Swift on MLX; on first launch the user picks a model, the
app downloads it, and that is all the setup there is.

![Turbo MLX: settings on the left, the image with its zoom controls and metadata on the right, the history below](docs/screenshot.webp)

- **Left column:** the prompt as blocks (Subject and Style to start with, each one a piece of the
  final text; renamable, resizable, and put in order by dragging), aspect ratio, steps and
  guidance; under *Advanced*, the resolution or a custom size, seed, number of images, transparent
  or white background, memory saving. At the bottom, the model (with download and status) above
  the main button, which does what the current state needs:
  *Download and Generate* → *Generate*.
- **Right column:** the image (on a checkerboard when transparent) with its metadata, the running
  generation with per-step progress and time left, and the history with the queue. Zoom: pinch,
  the mouse wheel, double-click, ⌘-scroll, the − % + controls or the View menu (⌘+, ⌘-, ⌘0 actual
  size, ⌘9 fit); two-finger scroll (⌥-wheel with a mouse) or drag to move around. Quick Look
  (space), drag and drop, copy, *Reuse Settings* (⌘R).

## Models

| Model | Memory | Download | Peak at 1024 px | Notes |
|---|---|---|---|---|
| [Ming-Image 0.1 Design](https://huggingface.co/joeynyc/Ming-Image-0.1-Design-mflux-q8-te5) | 24 GB | 19.3 GB | 14.7 GB | graphic design and typography, PNG with alpha · 12 steps · MIT |
| [Z-Image Turbo](https://huggingface.co/mflux-community/z-image-turbo-mflux-q4) | 16 GB | 5.9 GB | 8.5 GB | photorealism, text in the image · 4-bit · 9 steps · Apache 2.0 |
| [Z-Image Turbo](https://huggingface.co/mflux-community/z-image-turbo-mflux-q8) | 24 GB | 11 GB | ~13.6 GB | the 8-bit version, closer to the original |
| [FLUX.2 Klein 4B](https://huggingface.co/mflux-community/flux2-klein-4b-mflux-q4) | 24 GB | 4.6 GB | 14.1 GB | the fastest: 4 steps · Apache 2.0 |
| [Qwen-Image 2512](https://huggingface.co/mflux-community/qwen-image-2512-mflux-q4) | 32 GB | 27.6 GB | 21.3 GB | 20B, rich scenes and long text · 4-bit · 20 steps · Apache 2.0 |

Measured on an M1 Max with mflux 0.20, the Python reference the native engine is checked against
(the 8-bit Z-Image peak adds its larger weights to the measured 4-bit one). A model is listed
under the smallest common Mac memory size it peaks under 80% of. Times at 1024 × 1024: FLUX.2
Klein ~30 s, Z-Image Turbo 4-bit ~100 s; Ming-Image takes 76 s at 1024 × 576. The model stays
loaded between images.

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
`Engine/` changed. `TURBO_NO_ENGINE=1 xcodebuild …` builds the app without it, on the Python
engine. To try the first-run experience without touching your real installation, point the app
at an empty data folder:

```bash
open -n --env TURBO_MLX_HOME=/tmp/turbo-first-run "path/to/Turbo MLX.app"
```

## How it works

```
SwiftUI ── JSON lines (stdin/stdout) ──▶ turbo-engine serve   ──▶ MLX Swift (GPU)   every built-in family
                                    └──▶ turbo_worker.py serve ──▶ mflux / MLX (GPU)   only in a build without the engine
```

- **Native engine.** `Engine/` is a Swift package that runs all four families (FLUX.2 Klein,
  Z-Image Turbo, Qwen-Image 2512, Ming-Image) on MLX Swift from the same mflux checkpoints. The
  app's build embeds its `turbo-engine` binary in the bundle (`Contents/MacOS`, signed with the
  app), and every model runs there: nothing is installed on first launch, the engine starts in
  an instant, and no Python process sits between the app and the GPU. The engine keeps the
  selected model in memory, loads it as soon as a prompt is being written, encodes the prompts
  of the queued images while the text encoder is in memory, reports phases, per-step progress
  and the seconds each phase took (shown when hovering the time of an image), and stops a
  generation at the next step. See `Engine/README.md` and
  [docs/native-engine.md](docs/native-engine.md).
- **Python engine.** The reference the native ports are checked against, and the fallback of a
  build without the native binary: inference in [mflux](https://github.com/mflux-community/mflux)
  (Python + MLX) behind `Backend/turbo_worker.py`, which speaks the same protocol with one
  adapter per family (`FAMILIES`). `Backend/setup_backend.sh` creates the environment with
  **uv, which ships inside the app** (`Vendor/uv`, copied to `Contents/MacOS` and signed with
  the app); uv also downloads a standalone Python, so even then the Mac needs nothing
  preinstalled. mflux is installed from the GitHub source archive of the tested commit
  (`BackendController.mfluxCommit`), so git isn't needed either. With the native engine present
  the app never installs, updates or starts it.
- **Save memory.** Frees the text encoder of Ming-Image and Qwen-Image once the prompt is read
  (a new prompt reloads it, with the transformer released first, so the two are never in memory
  together; the native engine reloads only the text side, the Python one the whole model),
  decodes the image in tiles where the VAE allows it (not FLUX.2, whose tiles would show seams),
  and keeps MLX's buffer cache small.
- **Downloads.** The app downloads models itself (`Services/HubDownloader.swift`) into the
  Hugging Face cache, in the same layout huggingface_hub uses, so mflux and other tools share
  them; an interrupted download resumes. No engine is needed to download.
- The app reads the login shell's environment (PATH, `HF_HOME`, `HF_TOKEN`…).

| What | Where |
|---|---|
| History (PNGs + `history.json`) | `~/Library/Application Support/TurboMLX/History` |
| Models | `~/.cache/huggingface/hub` |
| Native engine (development builds without one embedded) | `~/Library/Application Support/TurboMLX/bin` |
| Python engine, when a build falls back to it | `~/Library/Application Support/TurboMLX/{venv,python}`, cache in `~/Library/Caches/TurboMLX/uv` |

```
TurboMLX/
  App/        entry point, AppModel (queue, worker events, selection, persistence)
  Models/     catalog and families, settings, jobs, history items
  Services/   child processes, engines, downloads, Hugging Face cache, history
  Views/      left column, output, history, settings, log
  Backend/    turbo_worker.py, setup_backend.sh (copied into the bundle)
Engine/       the native engine (Swift package: turbo-engine, TurboEngineCore, fixtures)
Vendor/uv/    the uv binary (update with scripts/update-uv.sh)
scripts/      release.sh, embed-engine.sh, build-engine.sh, ExportOptions.plist, update-uv.sh
```

The project uses Xcode's synchronized folders: files added under `TurboMLX/` join the target on
their own.

**Adding a model family:** a case in `ModelFamily` (`Models/ModelCatalog.swift`: download
patterns, components, steps, guidance), a `FamilyModel` under `Engine/Sources/TurboEngineCore/Families/`
with its fixture and `verify` stage (see `Engine/README.md`), and, for the Python fallback, an
adapter in `FAMILIES` in `turbo_worker.py`.

**Where this is going:** [docs/native-engine.md](docs/native-engine.md) is the case for the
native engine, what is checked so far, and what comes next: removing the Python fallback, and
video models (LTX).
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
`turbo-engine`, signed with the Developer ID: a release without it would install Python on
every Mac.

## Troubleshooting

- **Engine Log** (⌥⌘L): what the engine prints, warnings included.
- Settings (⌘,) → Engine: which engine this build has and its versions, restart; for the Python
  fallback, **Repair Python Engine** rebuilds its environment.
