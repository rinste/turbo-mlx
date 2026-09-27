# Turbo MLX

A macOS app (SwiftUI) that generates images on the Mac's GPU with MLX. It is built to be
distributed: the people who install it never open Terminal and need neither Python nor Xcode. On
first launch the app sets up its own engine and downloads the models the user picks.

![Turbo MLX: settings on the left, the image with its zoom controls and metadata on the right, the history below](docs/screenshot.webp)

- **Left column:** model (with download and status), the prompt as blocks (each one a piece of the
  final text, so the subject and the style can be written separately; renamable, and put in order
  by dragging), format (aspect ratio and resolution, or a custom size), steps, guidance, seed, number
  of images, transparent or white background, memory saving. The main button does what the current state needs:
  *Install Image Engine* → *Download and Generate* → *Generate*.
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

Measured on an M1 Max with mflux 0.20 (the 8-bit Z-Image peak adds its larger weights to the
measured 4-bit one). A model is listed under the smallest common Mac memory size it peaks under 80%
of. Times at 1024 × 1024: FLUX.2 Klein ~30 s, Z-Image Turbo 4-bit ~100 s; Ming-Image takes 76 s at
1024 × 576. The model stays loaded between images.

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

Then ⌘R in Xcode. To try the first-run experience without touching your real installation, point
the app at an empty data folder:

```bash
open -n --env TURBO_MLX_HOME=/tmp/turbo-first-run "path/to/Turbo MLX.app"
```

## How it works

```
SwiftUI ── JSON lines (stdin/stdout) ──▶ turbo_worker.py serve ──▶ mflux / MLX (GPU)
                                    └──▶ turbo-engine serve   ──▶ MLX Swift (GPU)   FLUX.2 Klein
```

- **Engine.** Inference runs in [mflux](https://github.com/mflux-community/mflux) (Python + MLX).
  `Backend/setup_backend.sh` creates the environment with **uv, which ships inside the app**
  (`Vendor/uv`, copied to `Contents/MacOS` and signed with the app). uv also downloads a
  standalone Python, so the user's Mac needs nothing preinstalled. mflux is installed from the
  GitHub source archive of the tested commit (`BackendController.mfluxCommit`), so git isn't needed
  either: Ming-Image support landed after mflux 0.20.0, the latest release on PyPI. When a new
  version of the app expects a different commit, the engine updates itself on first launch.
- **Worker.** `Backend/turbo_worker.py serve` keeps the model in memory, reports phases,
  per-step progress and the seconds each phase took (shown when hovering the time of an image),
  and stops a generation at the next step. One adapter per model family (`FAMILIES`) hides the
  differences between mflux's classes. It loads the selected model as soon as a prompt is being
  written, encodes the prompts of the queued images while the text encoder is in memory, keeps
  the model's buffers wired while it works, and writes each PNG once (mflux's own save encodes
  it three times).
- **Save memory.** Frees the text encoder of Ming-Image and Qwen-Image once the prompt is read
  (a new prompt reloads the model, so the text encoder and the transformer are never in memory
  together), decodes the image in tiles where the VAE allows it (not FLUX.2, whose tiles would
  show seams), and keeps MLX's buffer cache small.
- **Native engine.** `Engine/` is a Swift package that runs FLUX.2 Klein on MLX Swift and speaks
  the same protocol; when its `turbo-engine` binary is present (`scripts/build-engine.sh` installs
  it in the app's data folder) Klein images run there and the Python engine serves the rest. See
  `Engine/README.md` and [docs/native-engine.md](docs/native-engine.md).
- **Downloads.** The app downloads models itself (`Services/HubDownloader.swift`) into the
  Hugging Face cache, in the same layout huggingface_hub uses, so mflux and other tools share
  them; an interrupted download resumes. No engine is needed to download.
- The app reads the login shell's environment (PATH, `HF_HOME`, `HF_TOKEN`…).

| What | Where |
|---|---|
| Python environment and private Python | `~/Library/Application Support/TurboMLX/{venv,python}` |
| History (PNGs + `history.json`) | `~/Library/Application Support/TurboMLX/History` |
| Package cache | `~/Library/Caches/TurboMLX/uv` |
| Models | `~/.cache/huggingface/hub` |

```
TurboMLX/
  App/        entry point, AppModel (queue, worker events, selection, persistence)
  Models/     catalog and families, settings, jobs, history items
  Services/   child processes, engines, downloads, Hugging Face cache, history
  Views/      left column, output, history, settings, log
  Backend/    turbo_worker.py, setup_backend.sh (copied into the bundle)
Engine/       the native engine (Swift package: turbo-engine, TurboEngineCore, fixtures)
Vendor/uv/    the uv binary (update with scripts/update-uv.sh)
scripts/      release.sh, build-engine.sh, ExportOptions.plist, update-uv.sh
```

The project uses Xcode's synchronized folders: files added under `TurboMLX/` join the target on
their own.

**Adding an mflux model family:** a case in `ModelFamily` (`Models/ModelCatalog.swift`: download
patterns, components, steps, guidance) and an adapter in `FAMILIES` in `turbo_worker.py`.

**Where this is going:** [docs/native-engine.md](docs/native-engine.md) weighs replacing the Python
worker with an MLX Swift engine and prepares the app for video models (LTX).
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

The DMG ends up in `build/release/`.

## Troubleshooting

- **Engine Log** (⌥⌘L): mflux output, warnings and tracebacks.
- Settings (⌘,) → Engine: restart, **Repair Engine** (rebuilds the environment), versions.
