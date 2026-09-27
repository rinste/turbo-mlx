# Making generation faster: a plan

*Plan, September 2026. No code changes yet; every lever names the measurement that decides
it. Facts about the engine come from reading mflux `main` (v0.20.0, commit 80bae91) on 27
September 2026; figures from other projects are linked.*

## Summary

Generation time is set by three things, in this order: how many times the transformer runs
(steps × guidance passes), how many tokens each run processes (pixels), and how big the
transformer is. Everything else (language, process, file formats) is fixed cost around the
loop: worth trimming, but worth 1–10% each, not 2×.

Ranked by what they would give this catalog:

| Lever | Applies to | Expected gain | Evidence | Cost |
|---|---|---|---|---|
| Step-distilled Qwen-Image: the Lightning LoRA for 2512 (4 steps, no CFG) instead of 20 steps × 2 passes | Qwen-Image | **~10×** (40 passes → 4) once the engine stops running the unconditional pass at guidance 1 (it does today), ~5× until then | [ModelTC/Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning), Apache 2.0, `Qwen-Image-2512-Lightning-4steps-V1.0` | Catalog entry, a pre-baked 4-bit checkpoint, a 10-line engine fix |
| Keep activations in bf16 (Z-Image and Qwen-Image run their residual stream in float32 today) | Z-Image, Qwen-Image | **~1.4×** on the denoising loop, measured: 78 s → 56 s on an M5 Air; more on M5, where float32 also bypasses the neural accelerators | [mflux issue 761](https://github.com/mflux-community/mflux/issues/761) | An upstream fix, or ours |
| Resolution and drafts: work scales with pixels | All | 1.8× at 768 px, 4× at 512 px, for iteration; nothing on the final image | Cost model below, matches the README timings | Defaults and a UI mode |
| Fixed costs: one PNG encode instead of three, prompt cache for Klein and Z-Image, encode every queued prompt before dropping the encoder, pre-warm the selected model | All; largest share on Klein and in *Save memory* | 1–3 s per image; 10–40 s per prompt change on Ming-Image and Qwen-Image with *Save memory* | mflux source (below) | Worker and protocol changes, small |
| Caching inside the loop (TeaCache, MagCache) and guidance only where it matters | Only models still run at 20+ steps with CFG (Qwen-Image without Lightning) | 1.5–2.7× at 20–50 steps; **nothing on 4–9-step models** | [mlx-teacache](https://github.com/IonDen/mlx-teacache): Qwen-Image CFG 50 steps 2.68× on an M1 Max; "distilled schedules skip nothing and gain nothing" | Engine change; quality to verify |
| Never swap: wired weights, predicted peak before starting | All | Prevents 3–10× slowdowns; not a speed-up | mflux sets no wired limit today | App and worker |
| Kernels and hardware: `compile` and no per-block host sync for Qwen-Image, dense instead of quantized matmuls when memory allows, MLX current, M5 | All | 5–15% each where something is missing; M5 is a hardware step (Apple: FLUX-dev 4-bit 3.8× vs M4) | mflux source; [lablup benchmark](https://github.com/lablup/mlxcel/issues/1994); [MLX 0.30.0](https://github.com/ml-explore/mlx/releases/tag/v0.30.0) | Upstream work; measurements |
| Workflow: live preview and early cancel, batch prompts encoded ahead | All | Fewer wasted generations | — | Native engine (see `native-engine.md`) |

Put together, and in the order of the phases at the end: Qwen-Image 2512 goes from tens of
minutes to about a minute and a half at 1024 px on an M1 Max; Z-Image Turbo from ~100 s to
~70 s; Klein from ~30 s to ~25 s; Ming-Image loses the model reload on every new prompt.
None of this changes the images' quality except the Lightning row, which trades a little
fine detail (dense small text, hair) for the speed, and must ship as a separate catalog
entry, not a replacement.

## A cost model that matches the measurements

Per step, a diffusion transformer does roughly `parameters × tokens` multiply-adds in its
linear layers plus `tokens² × width` per layer in attention. At 1024 × 1024 an image is 4096
tokens (16 pixels per token side in every family here) and attention is about 15% of the
work; it reaches 40% only at 2048 px. So time per pass grows about linearly with pixels up
to ~1.5 MP, and faster above.

Passes per image (steps × guidance passes) and relative work at 1024 × 1024:

| Model | Parameters | Steps | Passes per step | Passes per image | Work vs Klein |
|---|---|---|---|---|---|
| FLUX.2 Klein 4B | 4 B | 4 | 1 | 4 | 1 |
| Z-Image Turbo | 6 B | 9 (8 evaluated) | 1 | 8 | 3 |
| Ming-Image 0.1 | 6.15 B | 12 (11 evaluated) | 1 (guidance off) | 11 | 4.2 |
| Qwen-Image 2512 | 20 B | 20 | 2 (CFG 4) | 40 | 50 |

The measured times fit: Klein ~30 s and Z-Image ~100 s on an M1 Max (README) against a
predicted ratio of 3–3.4; Ming-Image 76 s at 1024 × 576, which is 2304 tokens (predicted
~75 s). By the same model Qwen-Image 2512 needs about 15 × Z-Image's time at equal
resolution: tens of minutes on an M1 Max, minutes on an M4 or M5 Max. That single row is
where most of the catalog's waiting happens, and it is the row with the cheapest fix.

## Where the rest of the time goes

For Klein, four passes are ~16–20 s of the ~30 s; the rest is fixed cost. Per image today:

| Cost | When | Size |
|---|---|---|
| Model load | First image; after "Free Memory"; **every new prompt in *Save memory*** for Ming-Image and Qwen-Image (the worker unloads and reloads the whole model so the text encoder and the transformer are never co-resident) | 5–40 s by model and SSD |
| Prompt encoding | Every image for Klein and Z-Image (mflux caches nothing for them: FLUX.2 creates a cache and never uses it); every new prompt for Qwen-Image (unbounded dict) and Ming-Image (16-entry LRU) | 1–10 s; Ming-Image's 16 B mixture-of-experts encoder is the slowest |
| VAE decode | Every image; tiled in *Save memory* where allowed (512 px tiles with 64 px overlap, cosine-blended in NumPy) | 2–10 s at 1024 px |
| PNG encode | Every image, **three times**: PIL at compression level 6, then re-opened and re-saved for EXIF metadata, then again for XMP/IPTC text chunks | 1–3 s for a 1 MP RGBA image, before the app is told the image is done |
| Worker start | App launch, restart, force stop | 2–5 s (Python + imports), not per image |

## What the engine does today

Verified in mflux's source; the points that matter for speed:

- **Float32 residual stream in Z-Image and Qwen-Image.** Z-Image's timestep embedding and
  RoPE tables are float32 and promote the whole stream (`z_image_transformer/transformer.py`,
  `attention.py`); Qwen-Image's initial noise is created in float32 and never cast, so the
  image stream and, through the joint concatenation, the text stream run in float32. FLUX.2
  and Ming-Image cast to bf16. Issue 761 measured the denoising loop at 78.3 s → 55.6 s
  (15.0 → 10.6 s per step) with a bf16 stream on a MacBook Air M5 at 1024², 9 steps. On M5
  the cost is doubled: MLX's neural-accelerator kernels (matmul, quantized matmul, attention)
  take bf16/fp16 only, so float32 activations fall back to the old kernels.
- **CFG is two sequential full passes, never one batched pass,** in every family, and
  **Qwen-Image runs the unconditional pass even at guidance 1.0** (`qwen_image.py`, lines
  110–124; `compute_guided_noise` then reduces to the identity). A Lightning run at guidance 1
  pays twice what it should. No guidance-interval or adaptive-guidance option exists.
- **`mx.compile`:** the whole step is compiled for FLUX.2 and Z-Image (not on M1/M2 below Max),
  not for Qwen-Image nor Ming-Image (whose README argues it is compute-bound).
- **Qwen-Image forces a host sync in every block:** `if mx.all(joint_mask >= 0.999)` evaluates
  an array to a Python bool 60 blocks × 2 passes per step, plus one `.item()` per forward.
  Each sync drains the GPU queue.
- **Attention** uses `mx.fast.scaled_dot_product_attention` in all four families; Z-Image
  always passes an all-zero additive mask instead of `None`.
- **No step caching, no sparsity, latent batch fixed at 1;** several seeds are a Python loop.
- **LoRA** is supported for FLUX.2, Z-Image and Qwen-Image (not Ming-Image) and baked into
  the weights by default; on a base below 8 bits the touched layers are re-quantized at 8
  bits, so a Lightning LoRA baked at run time onto the 4-bit Qwen-Image checkpoint roughly
  doubles the transformer's memory. `--no-bake-lora` keeps runtime adapters (two small extra
  matmuls per layer). Lightning for Qwen-Image text-to-image is exercised only in a slow test
  (`Qwen-Image-Lightning-4steps-V2.0`, 4 steps, guidance 1.0, q8).
- **Memory:** cache limit and tiling only; no wired limit (`mx.set_wired_limit`). Issue 760:
  the compiled step closure keeps Z-Image's transformer alive through the VAE decode under
  low RAM.
- **Defaults** match the model cards: Z-Image Turbo 9 steps (8 evaluated, guidance 0), Klein 4
  steps at guidance 1, Qwen-Image 20 steps (CLI guidance 3.5, API 4.0), Ming-Image 12 steps
  (11 evaluated) at guidance 1.
- **MLX 0.32.x**, which mflux pins, already carries the M5 neural-accelerator kernels for
  matmul, quantized matmul (group size 64, K multiple of 64) and attention at head dimension
  128, which all four families use. Nothing in mflux is M5-specific; the gains arrive with the
  MLX version, provided the activations are bf16.

## The levers

### 1. Fewer transformer passes: Qwen-Image

Qwen-Image 2512 is 20 steps with real CFG: 40 passes of a 20 B model per image. The other
three families are already distilled (Klein 4 steps, Z-Image Turbo 8, Ming-Image 12 without
CFG) and there is no distilled variant of Ming-Image. Options for Qwen-Image:

- **Lightning LoRA** ([ModelTC/Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning),
  Apache 2.0): 4- and 8-step LoRAs for Qwen-Image, and `Qwen-Image-2512-Lightning-4steps-V1.0`
  for the exact checkpoint in the catalog. Settings from the repository: `true_cfg_scale=1.0`,
  4 steps, exponential shift 3 with dynamic shifting. Authors' caveats: dense small text and
  hair detail are better on the base model; V2.0 of the non-2512 LoRA reduces
  over-saturation. Ship it as **its own catalog entry** ("Qwen-Image 2512 Lightning · 32 GB",
  4 steps, guidance fixed off), with the 20-step model still there for people who want the
  last bit of detail.
- **How to ship it:** a pre-baked checkpoint (LoRA merged in bf16, then quantized to 4-bit and
  saved in mflux format, once, by us) keeps the download at the size of today's 4-bit model and
  avoids the run-time re-quantization to 8 bits. Runtime adapters (`--no-bake-lora`) are the
  fallback for "Add Model…" users bringing their own LoRA.
- **The engine fix that doubles the gain:** skip the unconditional pass when guidance is 1.0
  (a guard around `qwen_image.py` lines 117–123). Without it Lightning gives ~5×; with it
  ~10×. Ten lines, upstreamable; until merged the app pins its own commit anyway.
- **Qwen-Image-Flash** (NVIDIA, 4 steps, distilled from the 2508 weights, NVIDIA Open Model
  License: commercial use allowed with an attribution notice) is the alternative if the
  Lightning quality does not convince; it needs its own mflux support and a 4-bit conversion.
  Second choice.

Measure: same seeds and prompts at 20 steps CFG 4 vs Lightning 4 steps; time, peak memory,
and a side-by-side review, with text-heavy prompts included on purpose.

### 2. bf16 activations for Z-Image and Qwen-Image

The single cheapest large win, and the one that matters most on M5: cast the timestep
embedding and RoPE outputs back to the weights' dtype in Z-Image (Ming-Image's transformer
shows the pattern, `ming_transformer.py`) and create Qwen-Image's noise in the model's
precision. Expected ~1.4× on the denoising loop (measured on M5; M1–M4 should see a similar
share since the float32 path doubles memory traffic and halves matmul throughput on every
generation). Verify parity against the float32 output (PSNR, and a visual pass on small text,
which is where precision shows first).

### 3. Fewer tokens

Work is linear in pixels: 768 px is 1.8× faster than 1024 px, 512 px 4× faster. Nothing to
implement in the engine; everything in the product:

- **Defaults per family** at the resolution the model was trained for and no more.
- **A draft mode** (512–768 px, the model's minimum steps) for finding the prompt, then the
  final image at full size, same seed. Seeds do not transfer across resolutions exactly, but
  composition and style mostly do, which is what drafts are for.
- **Show the estimate before generating:** the history already records seconds per image;
  from the last few images of the same model the app can predict this one (scaled by pixels)
  and show "about 40 s" next to the button. People choose smaller when they see the price.
- **An upscaler** for the draft-then-upscale flow later (mflux ships SeedVR2 3B; a lighter
  Core ML super-resolution model would do for 2×).

### 4. Fixed costs

- **One PNG encode.** Save once, at compression level 1–3 (three to five times faster than
  level 6 at ~10–20% larger files), with the metadata chunks written in the same pass; or hand
  the app the image before the PNG exists (the native engine's `CGImage` path) and compress in
  the background. 1–3 s per image, i.e. 5–10% of a Klein image.
- **Prompt cache for Klein and Z-Image.** The worker can encode once per distinct prompt as it
  already does for Qwen-Image, or the fix goes upstream (FLUX.2's unused `prompt_cache`).
  0.5–2 s per image in a batch of seeds.
- **Encode every queued prompt before dropping the encoder.** In *Save memory* the worker
  unloads and reloads the whole model whenever a queued job brings a new prompt. The queue is
  known: when the encoder is resident, encode the prompts of all pending jobs of the same
  model (a `prefetch` command, or the prompts sent along with `generate`), then release it.
  Removes a 10–40 s reload per prompt change on Ming-Image and Qwen-Image.
- **Pre-warm.** Load the selected model when the app opens or the selection changes (an
  option; it costs memory while idle), so the first image starts at the denoising phase. Hides
  5–40 s once per session.
- **Untiled decode when it fits.** Tiles cost the overlap and the blend; when the predicted
  peak is under the budget, decode in one go. Seconds per image at 1024 px, more at 1536.
- **The Qwen-Image per-block sync and the Z-Image zero mask** are upstream one-liners (compute
  the mask decision once per forward from the tokenizer's attention mask; pass `None`). Small
  gains, worth having while touching those files for the bf16 change.

### 5. Skipping work inside the loop

TeaCache, First-Block Cache, MagCache and TaylorSeer skip or extrapolate transformer passes
when consecutive steps change little. They work at 20–50 steps and do nothing at 4–9: the
MLX port of TeaCache measured 2.68× on Qwen-Image at 50 steps with CFG, 1.57× on FLUX.1-dev at
25, and states that Klein's and schnell's distilled schedules "skip nothing and gain nothing".
Guidance interval (CFG only at middle noise levels) and adaptive guidance (stop CFG once the
two predictions agree, ~25% fewer passes) apply only to CFG models.

For this catalog that means: worth it for Qwen-Image **at 20 steps** (the quality entry), for
the Klein base variants, and for future 20+-step models; not for Klein, Z-Image Turbo, Ming
or Lightning. Do it after lever 1, in the native engine or upstream, and gate it on a parity
review: caching artifacts are subtle (softened detail, drift on long generations).

### 6. Kernels and hardware

- **`compile` for Qwen-Image** and a step compiled without the per-block sync: 5–10% by analogy
  with the compiled families; measure.
- **Dense matmuls when memory allows.** At 4096 tokens a step is compute-bound, so 4-bit
  weights save memory, not time; the one benchmark found (LLM prefill on an M1 Ultra) has
  dequantize-then-dense beating `quantized_matmul` by 4–13% from 1024 rows up. A
  "dequantize at load" option for Macs with headroom (Z-Image 6 B in bf16 is 12 GB) is a
  one-afternoon experiment per family; keep it only if it measures.
- **MLX version.** The neural-accelerator kernels landed in MLX 0.30 (macOS 26.2+) and were
  tuned for M5 Pro/Max in 0.31–0.32; mflux is on 0.32.x. Apple reports FLUX-dev 4-bit at
  1024² "more than 3.8× faster" on M5 than M4 with MLX. The app inherits this by keeping the
  engine current; the bf16 lever is what makes Z-Image and Qwen-Image eligible for it.
- **Batching seeds** in one pass (batch 2–4 latents): compute-bound already, so expect ≤10%
  and double the activation memory; low priority, measure once.

### 7. Never swap

Swapping turns a 10 s step into a minute. Three cheap defences: wire the weights
(`mx.set_wired_limit`, macOS 15+) so they are not paged out under pressure; predict the peak
before starting (the catalog's measured peaks scale with pixels; refuse or warn when the
prediction exceeds the budget, instead of letting the user discover it); and fix the leak of
issue 760 so *Save memory* actually frees the transformer during decode.

### 8. Workflow

The fastest generation is the one not run to the end. With the engine in-process (see
`native-engine.md`) a low-resolution preview every couple of steps costs almost nothing and
lets people stop a bad image at step 2 of 9; the FLUX.2 Swift port already exposes a per-step
image callback. Draft mode (lever 3) and queue-ahead prompt encoding (lever 4) belong to the
same idea: fewer seconds between an idea and a judgment.

### 9. Video

LTX-2's own fast path is the distilled two-stage pipeline (8 steps at half resolution, 4 at
full with a latent upscaler); caching at 8 steps is unproven, and the memory cliff is the
decode, which the official pipeline tiles. The levers that carry over unchanged: bf16 stream,
no wasted CFG pass, one write of the output, prediction of the peak before starting.

## What not to expect

- Swift instead of Python in the loop: the same kernels, the same time.
- 4-bit instead of 8-bit weights: less memory, not less time, at these token counts.
- Caching or fewer steps on Klein, Z-Image Turbo, Lightning: already at the floor; fewer
  steps break them.
- Batching seeds: small.

## Measuring

Before changing anything, make the numbers visible and repeatable:

- **Per-phase timers** in the worker events (load, encode, per step, decode, save), stored
  with each history item next to `seconds` and `peakMemory`. Today only the total is kept.
- **A benchmark script** (`turbo_worker.py bench`, or the native engine's `verify`): the
  catalog's models × three sizes × fixed prompts and seeds, reporting seconds per phase and
  peak memory; run on the developer's Macs, results in a table in `docs/`.
- **Parity for anything that touches pixels** (bf16 stream, Lightning, caching, dense
  matmuls): same seed before/after, PSNR plus a visual pass on text-heavy prompts.
- **One Metal System Trace per family** to see GPU idle gaps (the Qwen-Image syncs will show
  as bubbles) and the top kernels.

## Phases

**Phase A — days, no change to any image.** One PNG encode; prompt cache for Klein and
Z-Image in the worker; skip Qwen-Image's unconditional pass at guidance 1; wired memory limit;
pre-warm option; per-phase timers and the benchmark script. Klein −5–10%, Qwen-Image at
guidance 1 half the time.

**Phase B — weeks, the big two.** bf16 stream for Z-Image and Qwen-Image (upstream PR from
issue 761's measurements, pinned in the app meanwhile): Z-Image ~1.4×. Qwen-Image 2512
Lightning as a catalog entry with a pre-baked 4-bit checkpoint: ~10× on the slowest model.
Queue-ahead prompt encoding in *Save memory*; predicted-peak check; draft mode and time
estimate in the UI.

**Phase C — after B.** Caching and guidance interval for the 20-step Qwen-Image entry;
`compile` and the sync fix for Qwen-Image; the dense-matmul experiment; untiled decode when it
fits; batched seeds measured once.

**Phase D — with the native engine.** Live preview and finer cancellation; M5 tuning as MLX
moves; the same levers applied to the first video family.

## Risks

- **Quality regressions hide behind speed.** Lightning changes the look; bf16 can change small
  text; caching softens. Every lever that touches pixels ships behind the parity check and,
  where it changes the model, as a separate catalog entry.
- **Upstream latency.** The bf16, CFG-skip, prompt-cache and sync fixes belong in mflux; the
  app can carry them on its pinned commit (a fork) until they merge, at the cost of rebasing.
- **Memory creep.** A LoRA baked at run time on a 4-bit base doubles the transformer; dense
  matmuls double the weights. Both are opt-in by budget, never defaults below 64 GB.
- **Measurement noise.** Thermal throttling and background load move per-step times by
  10–20%; compare medians of repeated runs, same machine, same session.
- **M5-only gains.** The neural-accelerator numbers do not apply to M1–M4; the plan's
  M1-Max estimates above exclude them.
