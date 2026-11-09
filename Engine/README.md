# turbo-engine

The engine of Turbo MLX: the catalog's families (the image models, LTX-2 for video, the SeedVR2
upscaler, Idol, which makes a 3D object of a picture, MiniMax-Music3, which makes songs, and MOSS-SoundEffect,
which makes sound effects), and Qwen2.5-VL, which writes a prompt from a picture (Describe Image), on
[MLX Swift](https://github.com/ml-explore/mlx-swift), behind a JSON-lines protocol (below) that
the app speaks to it; the Python engine the app used to ship, `Reference/turbo_worker.py`, speaks
it too for the image families (see `docs/native-engine.md` for the why and the plan).

```
Sources/turbo-engine/         the executable: `serve` (the worker), `verify`, `verify-tokenizers`
                              and `compare` (parity checks), `bench` (the catalog's times and peaks)
Sources/TurboEngineCore/      protocol, mflux checkpoint loading, PNG output, the families
  Tokenizer/                  the tokenizers, read from `tokenizer.json`: byte-level BPE (Qwen,
                              Ling) and SentencePiece-style BPE with byte fallback (Gemma 3)
  Families/Family.swift       the FamilyModel protocol the engine drives, and the loader by family
  Families/Shared/            what families share: the Qwen3 prompter, the S3-DiT, the schedules,
                              tiled VAE decoding, mixture-of-experts layers, pixels, Pillow's
                              bicubic and Lanczos resizes
  Families/Klein/             FLUX.2 Klein: Qwen3 text encoder, FLUX.2 transformer, VAE decoder,
                              and the VAE encoder for the pictures an image is edited from
  Families/Flux3/             FLUX 3 Image Klein: the FLUX 3 transformer (per-mode blocks, then
                              single-stream ones), the flow-map schedule, the pictures' preparation;
                              Klein's Qwen3 encoder (Qwen3-VL's language tower) and FLUX.2's VAE
  Families/MossImage/         MOSS-Image: Qwen3.5-9B's language model (gated delta net layers, their
                              recurrence in a Metal kernel, and gated attention), the single-stream
                              DiT with one shared modulation, the snapped schedule; FLUX.2's VAE with
                              BFL's small decoder
  Families/ZImage/            Z-Image Turbo: the Qwen3 encoder read in float32, the S3-DiT,
                              the 16-channel decoder
  Families/QwenImage/         Qwen-Image 2512: Qwen2.5-VL text encoder, the dual-stream
                              transformer, the causal 3D decoder, classifier-free guidance; and
                              Qwen-Image-Edit 2511: the vision tower and the VAE encoder for the
                              picture an image is edited from; and Describe Image's Qwen2.5-VL 7B,
                              the same modules with the head, a key–value cache, the multimodal
                              rotary positions and greedy decoding
  Families/Ming/              Ming-Image: the Ling MoE encoder, the Qwen2 connector and heads,
                              the S3-DiT in bf16, the RGBA decoder
  Families/SenseNova/         SenseNova-U1.5: the two Qwen3 stacks (the prompt's, read once into a
                              key–value cache, and the image's), the pixel-space patch embedding
                              and pixel decoder, the flow loop; checkpoints are MLX packs
  Families/LTX/               LTX-2.3: Gemma 3 12B and the text connector, the audio–video
                              transformer, the video VAE (tiled decode, image encoder), the ×2
                              latent upsampler, the audio VAE and the 48 kHz vocoder, the
                              distilled two-stage pipeline
  Families/MiniMaxMusic3/     MiniMax-Music3: the official prompt rules in Python's string
                              semantics, Qwen3-8B with a key-value cache and the RVQ depth decoder
                              drawing the frames with guidance (mlx-audio's keys), the condition
                              encoder, the 1D flow transformer on overlapping windows, the vocoder
  Families/MossSoundEffect/   MOSS-SoundEffect v2.0: ftfy's text fixes and the Wan prompter's cleaning,
                              Qwen3-1.7B's last state, the Wan 2.1 audio DiT over 30 s of 50 Hz latents
                              with guidance, the DAC VAE's decoder at 48 kHz mono
  Families/Idol/              Idol: the picture's cutout (BiRefNet: Swin-L, deformable ASPP, its
                              columns in a Metal kernel) and crop in Pillow's arithmetic, DINOv3
                              ViT-H+/16, FLUX.2's VAE encoder and its posterior's sample, the
                              denoiser (learned per-token rotation), the octree and Gaussian
                              decoders, the GLB of discs
  Families/SeedVR2/           SeedVR2 3B: the causal 3D VAE (tiled encode and decode), the
                              windowed transformer, one flow step, the wavelet and Lab color
                              correction; the picture's preparation with Pillow's bicubic
  LoRA/                       LoRAs on a transformer: mflux's key mappings (the tables exported by
                              Fixtures/export_lora_mappings.py) and the adapted linear layer
  Resources/                  SeedVR2's fixed text embedding, as mflux ships it
VideoOutput.swift             MP4 (H.264 + AAC) with AVAssetWriter, frames as they are decoded
Fixtures/make_*_fixture.py    build the checkpoint + references `verify` compares against
                              (make_lora_fixture.py adds LoRA files to three of them)
Fixtures/export_lora_mappings.py  writes LoRA/LoRAMappingTables.swift from mflux's LoRA mappings
Fixtures/tokenizers.json      transformers' ids for a corpus of prompts, every family's pipeline
                              (made by make_tokenizer_corpus.py), for `verify-tokenizers`
Fixtures/requirements.txt     the mflux revision they (and the reference worker) run with
Fixtures/requirements-ltx.txt the ltx-2-mlx revision the LTX-2 port follows
Fixtures/requirements-sensenova.txt  SenseTime's code and the Python the SenseNova checks run with
Fixtures/requirements-moss.txt  OpenMOSS's pipeline file and the Python the MOSS-Image checks run with
Reference/turbo_worker.py     mflux behind the same protocol, to compare real images
Reference/sensenova_reference.py  SenseTime's code on an MLX pack's weights, for SenseNova's
Reference/idol_reference.py   idol-sdk's code on the MLX pack's weights, for Idol's checks
Reference/music3_reference.py  mlx-audio on the MLX pack, stage by stage, and the official prompt
                              functions on a corpus, for MiniMax-Music3's checks
Reference/moss_image_reference.py  OpenMOSS's pipeline on a checkpoint, for MOSS-Image's
Reference/qwen25vl_reference.py  mlx-vlm on the describer's pack, picture by picture, for Describe Image's checks
Fixtures/make_moss_soundeffect_fixture.py  MOSS-SoundEffect's prompt corpus, Qwen3's float32 states and the
                              schedule from PyPI code, with the goldens of OpenMOSS's code (see below)
Licenses/                     the licenses of the projects the ports follow (mflux, ltx-2-mlx,
                              SenseNova-U1, OpenMOSS's MOSS-Image pipeline, mlx-lm with
                              mlx-swift-lm for the mixture-of-experts layers and the gated delta
                              rule's kernel, Pillow for its resizes, tokenizers for the tokenizer,
                              mlx-audio, minimax-music3-mlx and diffusers for MiniMax-Music3,
                              MiniMax-Music3's own licence, OpenMOSS's MOSS-TTS and ftfy for
                              MOSS-SoundEffect, mlx-vlm for Describe Image), which the app's acknowledgements
                              reproduce
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
| FLUX 3 Image Klein | Qwen3-VL-4B's language tower (bf16 or quantized on loading), hidden states of every fourth layer from 4 to 32, padded to a multiple of 80 tokens with the padding at rotary position 1 | 5 per-mode blocks (image, pictures, text: each attends only to itself), then 28 single-stream blocks over [image, pictures, text], each token modulated by its mode; the distilled checkpoint is a flow map, told each step's target time; an edit's pictures (one padded by reflection to a 4 MP canvas, or up to ten cut to 1 MP each) at layers −1, −2… | FLUX.2's, the decoder whole, from the latents in bf16 (mflux: float32) | Base: CFG against the empty prompt |
| MOSS-Image (SFT) | Qwen3.5-9B's last hidden state over the whole chat-templated prompt (thinking on, the text trimmed, at most 4096 tokens), unpadded: three layers in four a gated delta net (causal depthwise convolution, l2-normalized keys, the decay-gated delta rule in float32), the fourth gated attention with a quarter of each head rotated | 42 single-stream blocks over [image, text], all modulated by one shared adaLN of the timestep, SiTU-gated feed-forward; the prompt first refined by two text blocks; a 4-axis rotation (the image's rows and columns centred, its frame at the prompt's length); bf16, or 8 bits on loading | FLUX.2's encoder with BFL's small decoder (96–384 channels), from the latents in bf16 (OpenMOSS: float32) | CFG 3 against the empty prompt, its pass skipping block 22 |
| Z-Image Turbo | Qwen3 4B in float32, second-to-last state, real tokens only, thinking on | S3-DiT, tokens padded to 32 with learned pad tokens, float32 stream | FLUX.1-style, 16 channels, tiles with Save memory | off |
| Qwen-Image 2512 | Qwen2.5-VL 7B (bf16, unquantized), the template's 34 tokens dropped | 60 dual-stream blocks, float32 stream, modulation producers at 8 bits | Wan-derived causal 3D, 16 channels, tiles | true CFG, rescaled to the conditional norm; the unconditional pass is skipped at 1 |
| Qwen-Image-Edit 2511 | Qwen2.5-VL 7B with its vision tower: the picture at ~384 × 384 (Pillow's bicubic twice, as mflux) in 14-pixel patches, windowed attention, 2 × 2 merged; its tokens replace the placeholders of a 64-token template, everything in float32, then float16 | Qwen-Image's; the picture's latents follow the image's (frame position 1) and are modulated at timestep 0 (`zero_cond_t`), only the image's come out | Qwen-Image's, plus the encoder for the picture at the image's size (its own proportions when they differ; Pillow's Lanczos) | as Qwen-Image, negative prompt empty |
| Ming-Image 0.1 Design | Ling-mini-2.0 MoE (256 experts, 8 routed with group-limited top-k, bf16 router as upstream), Qwen2 connector over 256 query tokens, direct-VLM head | S3-DiT in bf16, no padding, two caption streams | Qwen VAE for RGBA, one scaling factor, tiles | zeroed conditions |
| SenseNova-U1.5 8B-MoT (8-step) | the understanding stack (Qwen3, 42 layers of 4096, each head's first half rotated by text position, the rest by row and column, separate norms) reads the "neo1_0" query with SenseTime's system message into a key–value cache; guidance above 1 adds an unconditional query; an edit's picture (resized with Pillow's Lanczos to about the image's area, within 512² and 2048²) goes in through the stack's own patch embedding, its tokens between `<img>` and `</img>` at one text position, seeing each other; guidance's unconditional query is then the picture alone | the generation stack (the same layers' `_mot_gen` weights) over the image's tokens (32 × 32 pixels: a 16-pixel patch embedding with a 2D rotation, merged 2 × 2) plus time and noise-scale embeddings, attending to the cache without a mask; activations bf16 as the reference | none: a convolutional pixel head (pixel shuffles, two 3 × 3 convolutions) predicts the clean image, the velocity is derived from it; Euler steps from t = 0 on a schedule shifted by 3, noise scaled with the image's size | off (the distilled packs); true CFG otherwise |
| LTX-2.3 distilled | Gemma 3 12B (4-bit, all 49 hidden states, prompt left-padded to 1024), per-token RMS, two projections and two 8-block connectors with learnable registers | 48 audio–video blocks (4096 + 2048 wide, cross-modal attention both ways), block linears 4 or 8 bits, float32 activations | causal-3D conv VAE (non-causal decoder, 32 × 32 × 8, 128 channels), tiled over frames and pixels; audio VAE + BigVGAN vocoder with bandwidth extension to 48 kHz | none (distilled); two stages: 8 steps at half size, ×2 latent upsampler, 3 steps |
| Idol (3D object) | none: the picture, cut out by BiRefNet (unless it has an alpha), cropped around the object (1.2 × its box for a render, 1.7 for a photo) to 1024 on black; its DINOv3 ViT-H+/16 tokens (bf16, the affine final norm then a plain LayerNorm) and a sample of the FLUX.2 VAE's posterior, patchified and normalized, 4,101 tokens each | a rectified-flow transformer over the object's 8,192 × 16 latent and a camera token: the latent refined by two time-modulated blocks, the picture by two plain ones (once per guidance branch), then 24 joint blocks; each block's rotation learned per token (RePo); 8-bit linears, float16 stream, float32 norms and heads; 50 Euler steps on a grid shifted by 3 | an octree descent (eight levels, systematic resampling) places an anchor per 32 Gaussians, a cross-attending decoder turns each into 32; written as a GLB of discs, turned so the picture's view faces +Z | CFG 3 against zeroed picture tokens, the unconditional pass with them collapsed to two weighted keys |
| MiniMax-Music3 (song) | the description cleaned and the lyrics normalized as the official integration does (code point for code point), between the checkpoint's special tokens, at most 5,000; the unconditional prompt the same with every token but the first and last two replaced by the guidance token | a hierarchical autoregressive stage: Qwen3-8B (36 layers of 4096, 8-bit, its own 200k vocabulary and head) draws each frame's semantic code (25 a second) and a 4-layer depth decoder its seven residual codes, both with guidance 1.5 and top-50 sampling from mlx-audio's keys; each frame's hidden states (the global model's and the depth decoder's seven) mixed by learned weights into the condition; then a 36-layer 1D flow transformer (2048 wide, partial rotary, 8-bit) on windows of 200 frames every 100, 30 Euler steps each, the overlap pulled toward the previous window's | DAC-style vocoder (snake activations, ×512) to 44.1 kHz stereo, the middle of each window kept; a WAV | CFG 1.5 (language models), 1.7 (flow, against a zero condition) |
| MOSS-SoundEffect v2.0 (sound effect) | the prompt stripped with " duration: {s:.1f}s" appended, cleaned as `WanPrompter` does (ftfy's fixes without its mojibake repair, HTML references, Python's strip, runs of whitespace made one space), Qwen3's tokens with no special ones, at most 512; Qwen3-1.7B's last hidden state after its final norm (bf16), zeros past the prompt up to 512 | Wan 2.1's 1.3B DiT (30 blocks of 1536, 12 heads) over 1500 latent frames, always 30 s whatever the length: 1-D rotary on interleaved pairs, q and k RMS-normed over the whole width, unmasked cross-attention to the 512 text states through a GELU projection; bf16 projections with the norms, modulation and residual stream in float32, as the reference's autocast; 100 Euler steps on a shift-5 schedule | the continuous DAC VAE's decoder (snake activations, ×960 in five transposed convolutions, 2048 → 64 channels) in float32 to 48 kHz mono, only the latents the kept seconds need (16 frames of margin); a WAV | CFG 4 against the negative prompt, by default empty (all-zero states) |
| SeedVR2 3B (upscaler) | none: a fixed 58 × 5120 embedding | 32 blocks of 2560 (10 with separate video and text weights, then shared), the video's 2 × 2 patches attending in windows (shifted every other block) with the whole text, rotary frequencies read from the checkpoint, float16 weights and float32 activations; input: noise, the picture's latent and a mask of ones; one Euler step from t = 1000 | causal 3D VAE, 16 channels, encode and decode in 512-pixel tiles; then the picture's low frequencies under the result's detail (five-level wavelet) and its Lab a/b (and 20% of L) histograms | none |

Every family keeps a prompt cache and encodes the queued prompts while its text encoder is
resident. With *Save memory*, Qwen-Image and Ming-Image release the text side once the prompts are
encoded and reload only it when a new prompt arrives (after releasing the transformer, so the two
are never co-resident); the other two keep everything loaded. Qwen-Image-Edit encodes its prompt
with the picture, so at the start of each image rather than ahead for the queue, and caches the
last few prompt–picture pairs; with *Save memory* it releases its text side the same way.

Two things a request can ask for, both off the reference's path by design. `precision: "bf16"`
(the app's *16-bit precision*) keeps the transformer's residual stream in the weights' precision
where the reference promotes it to float32: Z-Image's, through its float32 timestep embedding
(the S3-DiT then runs as it does for Ming-Image, whose config asks for that), and Qwen-Image's
and its editor's, whose latents the reference creates in float32 and never casts (they start in
bf16 instead; the guidance's rescaling stays in float32, and so does the decode). The other
families run in 16 bits already. `preview: true` (the app's *Live preview*) makes the image
families report the image as it forms: at some steps (every one of a short run, six or so of a
long one, never the last) the clean image the model predicts, `latents − σ · velocity`, is decoded
small (the latent grid averaged down to about 384 pixels on the longer side, through the family's
own decoder in one piece; SenseNova's pixels directly) and written beside the output, whole or
not at all, with a `preview` event; the file goes once the image is decoded.

The Qwen/Ming VAE's mid block attends over every pixel of the latent frame with one head of 384
channels, which MLX's fused attention does not take (its kernels stop at 256), so the scores would
be materialized in full: 1 GB in float32 at 1024 × 1024, and their softmax as much again. They
are computed a chunk of pixel rows at a time instead (`ChunkedAttention`, 256 MB of scores at
most), the same math to rounding.

`loras` (a list of files, each with a `scale`) puts LoRAs on the transformer of Z-Image, FLUX.2
Klein, FLUX 3, Qwen-Image and its editor, the families mflux applies them to, and of LTX-2.3 and LTX-2.5
(both stages, both transformers), as ltx-2-mlx does (`LoRA/`). A file's keys are
looked up in mflux's own mappings, exported to `LoRAMappingTables.swift` by
`Fixtures/export_lora_mappings.py`, once their prefixes (`diffusion_model.`, `transformer.`, …) and
matrix names (`lora_A`, `lora_down`, `lora.down`, …) are stripped; Kohya's underscored keys are
read through the same rows, and BFL's fused `qkv` gives the query, key and value layers a third of
its up matrix each, as mflux does (and FLUX 3's final modulation, `(shift, scale)` in BFL's files, has
its halves swapped). A key no row names is taken as it is when it names a linear
layer of the model (Qwen-Image's `img_in` or `proj_out`, which mflux's mapping leaves out, as it
does the modulation layers, here mapped too). LTX-2's keys need no table: ltx-2-mlx's renames
(`to_out.0` → `to_out`, `ff.net.0.proj` → `ff.proj_in`, `linear_1` → `linear1`, …), on whole
segments and in Kohya's underscored spelling as well, give the names its modules (and the port's)
have. Each adapted layer then adds `scale · (x·A)·B` to its
own output (B times alpha / rank when the file has an alpha, as ComfyUI does; ltx-2-mlx leaves
alphas out, which Lightricks' files do not have), as mflux's `LoRALinear` computes it before it
bakes the sum into re-quantized weights (ltx-2-mlx always bakes it): the engine never bakes, so the quantized
weights stay as they are, another LoRA or strength only swaps the small matrices (in well under
a second, without loading the model again), and none gives the model back bit for bit. Qwen-Image
and LTX-2 put them back on a transformer that Save memory had released. A file
whose layers are not the model's, or not of its size, fails the image with a message saying so;
LoKr and DoRA adapters are refused.

## Building

Requires Xcode 26 (mlx-swift 0.32.2 asks for a Swift 6.3 toolchain) and a Mac with Apple silicon.
`swift build` alone does not compile mlx-swift's Metal shaders on macOS; use Xcode or `xcodebuild`
(from `Engine/`, with `-skipPackagePluginValidation` for mlx-swift's package plug-in).

The app's build does it: its *Embed turbo-engine* phase runs `scripts/embed-engine.sh`, which
builds the engine in Release (`scripts/build-engine.sh`, derived data in `build/engine`) and puts
it in the bundle, `turbo-engine` in `Contents/MacOS` and the resource bundles it loads
(mlx-swift's Metal library among them) in `Contents/Resources`, signed like the app and with
`turbo-engine.entitlements`, which make it inherit the app's sandbox (a sandboxed app can start
no other helper, and none from outside its bundle). Start the app: the engine status shows
"turbo-engine 0.4" and every model runs on it. The engine is rebuilt only when `Engine/`
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
or `-q8`, with `mlx-community/gemma-3-12b-it-4bit`, or `dgrauet/ltx-2.5-mlx-q4`, which carries its
own Gemma 4), recorded stage by stage by hooks around its own functions:

```bash
$PY Engine/Fixtures/make_ltx_fixture.py <pack> build/fixtures/ltx --gemma <gemma>   # 2.3, text to video
$PY Engine/Fixtures/make_ltx_fixture.py <pack> build/fixtures/ltx-i2v --gemma <gemma> --image picture.png
$PY Engine/Fixtures/make_ltx_fixture.py <2.5 pack> build/fixtures/ltx25             # 2.5: no --gemma
$PY Engine/Fixtures/make_ltx_fixture.py <2.5 pack> build/fixtures/ltx25-kf --image start.png --last-image end.png
#    then, with a fixture's own arguments, random LoRAs and the first pass with them
$PY Engine/Fixtures/make_ltx_fixture.py <2.5 pack> build/fixtures/ltx25 --lora build/fixtures/ltx25/lora
build/bin/turbo-engine verify build/fixtures/ltx
```

`verify` feeds each stage the reference's own inputs: the tokens (identical), Gemma's states, the
connector on the reference's 49 states (bit-identical), the noise and the positions (identical),
every transformer pass of both stages (within 1e-4), the upsampler, the image encoder, the
decoders and the vocoder. Three stages are shown but do not decide, for the reasons the output
gives: Gemma's deeper states and the contexts they lead to carry bfloat16 rounding through 48
layers; the first stage's loop moves a bfloat16 latent by about one unit of its last place at each
early step, so two correct loops end a percent apart; and the video encoder and decoder, whose
bfloat16 convolutions follow MLX's conv3d. The encoder and decoder are therefore also compared in
float32, where they match within 1e-4 (the generator writes those references too, from the
reference's modules with upcast weights, next to Gemma's 49 states for the connector's check).
Since the engine's mlx-swift carries the reference's MLX (0.32.2 on both sides, 29 September
2026) the bfloat16 decoders match exactly as well, where MLX 0.31's conv3d left them 25% apart on
the fixture; the text connector, bit-identical before, now differs by about 1.5% (the two builds
of the same MLX round a kernel differently), well within the tolerance. Both packs pass, the
4-bit one from a prompt and from an image.

LTX-2.5 (30 September 2026, the 4-bit pack, from a prompt and from an image) adds its Gemma 4
tower, the ancestral sampler (stage 1's at first, both stages' since 31 October; each noise draw
is recorded and matches exactly), and the keyframe marker on the first latent frame of both
stages; every transformer pass is again within 3e-4. Its connector is more sensitive to bfloat16
than 2.3's: the reference itself moves by about 1 in 20 between bfloat16 and float32, so the
connector is shown in bfloat16 and decided in float32 (`connector_f32.safetensors`, within 5e-5).
Its DurationHead, which picks a clip's length from the prompt's contexts (`auto_duration` in a
request, `frames` then the longest), gives the reference's seconds exactly on its contexts, and
the seconds-to-frames rule matches on a table of 26 cases.

LoRAs (9 October 2026): `--lora` writes two random LoRA files over blocks 0, 1 and 47 and the
layers outside the blocks (130 layers), keyed as Lightricks' (`diffusion_model.….to_out.0`,
`ff.net.0.proj`, `linear_1`) and as Kohya's (with an alpha), and the first pass with each applied
unbaked, against which the port's pass lands within 5e-5, video and audio, on 2.3 (8-bit) and 2.5
(4-bit); taking them off gives the plain pass back bit for bit. ltx-2-mlx's own way, the deltas
fused into the weights and those quantized again, is shown next to it: 0.6% from the exact sum
with the 8-bit pack, 24% with the 4-bit one, where quantizing again moves every adapted weight.
Through the protocol, a 2.5 clip with a LoRA is the same bit for bit with and without Save memory,
also when the transformer is loaded again for it (the LoRA goes back on with it). The LTX-2.3
fixtures were made again then with the 8-bit pack, the 4-bit one being no longer on this Mac.

The full pipeline (`quality` in a request, the app's *Quality*: `TI2VidTwoStagesPipeline`,
`LTXGuidance.swift`). Stage 1 runs the pack's `transformer-dev.safetensors`, which the app
downloads only for it, at half resolution on `ltx2_schedule` (30 steps by default, shifted for
the stage's token count and stretched to end at 0.1), each step four passes: the prompt, the
negative prompt (the request's, or the reference's long default list), STG (the self-attention of
block 28 replaced by its values, video and audio) and the modalities isolated (the audio–video
cross-attention left out of every block); the reference multiplies what it leaves out by zero, the
port skips it, the same numbers. The guiders combine them, `cond + (cfg − 1)(cond − uncond) +
stg(cond − perturbed) + (modality − 1)(cond − isolated)` rescaled 70% towards the conditioned
prediction's spread, with CFG 3 for the video (the request's guidance) and 7 for the sound, STG 1
and modality 3 for both. Stage 2 is the distilled pipeline's on the distilled transformer, which
the reference uses in its low-memory mode in place of the dev one with the distilled LoRA (the
same weights at the LoRA's strength 1); one transformer is in memory at a time, swapped in a few
seconds. The sound decoded is stage 1's, as the reference keeps it. `make_ltx_fixture.py --quality`
runs the reference at the fixture's size with four guided steps: the schedule is the same doubles,
every step's four passes on the reference's own inputs land within bfloat16's last digits (at most
0.2% of their scale), the guiders' combination of the reference's passes is bit for bit, and the
whole guided stage from the reference's noise ends 3.6% apart (bfloat16 drift, which guidance
amplifies). A 2 s 768 × 512 clip on the 2.5 4-bit pack took 663 s of denoising against 79 s fast,
peaking at 30.9 GB both ways without Save memory (13 GB with it), and the reference's own run of
the same request is the same clip: 27.8 dB per frame on average through H.264, the same loudness
(a distilled clip lands at 32.6: thirty guided steps carry more bfloat16 drift, which guidance
amplifies).

The last frame (31 October 2026, `last_image` in a request, beside `image` or alone): the
reference binds an image at any frame but the first as a keyframe, `VideoConditionByKeyframeIndex`
(`generate --image end.png <last frame> 1.0` in ltx-2-mlx). Its tokens, encoded like the first
frame's, are appended after the clip's in both stages, preserved by the mask (timestep 0, no
keyframe marker, the attention unmasked), at positions of their own: the cells' pixel centers,
and as time the middle of that single pixel frame, not of the eight a latent frame covers. Stage 1
leaves them behind before the upsampler, stage 2 before the decoder; the Quality pipeline does
the same, its schedule shifted for the clip's tokens only. `--last-image` records them: with and
without a first image (`ltx25-kf`, `ltx25-last`, the reference's own hedgehog pair), the noise,
the conditioned states of both stages, their masks and positions are identical, the image tokens
match in float32, and every transformer pass is within 2e-4 with the appended tokens. A 4-second
768 × 512 clip from both images (the 2.5 4-bit pack, seed 5) took 201 s, 177 of them denoising,
peaking at 33 GB without Save memory; its last frame is 25.5 dB from the end image fitted to the
clip (the first frame, pinned, 28.5 dB from its own through the same measure), about the gap
ltx-2-mlx reports between its first and last frames: past the first, an image is a strong hint,
not a pin.

ltx-2-mlx 0.16.1 (31 October 2026, the revision now pinned) brings two of the reference's fixes,
both following Lightricks' LTX-2. LTX-2.5's distilled stage 2 runs the ancestral sampler too (as
upstream since v1.4.0), on noise of its own (seed + 20000; stage 1's is seed + 10000, the
renoising before stage 2 still seed + 2); the Quality pipeline's stage 2 stays on the Euler loop,
as the reference's. And a conditioning picture is prepared as `load_image_and_preprocess`
prepares it: turned by its EXIF orientation (3, 6 and 8, the ones the reference applies), in
sRGB, cropped to even sizes as the H.264 round trip begins (the round trip itself, at CRF 18 for
2.5 and 33 for 2.3, is still skipped), scaled to cover the stage's size with PyTorch's bilinear
interpolation on floats (half-pixel centers, no antialiasing) instead of a Lanczos filter on
bytes, and normalized unquantized (x / 127.5 − 1). `verify` now compares each stage's pixels made
from the fixture's own picture: the hedgehog pair's bit for bit, and those of a Display P3
picture of odd size stored sideways (EXIF 6, `ltx25-i2v`) within one level of 255, which is how
far ColorSync's and LittleCMS's P3-to-sRGB conversions fall apart (RMS 0.003). It also checks
every step's update on the reference's own pass (its input, velocity and noise draws) against the
reference's next state: bit for bit, both stages. The ancestral stage 2 carries the passes'
rounding much further than the Euler loop did (a nudge of one unit in the last place of its input
moves its output by up to 28%, the Euler loop's by 9.5%): the whole stage-2 loop, 0.5% from the
reference before, ends 4.6% from it on `ltx25`, so it is shown and does not decide, as stage 1's.
The four 2.5 fixtures pass. Through the protocol, 768 × 512 × 49 frames, seed 7, the 4-bit pack
without Save memory: from the prompt, 29.0 dB per frame on average through H.264 against the
reference's clip, the same loudness, in 84 s; from a picture (P3, EXIF 6), the first frame is
40.3 dB from the reference's before either is encoded, and the clip 26.8 dB on average through
H.264 (34.9 on the first frame, 23.3 on the last eight, where the two ancestral paths part), the
same loudness, in 98 s, peaking at 31.4 GB. With a picture of odd size, ltx-2-mlx's `--image
picture 0 1.0 0` (CRF 0) also skips the crop to even, which scales its picture from one more row
and column: the same comparison then falls to 20.9 dB.

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

# 1. A tiny checkpoint with random weights, and what mflux computes from it (only when a
#    reference changes: the fixtures are files, kept in build/fixtures, which git ignores)
$PY Engine/Fixtures/make_klein_fixture.py      build/fixtures/klein
$PY Engine/Fixtures/make_flux3_fixture.py      build/fixtures/flux3
$PY Engine/Fixtures/make_zimage_fixture.py     build/fixtures/zimage
$PY Engine/Fixtures/make_qwen_image_fixture.py build/fixtures/qwen-image
$PY Engine/Fixtures/make_qwen_image_edit_fixture.py build/fixtures/qwen-image-edit
$PY Engine/Fixtures/make_ming_fixture.py       build/fixtures/ming
$PY Engine/Fixtures/make_seedvr2_fixture.py    build/fixtures/seedvr2
#    random LoRA files for three of them, with the pass mflux computes with each applied
$PY Engine/Fixtures/make_lora_fixture.py build/fixtures/zimage build/fixtures/klein build/fixtures/flux3 build/fixtures/qwen-image

# 2. The same computations in Swift, no Python: one fixture (it names its family) or all of them
build/bin/turbo-engine verify build/fixtures/klein
build/bin/turbo-engine verify --all                        # every fixture in build/fixtures
```

`verify` reports, stage by stage, the largest difference relative to the reference's scale (and the
RMS one): the text side, the initial noise for the fixture's seed, one transformer pass (for Ming
also the unconditional one), the schedule, the whole denoising loop (guided where the family uses
guidance), and the decode; for Klein also an edit from a reference picture (the VAE encoder, the
picture's tokens and ids, one pass and the loop with them, its decode, and the sizes pictures of
several shapes are encoded at); for FLUX 3 (whose fixture is saved in the layout of BFL's own
repository and read back through mflux's loader, quantized to 8 bits on loading as the catalog
runs it) the padding's rotary positions, the encoder in float32, one pass and the Base's guided
pass (both bit-identical), the loop, the decode, two pictures prepared as several (byte for byte)
with their tokens, a pass and the loop, one picture padded by reflection and the box it covers,
and the sizes pictures of many shapes are prepared at; for Qwen-Image-Edit the picture's resizes byte for byte, its
patches, the vision tower, the prompt with the picture's tokens, the picture's latents, a pass (and
the picture's own tokens out of it, which `zero_cond_t` modulates at timestep 0) and the loop over
the image and the picture; for SeedVR2 (whose fixture is saved in the original
checkpoint's format and read back through mflux's own mapping first) the picture's preparation
byte for byte with and without softness, the tiled encode, the noise, the transformer's input and
one pass (bit-identical), the step, the tiled decode, the color correction, and the whole upscale
from the picture, shown with its PSNR (about 49 dB: the random weights and the histogram match
spread the VAE's bf16 rounding over a few pixels). Where `make_lora_fixture.py` has run, Z-Image,
Klein, FLUX 3 and Qwen-Image also put each of its LoRA files on the transformer for the pass (Z-Image's in
ai-toolkit's and Kohya's spellings, Klein's in diffusers' and BFL's, with the fused `qkv`, FLUX 3's
in diffusers' and BFL's, with the swapped final modulation,
Qwen-Image's in ai-toolkit's and Kohya's; some with an alpha), against mflux's pass with the same
file, then take them off, which must give the plain pass back bit for bit. Anything above 3% fails. Identical math lands well below that; a
wrong reshape, a swapped rotary pair or a missing cast shows up as a large error at the first stage
it touches.

Two things to know when reading the numbers. Klein's text encoder is also run with float32
activations, and that check decides for it: in bf16 its layers carry the rounding of MLX's
kernels, which the fixture's random weights amplify (shown, marked "·"). They match exactly while
mlx-swift carries the MLX of mflux's venv (0.32.2 on both sides since 29 September 2026; with
mlx-swift 0.31.6 they were 3–4% apart), and drift again when the two versions part. Ming's router
picks experts from bf16 scores, so a rounding difference between two MLX versions can flip a
choice; the fixture's random weights make that unlikely, and it would show as a large error on the
caption features alone. Ming's text side and loop now match mflux exactly too; its decode, whose
start runs in bf16 as in mflux, lands at 2.4% on the fixture where the other families' stages stay
under 1%.

The fixture generators are ordinary mflux code and run wherever mflux imports (they were exercised
on a Linux CPU build of MLX while the ports were written); `verify` needs the Mac.

Then the real thing, which the fixtures cannot stand in for: they are saved straight from mflux's
modules, and a published checkpoint can store a tensor in another shape (the Qwen VAE's norms
are flat in the catalog's checkpoints, for one). Start `build/bin/turbo-engine serve` and
`$PY Engine/Reference/turbo_worker.py serve`, send both the same `generate` line with the same
checkpoint, prompt, seed and size, and compare the two PNGs with `build/bin/turbo-engine compare
a.png b.png` (the PSNR and the largest difference; two clips frame by frame, with their sound):
the same image, with small local differences from bf16 rounding. On 29 September 2026, with the same MLX on both sides, at
512 × 512: Klein 39 dB, Z-Image 39 dB (4-bit) and 44 dB (8-bit), Qwen-Image 36 dB, Ming 33 dB,
Qwen-Image Edit 60 dB (27–35 dB before, with mlx-swift 0.31.6). On 16 October 2026, with the
checkpoints mflux-community saved again with mflux 0.22, which carry the output norm's bias that
mflux 0.16 to 0.21 dropped (mflux PR 839; the engine takes it, and puts zeros in place of it in the
older checkpoints, which generate as before): Qwen-Image 2512 40.9 dB at 512 × 512 in 20 steps,
Qwen-Image Edit 53.8 dB, both against mflux 0.22.0 on MLX 0.32.2. The same day the editor took
`zero_cond_t` (mflux PR 844: Edit 2511 modulates the picture's tokens at timestep 0, which
mflux's 2509 configuration, the one the port first followed, does not): the same edit lands
37.3 dB from mflux run as 2511, in about the same time, and does what the prompt says where it
was half done before (a sign rewritten whole, not its new words squeezed under the old ones;
13.4 dB from the edit without it). The reference worker takes
`params.loras` too, applied unbaked as the engine applies them: on 9 October 2026, Z-Image 8-bit at
768 × 768 with a LoRA of ai-toolkit's (240 layers) lands 36.9 dB from mflux, 38.5 dB without it.

A Klein edit (`params.image`, the reference picture's path) is compared the same way against
mflux's `Flux2KleinEdit`, which the Python worker does not run. Give both an sRGB picture: the
engine draws the reference into sRGB, while mflux takes a Display P3 file's values as they are.
At 1024 × 640 the two land 25 dB apart. Against mflux run in float32, the engine is 2 dB behind
mflux in both cases: an edit 25 dB (mflux 27), text-to-image 28.5 dB (mflux 30). An edit carries
more of the bf16 rounding in both engines, and the engine's encoder is no worse than mflux's:
both are 3% (RMS) from a float32 encode.

FLUX 3 Image Klein runs through the reference worker too (`flux3-klein`, with the spec's
`quantize`: BFL's bf16 repository read and quantized to 8 bits by both engines). On 18 October
2026, against mflux 0.23.0 on MLX 0.32.2, 16 steps: 39.5 dB at 512 × 512, 37.0 dB at 1024 × 1024;
an edit of a 600 × 450 picture 41.5 dB at 768 × 576 (the picture padded by reflection) and 44.9 dB
at its own size (made on the padded canvas and cropped back, as mflux does without a size); a
diffusers LoRA 42.8 dB. On the real weights the passes are not bit for bit as on the fixture:
the activations reach millions (the single stream about 10⁷), where a bf16 step is thousands, and
the two builds of MLX round a few values apart from the first image block on (1–2% RMS by the end
of a pass, which few steps amplify: 27–34 dB at 1 to 8 steps, more again at 16). Against mflux
with every parameter and activation in float32, mflux's own image is 40.6 dB away at 512 px and
the native engine's 38.2 dB, the 2 dB Klein shows too; with an ai-toolkit LoRA in BFL's names,
which makes the model touchier, 29.6 and 30.9 dB (the two engines 26.9 dB apart, the same image).
Those numbers decode as mflux does, from the loop's float32 state, which keeps FLUX.2's whole decoder
in float32 (its residual stream): 43–49 GB at 2048 × 2048. The engine rounds the latents to bf16
first, as Klein's are: 31 GB there, 22.7 instead of 33.6 at 1536, a quarter faster, 47–48 dB from the
float32 decode and 0.1–0.3 dB further from mflux (39.2 and 36.9 dB at 512 and 1024). Chunking the
VAE's mid-block attention (512 channels, which MLX's fused kernels do not take) changed neither
peak: the convolutions at full size set it, not the attention.

SeedVR2 (`params.image` and `params.upscale`, the factor, with `params.softness`) is compared
against mflux's `SeedVR2(model_path=…).generate_image(resolution=ScaleFactor(…))`, which the
Python worker does not run either, on the original checkpoint's snapshot: 688 × 384 to
1376 × 768 lands 56.7 dB from mflux (at most 7 levels apart), 672 × 880 to 1344 × 1760 62 dB
(at most 4), in 18 s and 34 s against mflux's 37 s and 51 s, peaking at 10.4 GB against 18 GB.

## Checking SenseNova-U1.5 against SenseTime's code

mflux does not run SenseNova-U1.5: its reference is SenseTime's own PyTorch code
([SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1), branch `feat/u1.5`, the commit in
`Fixtures/requirements-sensenova.txt`), in a venv of its own. The fixture is a small model with
random weights (2 layers of 256, the original's structure) saved in the MLX pack's layout, and what
the reference's unchanged `t2i_generate` computes from it in float32, recorded by wrapping its
methods: the prompt caches layer by layer, the schedule, the first step's input and both
velocities, and the whole loop with guidance 2 and without. The port matches all of it within
1e-5.

```bash
build/sensenova-venv/bin/python Engine/Fixtures/make_sensenova_fixture.py <SenseNova-U1 clone> build/fixtures/sensenova
build/bin/turbo-engine verify build/fixtures/sensenova
```

For real images, `Reference/sensenova_reference.py` dequantizes an MLX pack into bf16, puts it back
in the original's layout and runs the reference on MPS (about 40 GB of memory) from the engine's
own noise for the seed; `--record <dir>` also writes the stages of that run, which `verify` then
compares with the pack's weights (`--perturb` shows how far the reference moves by itself). On 30
September 2026, with the 4-bit pack at 8 steps: 37.3 dB at 512 × 512 and 29.4 dB at 1024 × 1024,
while every step's velocity, from the reference's own image, is within 0.2–0.5% (RMS) of it and the
prompt's cache within 1–3% at the deepest layers (bf16 rounding over 42 layers, the same with
the dequantized weights). The distilled model is that sensitive by itself: noise 0.2% off moves
the reference's own image to 32.9 dB.

The fixture also edits a picture (`picture.png`, 300 × 200) through `it2i_generate`, prepared as the
engine prepares it: the picture's bytes after the resize (identical), its normalized pixels, its
patch embedding, the query's positions, both prefixes, the first step's velocities and the loop,
all within 1e-5. With the real pack, an edit of a 1024 × 1024 image ("make it winter") lands
36.7 dB from the reference (43.7 dB at 512 px): the picture holds the two runs together.

## Checking MOSS-Image against OpenMOSS's pipeline

MOSS-Image has no MLX reference either: its reference is OpenMOSS's own diffusers pipeline,
`pipeline_moss_image.py` in [OpenMOSS-Team/MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)
(the revision in `Fixtures/requirements-moss.txt`), with transformers' `Qwen3_5Model` for the text
encoder, in a venv of its own. The fixture is a small checkpoint with random weights in the
repositories' layout (a Qwen3.5 of 8 layers with a vision tower nobody reads, a DiT of 24 blocks so
that block 22 exists to be skipped, FLUX.2's VAE with the small decoder at its real size), and what
the pipeline computes from it in float32: the encoder layer by layer for prompts shorter and longer
than the delta rule's 64-token chunk, the snapped schedule for 1 to 100 steps (bit for bit), the
rotary table, one forward plain, with a long prompt and with block 22 skipped, the loop with
guidance and without, and the decode. The port matches all of it within 1e-5 but the decode, which
runs in bf16 here: a random decoder turns rounding into a percent or two of the image, so it passes
when it is no further from the float32 image than twice the reference's own bf16 decode.

```bash
build/moss-venv/bin/python Engine/Fixtures/make_moss_image_fixture.py <pipeline_moss_image.py> build/fixtures/moss-image
build/bin/turbo-engine verify build/fixtures/moss-image
```

For real images, `Reference/moss_image_reference.py` runs the pipeline as OpenMOSS ships it (bf16,
the VAE in float32) on MPS from the engine's own noise for the seed; `--record <dir>` writes the
token ids, the prompts' states, the first step's velocities, the final latents and the image, which
`verify` compares with the checkpoint's own weights (`"quantize": 8` in its `fixture.json` loads
them as the app does), and `--truth <dir> <snapshot>` adds Qwen3.5's states in float32. On 18
October 2026, 512 × 512, 50 steps, guidance 3: the tokens identical; the image 31.6 dB from the
reference's in bf16, 30.4 dB in 8 bits; from the reference's own prompt states 43 dB, and its
latents decoded here 52 dB from its float32 decode. Both bf16 encoders sit 1.2% (RMS) from the
float32 one, which is what moves the prompt's states 1.6% apart.

## Checking Idol against idol-sdk

Idol's reference is Sperid Labs' own code, [idol-sdk](https://github.com/speridlabs/idol-sdk) (Apache 2.0), run on
MPS with the weights of the MLX pack the app downloads (`ddalcu/Idol-MLX-Serve-8bit`, mlx-serve's conversion of the
release), its 8-bit linears dequantized, DINOv3 and the VAE's encoder from the pack's own folders, so nothing gated
is fetched. `Reference/idol_reference.py` runs it stage by stage and writes the GLB and the PLY; `--record <dir>`
writes what every stage computed, from the matte's picture to both decoders on fixed inputs; `--truth <dir>` adds the
16-bit stages again in float32 and the reference's own float16 loop run another way; `--latent <file>` decodes a
latent denoised elsewhere with idol-sdk's decoders.

```bash
$PY -I Engine/Reference/idol_reference.py --sdk <idol-sdk> --pack <snapshot> --image fox.jpg --out <dir> --record build/fixtures/idol
$PY -I Engine/Reference/idol_reference.py --truth build/fixtures/idol --sdk <idol-sdk>
build/bin/turbo-engine verify build/fixtures/idol
build/bin/turbo-engine bench-idol build/fixtures/idol [--collapse]            # the denoiser alone, step by step
build/bin/turbo-engine bench-idol --generate <snapshot> --out <dir> [--exact build/fixtures/idol] \
    can=can.jpg:synthetic:50 glasses=glasses.jpg:real:50                       # whole objects through the engine
```

On 1 November 2026 (an M1 Max, the 8-bit pack): BiRefNet's alpha map within 2e-5 of the reference's, its bytes off by
one in 0.17% of them; the crop to 1024 on black from the reference's alpha byte for byte (Pillow's Lanczos on
premultiplied RGBA, its MinFilter and paste); DINOv3 in bf16 as close to its float32 run as the reference's (1.2% RMS
against 1.1%); the VAE's moments in bf16 at cos 0.9998 to float32 (the reference's 0.99997), far below the scatter of
the posterior's sample (60% of its mean), the sample's arithmetic exact; the octree's logits within 0.3%; the Gaussian
decoder as close to float32 as the reference (0.27% against 0.29%), its activations exact; one guided velocity at cos
0.99998. A guided loop in float16 carries rounding a long way (four coarse steps land ~8% from float32, in both
engines), so the short loop is decided against float32 too; from the reference's own noise the whole pipeline makes
the reference's object (disc centres 0.4% of its size apart, the same colours). The engine draws its own noise (MLX's
generator, not PyTorch's), so a seed gives another object of the same picture. Through the protocol a 262,144-Gaussian
object takes 300 s at 50 steps (5.9 s a guided step) and 153 s at 25, peaking at 8.3 GB (6.8 with Save memory),
against 530–607 s for the reference; picture, encoders and decoders take 6 s of it.

## Checking MiniMax-Music3 against mlx-audio

MiniMax-Music3's reference is mlx-audio's MLX port (`mlx_audio/music/models/minimax_music3`, the release on PyPI as
0.6.3, the same files as its main branch on 1 November 2026), which follows MiniMax's own integration in diffusers
(`modular_pipelines/minimax_music3`); the prompt's cleaning, where mlx-audio differs from it in edge cases, follows
diffusers', whose functions `Reference/music3_reference.py` copies. It runs on the pack the app downloads
(`mlx-community/MiniMax-Music3-8bit`). `--record <dir>` writes the prompt and its ids, the first frames of the
autoregressive stage step by step (keys, guided logits, codes, depth hidden states, feedback), every frame from the
seed, and the flow on them (condition, noise, both velocities at one time, a window's 30 steps, the vocoder, two
windows with their overlap); `--truth <dir>` the same stages in float32 from the recording's inputs; `--prompts <dir>`
captions and lyrics as the official functions make them.

```bash
build/music3-venv/bin/python -I Engine/Reference/music3_reference.py <snapshot> --record build/fixtures/minimax-music3 --prompts build/fixtures/minimax-music3
build/music3-venv/bin/python -I Engine/Reference/music3_reference.py <snapshot> --truth build/fixtures/minimax-music3
build/bin/turbo-engine verify build/fixtures/minimax-music3
```

On 1 November 2026 (an M1 Max): the prompt text and ids identical, 35 tricky captions and lyrics as the official
functions make them, the tokenizer identical to `transformers` on the corpus; the guided semantic logits, the
feedback, the condition encoder and the noise from seed + 7 exactly the reference's; everything else as close to
float32 as mlx-audio's own run, or closer (the prompt's last hidden states 1.07% from float32 against 1.09%, the depth
decoder 0.5–0.7% against 0.4–0.6%, the flow's velocities 4.3–5.0% both, a window's latents after 30 steps 2.7%
against 3.3%, the vocoder 1.3% both, two windows of four steps 10.3% against 9.8%). The two builds of MLX (PyPI's
0.32.4 and mlx-swift's) round bf16 apart by an ulp here and there, and a near tie then draws another code: a seed
gives another song than mlx-audio's from the same prompt, the frames identical only while the draws agree. Through
the protocol 6 s took 53 s on both (composing 16.9 s here against 17.6 s, the flow 35 s), 10 s 116 s, 60 s 836 s
(composing 171 s, 114 ms a frame; 420 flow steps on 14 windows 651 s, 1.55 s each; the vocoder 13 s), peaking at
17.8–18.8 GB, 14.4 GB with Save memory, which lets the language models go once the song is composed.

The 4-bit pack (`mlx-community/MiniMax-Music3-4bit`, the same layout with the linears in 4 bits; 2 November 2026) was
recorded the same way into `build/fixtures/minimax-music3-4bit` and passes against it: the prompt's last hidden states
1.01% from its float32 against mlx-audio's 1.04%, the velocities 5.0% against 5.1%, a window's latents 4.5% against
4.7%, two windows 7.1% against 6.9%. Against the 8-bit pack's float32 it is much further (the prompt's states 13%, the
velocities 23%, a window's latents 17%), what 4 bits cost; the semantic tokens of the first frames are still the
same. 20 s of song took 229 s (compose 51 s, flow 172 s) against 242 s for the 8-bit pack, peaking at 12.7 GB, 9.6 GB
with Save memory (the 8-bit pack 14.5).

## Checking MOSS-SoundEffect against OpenMOSS's code

MOSS-SoundEffect v2.0's reference is OpenMOSS's own pipeline (`moss_soundeffect_v2/` in
[MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), PyTorch), which is not on PyPI; it is not run here. Its DiT, its
loop and its decode are checked against the float32 golden tensors that
[xocialize/moss-soundeffect-mlx](https://github.com/xocialize/moss-soundeffect-mlx) captured from that code on a CPU
(`tests/fixtures/swift_goldens.safetensors` and `golden_meta.json`, commit 4e8e3e3: "a heavy wooden door creaks open
slowly" for 10 s, numpy noise, the velocity at t = 1000 and 500, ten guided steps at scale 4, the decode); their
prompt states match transformers' float32 Qwen3 on the same tokens to 1.5e-6, which vouches for them. The rest comes
from PyPI code on the real weights (`Fixtures/make_moss_soundeffect_fixture.py`): the prompt text and its cleaning
(ftfy, html, regex) and tokens for a corpus (curly quotes, fullwidth punctuation, ligatures, HTML references,
control characters, combining marks, Chinese, a prompt past 512 tokens), Qwen3-1.7B's last hidden state in float32
and in bf16 (transformers), and the schedule (torch). `verify` runs the port in float32 with OpenMOSS's float32 DiT
(`transformer/diffusion_pytorch_model.safetensors` of `OpenMOSS-Team/MOSS-SoundEffect-v2.0`), then the MLX pack's bf16
DiT on the same inputs:

```sh
build/moss-sfx/venv/bin/python -I Engine/Fixtures/make_moss_soundeffect_fixture.py <goldens dir> <mlx pack snapshot> <OpenMOSS snapshot> build/fixtures/moss-soundeffect
build/bin/turbo-engine verify build/fixtures/moss-soundeffect
```

Everything in float32 is the reference's (2 November 2026): the corpus's texts, cleaning and tokens identical, the
schedule exact for 1 to 200 steps, the encoder 1e-6 RMS from transformers, both velocities 1e-6, the ten guided steps
3e-6, the decode exact, and the decode of only the latents a shorter clip needs equal to the whole one's first
samples. In bf16 the encoder lands 2.4–5.7% from float32 against transformers' 2.7–5.1% in bf16, the velocities 0.4
and 0.6%, the ten steps 1.1%, the decoded sound 2.2%.

Through the protocol on an M1 Max, 100 guided steps took 157–164 s for 6, 10 and 30 s of sound alike (0.8 s a DiT
pass over the 1500 latents), the decode 1.1 s for 10 s and 2.5 s for 30 s, reading the prompt 0.1 s. The peaks: 7.4 /
10.2 / 11.9 GB for 1 / 10 / 30 s, the weights 7.0 GB of it (Qwen3 without its tied head 3.2, the DiT 2.6, the VAE
1.4); with Save memory, which lets the text encoder go once the prompts are read, 7.0 GB to 10 s and 8.7 GB for 30 s.
The three sounds (a door creaking open, typing, rain with thunder) look as described in their spectrograms: the
creak's pulses, the keys' even clicks, the rain's broadband hiss under two low rumbles.

## Checking Describe Image against mlx-vlm

Describe Image's reference is mlx-vlm's port of Qwen2.5-VL (`mlx_vlm/models/qwen2_5_vl`, the release on PyPI as
0.7.12: its image processor, its chat template and `generate_step` at temperature 0) on the pack the app downloads
(`mlx-community/Qwen2.5-VL-7B-Instruct-4bit` at fdcc572e). `Reference/qwen25vl_reference.py` reads each picture at the
describer's 256 to 1280 tokens and records the pixel values, the prompt's ids, the vision tower's tokens, the first
step's logits and the tokens written, then the vision tokens and the logits again with the model in float32;
`verify` checks the pixels and the ids exactly, judges the vision tokens and the first logits by their distance from
float32 against mlx-vlm's own, and counts the tokens written as the reference's. Pillow 13 orders its two resize
passes by how much each side shrinks (`PILResample.resizePillow13`). mlx-vlm wants MLX 0.32.4; the check is closest
with the MLX of the engine's mlx-swift (0.32.3, which carries MLX 0.32.2) put in its place:

```bash
uv venv build/vlm-venv && uv pip install --python build/vlm-venv/bin/python mlx-vlm==0.7.12
uv pip install --python build/vlm-venv/bin/python --no-deps mlx==0.32.2 mlx-metal==0.32.2
build/vlm-venv/bin/python -I Engine/Reference/qwen25vl_reference.py <snapshot> build/fixtures/qwen25-vl <picture>...
build/bin/turbo-engine verify build/fixtures/qwen25-vl
```

On 9 November 2026 (an M5 Ultra), with eight PNGs (photos, a watercolour, a poster with text, a 3D render, close-ups,
from 512 × 512 to 1344 × 768): the pixel values and the prompt's ids identical; the vision tokens as far from float32
as mlx-vlm's (1.1–8.8% RMS, both); the first logits 0.6–3.7% from float32 against mlx-vlm's 0.8–3.7% (0.7 to 1.6
times its distance: the language model's norms are Qwen-Image's, which round once where mlx-vlm's fused norm rounds
twice) and the same first token; then 971 of the 1,085 tokens written as the reference's, six texts of eight word for
word, the two others parting where the two candidates' logits are the same in float16. In float32 the port lands
0.06–0.4% from mlx-vlm's float32 run, vision tokens and logits alike. With PyPI's MLX 0.32.4 the reference's float16 is
nearer float32 (its logits 0.3–1.6% away) and the texts still agree, seven of eight. Through the protocol a
description took 1 to 2.5 s (loading the pack from the file cache 0.3 s, the picture and the prompt 0.1–0.5 s,
writing 140 tokens a second; the first after the engine starts compiles its kernels, a few seconds more), peaking at
6.5–7.1 GB; mlx-vlm writes 180 tokens a second.

## Checking the tokenizers

The engine tokenizes on its own (`Tokenizer/`), with no library: each family's prompt pipeline
(Qwen3's chat template with thinking off and on, Qwen3.5's, which trims the prompt as Jinja's `trim`
does and opens a think block, Qwen-Image's and Qwen-Image Edit's templates, Ming's,
Gemma's with `<bos>`, SenseNova's queries, against the tokenizer its code builds from the original's
vocabulary and merges) is checked id for id against `transformers` on a corpus of 64 prompts, from
empty and whitespace-only to Italian, Chinese, Arabic, Thai, emoji sequences, decomposed accents,
special tokens typed by hand and prompts past the encoders' limits:

```bash
build/bin/turbo-engine verify-tokenizers                   # Engine/Fixtures/tokenizers.json
$PY Engine/Fixtures/make_tokenizer_corpus.py Engine/Fixtures/tokenizers.json   # to regenerate it
$PY Engine/Fixtures/make_tokenizer_corpus.py Engine/Fixtures/tokenizers.json --only qwen35-chat-thinking   # one pipeline
```

It reads the catalog's tokenizers from the Hugging Face cache (a pipeline whose checkpoint is not
downloaded is skipped); only regenerating the corpus needs Python. Two things Foundation gets in
the way of, which the tokenizer avoids: Swift's `String` treats canonically equivalent tokens as one
(";" and the Greek question mark, two combining graves: Gemma's vocabulary has both), so tokens are
matched by their bytes; and `JSONSerialization` drops a U+FEFF that opens a string, so
`tokenizer.json` is read by a parser of its own (`TokenizerJSON`).

## Timing the catalog

`turbo-engine bench` runs the catalog's models found in the Hugging Face cache through the same
`Engine` the app talks to, on fixed prompts, seeds and sizes, and prints the seconds per phase and
the peak memory of each as a table (`build/bench/<date>/bench.md`, with `bench.json` beside it
and the images and clips it made):

```bash
build/bin/turbo-engine bench                                  # the rows of docs/swift-engine-plan.md's tables
build/bin/turbo-engine bench --only z-image --sizes 512,1024,2048 --save-memory on   # a memory sweep
build/bin/turbo-engine bench --only z-image --only qwen-image --half                 # the 16-bit option
build/bin/turbo-engine bench --only klein --preview                                   # the live preview's cost
build/bin/turbo-engine bench --only qwen-image-2512 --sizes 1024 --prompt "A sign that reads …"   # small text
```

By default the requests are those of the Status tables in `docs/swift-engine-plan.md`, so a run
compares with them; `--sizes` replaces the text-to-image rows with squares of those sides at each
model's default steps, the sweep `TurboMLX/Models/MemoryEstimate.swift` is fitted on;
`--save-memory on|off` sets Save memory for every row, `--half` asks for the 16-bit stream where a
model has it, `--preview` for the live preview, `--prompt` replaces the text-to-image rows' prompt
(so `compare` can put the two precisions side by side on small text), `--repeat N` keeps the median of N runs
(and the highest peak), `--hub` names another cache, `--out` another folder. A model that is not in
the cache is skipped. The pictures the edits and upscales start from are drawn by the bench. It is
what to run after an mlx-swift update, before the peaks in `MemoryEstimate.swift` are trusted again.

## Protocol

Commands on stdin, one JSON object per line: `generate` (id, model, params), `load` (model),
`describe` (id, model, image), `cancel` (id), `unload`, `shutdown`; `params` may carry `precision` ("bf16"), `preview`
(true), `loras` (`[{"path": …, "scale": 1}]`) and, for LTX-2, `quality` (true: the full pipeline,
`steps` then its first stage's and `guidance` the video's CFG), described under Families and in
Checking LTX-2 against ltx-2-mlx. Idol (`family: "idol"`) reads the picture (`images` or `image`), `num_gaussians`
(32,768 to 1,048,576, a multiple of 32), `mode` ("synthetic" or "real"), `steps` (1 to 100), `guidance` and `seed`, and
refuses anything else by name before loading; it reports the phases `segmenting`, `encoding`, `denoising` (with its
`total`), `decoding` and `saving`, writes the GLB at `output` (whole or not at all), and its `done` gives the discs
written as `gaussians` and the timings `segment`, `encode`, `denoise`, `decode` and `save`, with no size or poster.
MiniMax-Music3 (`family: "minimax-music3"`) reads the description as `prompt`, `lyrics` (section tags on lines of
their own, `[instrumental]` for none), `duration` (the longest the song may last, up to 360 s; the model may end it
sooner), `steps` (1 to 30, the flow's on each window) and `seed`, and refuses anything else by name before loading; it
reports `encoding`, `composing` (`total`: the most frames, 25 a second, `progress` counting them), `denoising` (with
its `total`), `decoding` and `saving`, writes a 16-bit WAV at `output`, and its `done` gives `duration` (seconds),
`sample_rate`, `frames` and the timings `encode`, `compose`, `denoise`, `decode` and `save`, with no size.
MOSS-SoundEffect (`family: "moss-soundeffect"`) reads the description as `prompt`, `duration` (the clip's length, 0.1 to
30 s, in tenths), `steps` (1 to 200), `guidance` (1: no unconditional pass), `negative_prompt` (empty: the reference's
all-zero states) and `seed`, and refuses anything else by name before loading; it reports `encoding`, `denoising`
(with its `total`), `decoding` and `saving`, writes a 16-bit mono WAV at 48 kHz, and its `done` gives `duration`,
`sample_rate`, `frames` (the kept latent frames, 50 a second) and the timings `encode`, `denoise`, `decode` and `save`.
`describe` writes a prompt describing the picture at `image` with Qwen2.5-VL (`model`: family `qwen2.5-vl`, the pack's
path), after whatever command came before it: it reports `loading`, `encoding` (the picture and the prompt read) and
`writing`, sends the description so far in `text` events (id, text) as it grows, and its `done` gives the `text`, the
`tokens` written and the timings `load`, `read` and `write`; `cancel` stops it between two tokens. The describer is
loaded for the description and let go after it; the image model stays loaded unless the two would not fit in the GPU's
working set, when it goes first (`unloaded`).
Events on stdout: `ready`, `phase`, `progress`, `preview` (id,
step, path: a small PNG of the image as it forms, written again at each), `done` (path, seed, size,
seconds, peak_memory, timings per phase), `failed`, `cancelled`, `model_loaded`, `load_failed` (a
`load` that did not), `unloaded`. Everything else goes to stderr, which the app shows as the
engine log.

Every PNG, MP4 and WAV the engine writes carries IPTC's Digital Source Type in XMP (`Provenance.swift`):
`trainedAlgorithmicMedia` for a result from a prompt, `compositeWithTrainedAlgorithmicMedia` for
one made from `params.image`, unless that picture declares `trainedAlgorithmicMedia` itself. A PNG
has it in ImageIO's XMP chunk, beside the parameters (`dc:description`); an MP4 in a top-level
`uuid` box with Adobe's XMP UUID, appended once the writer has finished, where exiftool reads it; a WAV in a `_PMX`
chunk, with the parameters as JSON in its INFO comment (`AudioOutput.swift`).
