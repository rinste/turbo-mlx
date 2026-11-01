#!/usr/bin/env python3
"""speridlabs' Idol (idol-sdk, Apache-2.0: one picture of one object to a 3D Gaussian splat) run on this Mac, on MPS or
the CPU, with its weights read from the MLX pack `ddalcu/Idol-MLX-Serve-8bit`: the reference turbo-engine's `idol`
family is checked against, and a way to judge Idol's quality and time before porting it.

The pack is mlx-serve's conversion of the release (`tests/convert_idol_weights.py` in github.com/ddalcu/mlx-serve),
read backwards here, so neither gated repo idol-sdk asks for is needed (facebook/dinov3-vith16plus-pretrain-lvd1689m,
black-forest-labs/FLUX.2-dev):
  - denoiser, decoder, decoder_gs: the blocks' Linear layers whose input is a multiple of 64 are affine 8-bit, group
    64 (`weight` uint32 packing four bytes, low byte first, `scales` and `biases` float16): dequantized to float32 and
    loaded into idol-sdk's models, which keep their own dtypes (float16 blocks, float32 embeddings and norms);
  - dinov3/: DINOv3 ViT-H+/16 in transformers' layout, its matrices float16, loaded in bfloat16 as idol-sdk runs it;
  - vae/: the FLUX.2 VAE's encoder, quant_conv and batch-norm statistics, its convs channels-last (OHWI, turned back
    to OIHW), `to_out.0` stored as `to_out`, the middle attention's four Linears 8-bit (bfloat16 scales);
  - birefnet: as released (float16, loaded in float32 as idol-sdk does off CUDA).

Then idol-sdk's own code, stage by stage and timed, in its order and with its random draws: BiRefNet's matte and the
crop (`preprocess`), DINOv3's tokens, a SAMPLE of the VAE posterior (`ImageEncoder`), Euler steps with guidance
(`sample_flow`, the conditional and unconditional passes apart, as idol-sdk runs everywhere but CUDA), the octree
descent and the Gaussian decoder (`decode`). As in mlx-serve's own oracle (`tests/dump_idol_fixtures.py`), the RePo
layers give real (cos, sin) pairs, the same rotation as the complex form MPS lacks.

Each picture writes, to --out:
  <name>.glb        the disc mesh mlx-serve's `splat.zig` writes: one primitive (mode 4, POSITION + TEXCOORD_0 only),
                    one opaque quad per Gaussian in the plane of its two longest axes (1.5 standard deviations, at most
                    0.012), its colour one texel of a PNG atlas (NEAREST, clamped), unlit, double-sided, metallic 0,
                    roughness 1, glTF y-up; Gaussians under 5% opacity are left out;
  <name>.ply        the standard 3DGS PLY (idol-sdk's `save_ply`: its +y points down in splat viewers);
  <name>_input.png  the 1024 x 1024 picture the encoders saw (background removed, cropped, on black);
  <name>.json       the settings, the predicted camera, each stage's time and the peak memory.

  $PY -I Engine/Reference/idol_reference.py --sdk <idol-sdk clone> --pack <Idol-MLX-Serve-8bit snapshot> \\
      --image photo.jpg --mode real --out <dir> [--name photo] [--steps 50] [--guidance 3] \\
      [--gaussians 262144] [--seed 42] [--device mps] [--record build/fixtures/idol]

With `--record`, it also writes what every stage computed to a folder `turbo-engine verify` reads with the pack's own
weights (`fixture.json` names the family and the pack): the matte's picture, input, backbone passes, squeezed features,
alpha map and bytes, the prepared picture; DINOv3's embeddings, layers 0 and 15 and last state; the VAE posterior's
moments and the noise of its sample; the conditioning tokens of both encoders, the noise, the conditional and
unconditional velocities at t = 700 with the denoiser's inner states on the way (the time embedding, the refined
streams, the first joint block's rotary table and output, the last block's output), a 4-step guided run from the same
noise, the whole run's final latent and camera and its anchors; and both decoders on fixed inputs drawn on the CPU. The extra passes draw nothing at random, so the run's
Gaussians are the same with and without it. `$PY -I Engine/Reference/idol_reference.py --truth <record> --sdk <sdk>`
then adds the same velocities and short run with the denoiser in float32, and the short run again in float16 with the
guidance passes batched: what float16 rounding alone moves the reference by. And `--latent <file> --sdk <sdk> --pack
<pack> --out <dir> --name <name>` decodes a latent denoised elsewhere (turbo-engine's `bench-idol --save`) with
idol-sdk's decoders into the same GLB and PLY.

Needs torch, torchvision, transformers >= 4.56, diffusers >= 0.36 (AutoencoderKLFlux2), tensordict, safetensors,
plyfile, pillow, numpy: idol-sdk's requirements.
"""

import argparse
import ctypes
import json
import math
import os
import struct
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

GROUP = 64
SH0 = 0.28209479177387814
# The disc mesh, as mlx-serve's splat.zig makes it.
DISC_MIN_OPACITY = 0.05
DISC_REACH = 1.5
DISC_MAX_REACH = 0.012


# -- The pack, read back into idol-sdk's layout ------------------------------------------------------------------------

def dequantize(packed: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """MLX's affine quantization undone: `packed` [O, I * bits / 32] uint32, `scales` and `biases` [O, I / 64]."""
    assert bits == 8, "the pack is 8-bit"
    q = packed.contiguous().view(torch.uint8).to(torch.float32)          # little-endian: element j is byte j
    rows, columns = q.shape
    q = q.view(rows, columns // GROUP, GROUP)
    w = q * scales.to(torch.float32)[..., None] + biases.to(torch.float32)[..., None]
    return w.view(rows, columns)


def dequantized_state(path: Path) -> dict:
    """A safetensors file of the pack with every quantized Linear made a float32 `weight` again."""
    tensors = load_file(str(path))
    state = {}
    for name, tensor in tensors.items():
        if name.endswith((".scales", ".biases")):
            continue
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        if base is not None and base + ".scales" in tensors:
            state[name] = dequantize(tensor, tensors[base + ".scales"], tensors[base + ".biases"])
        elif tensor.is_floating_point():
            state[name] = tensor.to(torch.float32)
        else:
            state[name] = tensor
    return state


def load_idol_model(model: torch.nn.Module, path: Path, device: torch.device) -> torch.nn.Module:
    model.load_state_dict(dequantized_state(path), strict=True)
    return model.eval().requires_grad_(False).to(device)


def load_dinov3(folder: Path, device: torch.device) -> torch.nn.Module:
    from transformers import AutoConfig, AutoModel

    model = AutoModel.from_config(AutoConfig.from_pretrained(folder))
    expected = set(model.state_dict())
    state = {}
    for name, t in load_file(str(folder / "model.safetensors")).items():
        # transformers 5 holds the blocks under `model.layer.N`, the checkpoint (4.56's layout) under `layer.N`
        if name not in expected and f"model.{name}" in expected:
            name = f"model.{name}"
        state[name] = t.to(torch.float32) if t.is_floating_point() else t
    model.load_state_dict(state, strict=True)
    return model.to(torch.bfloat16).eval().requires_grad_(False).to(device)


def load_flux2_vae(folder: Path, device: torch.device) -> torch.nn.Module:
    """The VAE whole, its encoder from the pack (the decoder, which Idol never runs, stays as initialised)."""
    from diffusers.models import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_config(AutoencoderKLFlux2.load_config(folder))
    state = {}
    for name, tensor in dequantized_state(folder / "model.safetensors").items():
        if ".mid_block.attentions." in name and ".to_out." in name:
            name = name.replace(".to_out.", ".to_out.0.")
        if name.endswith(".weight") and tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2).contiguous()             # OHWI -> OIHW
        state[name] = tensor
    missing, unexpected = vae.load_state_dict(state, strict=False)
    stray = [k for k in missing if not k.startswith(("decoder.", "post_quant_conv.")) and k != "bn.num_batches_tracked"]
    if stray or unexpected:
        raise RuntimeError(f"the VAE in the pack does not map back: missing {stray[:8]}, unexpected {unexpected[:8]}")
    return vae.to(torch.bfloat16).eval().requires_grad_(False).to(device)


# -- Time and memory ---------------------------------------------------------------------------------------------------

def lifetime_peak_footprint() -> int:
    """The process's highest physical footprint so far (rusage_info_v4.ri_lifetime_max_phys_footprint): what
    `/usr/bin/time -l` reports as the peak memory footprint, Metal buffers included."""
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    info = (ctypes.c_uint64 * 40)()
    if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:
        return 0
    return int(info[30])


class Stages:
    def __init__(self, device: torch.device):
        self.device = device
        self.seconds = {}
        self.peak_after = {}
        self.metal_after = {}

    def run(self, name, work):
        self.sync()
        start = time.perf_counter()
        result = work()
        self.sync()
        self.seconds[name] = round(time.perf_counter() - start, 2)
        self.peak_after[name] = round(lifetime_peak_footprint() / 2 ** 30, 2)
        if self.device.type == "mps":
            self.metal_after[name] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 2)
        print(f"  {name}: {self.seconds[name]:.2f} s, peak footprint {self.peak_after[name]:.2f} GB", flush=True)
        return result

    def sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()


# -- The disc mesh -----------------------------------------------------------------------------------------------------

def gltf_up(points: np.ndarray) -> np.ndarray:
    """The model's z-up to glTF's y-up: (x, y, z) -> (x, z, -y)."""
    return np.stack([points[..., 0], points[..., 2], -points[..., 1]], axis=-1)


def write_disc_glb(path: Path, xyz, features_dc, scaling, rotation, opacity) -> dict:
    """splat.zig's `writeDiscGlb`: one quad per Gaussian of at least 5% opacity, its colour a texel of an atlas the
    four corners all point at (viewers ignore per-vertex colour)."""
    opacity = opacity.reshape(-1)
    kept = np.nonzero(opacity >= DISC_MIN_OPACITY)[0]
    n = len(kept)
    if n == 0:
        raise ValueError("no Gaussian is opaque enough for the disc mesh")
    side = math.isqrt(n - 1) + 1
    centre = xyz[kept].astype(np.float32)
    w, x, y, z = (rotation[kept, i].astype(np.float32) for i in range(4))
    # the rotation's columns: the Gaussian's three axes in the model's frame
    axes = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)], axis=-1),
        np.stack([2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)], axis=-1),
        np.stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)                                                             # [n, axis, xyz]
    scale = scaling[kept].astype(np.float32)
    order = np.argsort(-scale, axis=1, kind="stable")                     # longest first
    rows = np.arange(n)
    reach_a = np.minimum(DISC_REACH * scale[rows, order[:, 0]], DISC_MAX_REACH)[:, None]
    reach_b = np.minimum(DISC_REACH * scale[rows, order[:, 1]], DISC_MAX_REACH)[:, None]
    axis_a, axis_b = axes[rows, order[:, 0]], axes[rows, order[:, 1]]
    corners = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
    positions = (centre[:, None, :] + corners[None, :, 0:1] * (reach_a * axis_a)[:, None, :]
                 + corners[None, :, 1:2] * (reach_b * axis_b)[:, None, :])
    positions = gltf_up(positions).astype(np.float32).reshape(-1, 3)

    k = np.arange(n)
    texel = np.stack([(k % side + 0.5) / side, (k // side + 0.5) / side], axis=-1).astype(np.float32)
    uv = np.repeat(texel, 4, axis=0)
    base = (k * 4).astype(np.uint32)[:, None]
    indices = (base + np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)[None, :]).reshape(-1)

    colour = np.clip(0.5 + SH0 * features_dc.reshape(-1, 3)[kept], 0, 1)
    atlas = np.zeros((side * side, 3), dtype=np.uint8)
    atlas[:n] = np.round(colour * 255).astype(np.uint8)
    png = BytesIO()
    Image.fromarray(atlas.reshape(side, side, 3), "RGB").save(png, format="PNG")

    blobs = [positions.tobytes(), uv.tobytes(), indices.tobytes(), png.getvalue()]
    binary, views = bytearray(), []
    for blob in blobs:
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob
    binary += b"\0" * (-len(binary) % 4)
    gltf = {
        "asset": {"version": "2.0", "generator": "Turbo MLX idol_reference.py"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"mode": 4, "attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2,
                                    "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0,
                                                "roughnessFactor": 1},
                       "doubleSided": True, "extensions": {"KHR_materials_unlit": {}}}],
        "textures": [{"source": 0, "sampler": 0}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": n * 4, "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": n * 4, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": n * 6, "type": "SCALAR"},
        ],
    }
    text = json.dumps(gltf, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    total = 12 + 8 + len(text) + 8 + len(binary)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(text), 0x4E4F534A) + text)
        f.write(struct.pack("<II", len(binary), 0x004E4942) + bytes(binary))
    return {"discs": n, "atlas": side, "bytes": total}


# -- The recording -----------------------------------------------------------------------------------------------------

RECORD_T = 700.0
RECORD_SHORT_STEPS = 4


def capture_stages(pipeline, captured: dict):
    """`--record`: hooks that keep what the matte and the encoders compute on the way, without changing it. The
    matte: the picture BiRefNet is given (bytes), its normalized input, both backbone passes, the squeezed features,
    the alpha map and the alpha bytes; DINOv3: its embeddings, layers 0 and 15, its last hidden state; the VAE: the
    posterior's moments and the noise its sample draws."""
    import diffusers.models.autoencoders.vae as vae_module

    birefnet = pipeline.background_remover
    backbone_calls = []

    def keep(name):
        def hook(module, inputs, output):
            captured[name] = output
        return hook

    forward = birefnet.forward          # remove_background calls forward itself, so hooks would not see it

    def recorded_forward(x):
        captured["matte_input"] = x
        captured["matte_alpha_map"] = forward(x)
        return captured["matte_alpha_map"]

    birefnet.forward = recorded_forward
    birefnet.bb.register_forward_hook(lambda module, inputs, output: backbone_calls.append(output))
    birefnet.squeeze_module.register_forward_hook(keep("matte_squeezed"))
    remove_background = birefnet.remove_background

    def recorded_remove_background(image):
        captured["matte_rgb"] = torch.from_numpy(np.array(image))
        out = remove_background(image)
        captured["matte_alpha"] = torch.from_numpy(np.array(out.getchannel(3)))
        for index, name in ((0, "matte_bb"), (1, "matte_bb_half")):
            if index < len(backbone_calls):
                for level, feature in enumerate(backbone_calls[index]):
                    captured[f"{name}{level}"] = feature.half()
        return out

    birefnet.remove_background = recorded_remove_background
    dinov3 = pipeline.image_encoder.dinov3
    dinov3.embeddings.register_forward_hook(keep("dinov3_embeddings"))
    dinov3.model.layer[0].register_forward_hook(keep("dinov3_layer0"))
    dinov3.model.layer[15].register_forward_hook(keep("dinov3_layer15"))
    dinov3.register_forward_hook(lambda module, inputs, output: captured.__setitem__("dinov3_hidden", output.last_hidden_state))
    pipeline.image_encoder.flux2_vae.quant_conv.register_forward_hook(keep("vae_moments"))
    randn_tensor = vae_module.randn_tensor

    def recorded_randn_tensor(*args, **kwargs):
        noise = randn_tensor(*args, **kwargs)
        captured["vae_eps"] = noise
        return noise

    vae_module.randn_tensor = recorded_randn_tensor


def decoder_references(pipeline, latent) -> dict:
    """The two decoders on fixed inputs drawn on the CPU (their sampling draws on the device's generator, which no port
    can follow): the octree decoder's logits for 256 random cells at levels 1 to 8 and the Gaussian decoder's raw
    outputs and activated Gaussians for 1,024 random anchors, both on the run's final latent."""
    refs = {}
    generator = torch.Generator().manual_seed(1)
    device = latent.device
    for level in (1, 5, 8):
        res = 1 << (level - 1)
        cells = torch.randint(0, res, (1, 256, 3), generator=generator)
        centres = ((cells.float() + 0.5) / res).to(device)
        refs[f"oct_centres_{level}"] = centres
        refs[f"oct_logits_{level}"] = pipeline.decoder(centres, torch.full((1,), 1 << level, dtype=torch.long, device=device), latent)
    points = torch.rand(1, 1024, 3, generator=generator).to(device)
    raw = pipeline.decoder_gs(points, latent)
    gaussian = pipeline.decoder_gs.to_representation(points, raw)[0]
    refs.update({"gs_points": points, "gs_raw": raw, "gs_xyz": gaussian.xyz, "gs_color": gaussian.features_dc.flatten(1),
                 "gs_scaling": gaussian.scaling, "gs_rotation": gaussian.rotation, "gs_opacity": gaussian.opacity})
    return refs


def record(folder: Path, args, denoiser, cond, noise, sample, sample_flow, extra=None):
    """`--record`: the denoiser's inputs and outputs for `turbo-engine verify` (see the module's docstring), and with
    `extra` the other stages'."""
    from safetensors.torch import save_file

    refs = {
        "feature1": cond["feature1"], "feature2": cond["feature2"],
        "noise_latent": noise["latent"], "noise_camera": noise["camera"],
        "final_latent": sample["latent"], "final_camera": sample["camera"],
    }
    refs.update(extra or {})
    neg_cond = cond.apply(torch.zeros_like)
    t = torch.tensor([RECORD_T], device=noise["latent"].device, dtype=torch.float32)
    captured = {}

    def keep(name):
        def hook(module, inputs, output):
            captured[name] = output
        return hook

    watched = [
        (denoiser.t_embedder, "t_emb"), (denoiser.noise_refiner[-1], "noise_refined"),
        (denoiser.context_refiner[-1], "context_refined"), (denoiser.cam_refiner, "camera_refined"),
        (denoiser.repo_layers[0], "rope_block0"), (denoiser.blocks[0], "block0"), (denoiser.blocks[-1], "block_last"),
    ]
    with torch.no_grad():
        handles = [module.register_forward_hook(keep(name)) for module, name in watched]
        conditional = denoiser(noise, t, cond)
        for handle in handles:
            handle.remove()
        refs.update(captured)
        unconditional = denoiser(noise, t, neg_cond)
        refs["v_cond_latent"], refs["v_cond_camera"] = conditional["latent"], conditional["camera"]
        refs["v_uncond_latent"], refs["v_uncond_camera"] = unconditional["latent"], unconditional["camera"]
        short = sample_flow(denoiser, noise, cond=cond, neg_cond=neg_cond, steps=RECORD_SHORT_STEPS,
                            rescale_t=args.rescale_t, cfg_strength=args.guidance, verbose=False, batch_cfg=False)
        refs["short_latent"], refs["short_camera"] = short["latent"], short["camera"]
    folder.mkdir(parents=True, exist_ok=True)

    def stored(value):
        value = value.detach()
        return value.cpu().contiguous() if value.dtype in (torch.uint8, torch.float16) else value.float().cpu().contiguous()

    save_file({k: stored(v) for k, v in refs.items()}, str(folder / "references.safetensors"))
    meta = {
        "family": "idol", "pack": str(args.pack.resolve()), "image": str(args.image), "mode": args.mode,
        "seed": args.seed, "steps": args.steps, "guidance": args.guidance, "rescale_t": args.rescale_t,
        "t": RECORD_T, "short_steps": RECORD_SHORT_STEPS, "device": str(noise["latent"].device), "torch": torch.__version__,
    }
    (folder / "fixture.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"recorded {folder}: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in refs.items()), flush=True)


def add_truth(folder: Path, sdk: Path, device: torch.device):
    """`--truth`: what a recording's 16-bit stages are measured against, added to it. The same 4 guided steps with
    the denoiser wholly in float32 (and both velocities at t), where rounding stays at float32's level, and the
    reference again in float16 with its two guidance passes batched, which rounds differently: how far two float16 runs
    of the same model fall apart (`*_f32`, `short_*_batched`). Then DINOv3's tokens and the VAE's moments in float32
    on the recorded picture, and the Gaussian decoder's raw outputs in float32 on the recorded anchors."""
    from safetensors.torch import save_file
    from tensordict import TensorDict

    sys.path.insert(0, str(sdk.resolve()))
    from idol.models import LatentSeqMMFlowModel
    from idol.models.layers import RePo3DRotaryEmbedding
    from idol.sampler import sample_flow

    meta = json.loads((folder / "fixture.json").read_text())
    pack = Path(meta["pack"])
    refs = load_file(str(folder / "references.safetensors"))
    config = json.loads((pack / "config.json").read_text())
    denoiser = load_idol_model(LatentSeqMMFlowModel(**config["denoiser"]["args"]), pack / "denoiser.safetensors", device)
    for module in denoiser.modules():
        if isinstance(module, RePo3DRotaryEmbedding):
            module.real_valued = True
    cond = TensorDict({"feature1": refs["feature1"], "feature2": refs["feature2"]}, batch_size=1).to(device)
    noise = TensorDict({"latent": refs["noise_latent"], "camera": refs["noise_camera"]}, batch_size=1).to(device)
    neg_cond = cond.apply(torch.zeros_like)

    def short(batch_cfg):
        return sample_flow(denoiser, noise, cond=cond, neg_cond=neg_cond, steps=meta["short_steps"],
                           rescale_t=meta["rescale_t"], cfg_strength=meta["guidance"], verbose=False, batch_cfg=batch_cfg)

    with torch.no_grad():
        batched = short(True)
        refs["short_latent_batched"], refs["short_camera_batched"] = batched["latent"], batched["camera"]
        denoiser.float()
        denoiser.dtype = torch.float32
        t = torch.tensor([meta["t"]], device=device, dtype=torch.float32)
        conditional, unconditional = denoiser(noise, t, cond), denoiser(noise, t, neg_cond)
        refs["v_cond_latent_f32"], refs["v_uncond_latent_f32"] = conditional["latent"], unconditional["latent"]
        truth = short(False)
        refs["short_latent_f32"], refs["short_camera_f32"] = truth["latent"], truth["camera"]
        del denoiser
        if "prepared" in refs:
            # The encoders in float32 on the recorded picture: DINOv3's tokens and the VAE's moments.
            from idol.conditioning import ImageEncoder
            pixels = refs["prepared"].permute(2, 0, 1).float().div(255.0)[None].to(device)
            encoder = ImageEncoder.__new__(ImageEncoder)
            encoder.device = device
            encoder.dinov3 = load_dinov3(pack / "dinov3", device).float()
            refs["feature1_f32"] = encoder._dinov3_tokens(pixels)
            del encoder.dinov3
            vae = load_flux2_vae(pack / "vae", device).float()
            refs["vae_moments_f32"] = vae.quant_conv(vae.encoder(pixels * 2 - 1))
            del vae
        if "gs_points" in refs:
            from idol.models import GaussianFixedlenDecoder
            decoder_gs = load_idol_model(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), pack / "decoder_gs.safetensors", device)
            decoder_gs.float()
            decoder_gs.dtype = torch.float32
            refs["gs_raw_f32"] = decoder_gs(refs["gs_points"].to(device), refs["final_latent"].to(device))
    save_file({k: v.detach().float().cpu().contiguous() for k, v in refs.items()}, str(folder / "references.safetensors"))
    print(f"added float32 and batched references to {folder}", flush=True)


# -- The run -----------------------------------------------------------------------------------------------------------

def decode_latent(path: Path, sdk: Path, pack: Path, out: Path, name: str, gaussians: int, seed: int, device: torch.device):
    """`--latent`: a denoised latent decoded by idol-sdk's own decoders into the GLB and PLY a run writes (`latent` and
    `camera` in the file, or a recording's `final_latent` and `final_camera`), the octree's draws from `seed`; for
    turbo-engine's denoiser while the decoders are not ported."""
    sys.path.insert(0, str(sdk.resolve()))
    from idol.camera import decode_camera
    from idol.models import GaussianFixedlenDecoder, OctreeProbabilityFixedlenDecoder

    tensors = load_file(str(path))
    prefix = "" if "latent" in tensors else "final_"
    latent = tensors[prefix + "latent"].float().to(device)
    camera = tensors[prefix + "camera"].float()
    config = json.loads((pack / "config.json").read_text())
    decoder = load_idol_model(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]), pack / "decoder.safetensors", device)
    decoder_gs = load_idol_model(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), pack / "decoder_gs.safetensors", device)
    torch.manual_seed(seed)
    with torch.no_grad():
        anchors = OctreeProbabilityFixedlenDecoder.sample(
            decoder, latent, num_points=max(1, gaussians // decoder_gs.rep_config["num_gaussians"]),
            level=config["max_voxel_level"])
        gaussian = decoder_gs.to_representation(anchors, decoder_gs(anchors, latent))[0]
    out.mkdir(parents=True, exist_ok=True)
    gaussian.save_ply(str(out / f"{name}.ply"))
    mesh = write_disc_glb(out / f"{name}.glb", gaussian.xyz.float().cpu().numpy(),
                          gaussian.features_dc.float().cpu().numpy(), gaussian.scaling.float().cpu().numpy(),
                          gaussian.rotation.float().cpu().numpy(), gaussian.opacity.float().cpu().numpy())
    report = {"name": name, "latent": str(path), "gaussians": int(gaussian.xyz.shape[0]), "seed": seed,
              "camera": decode_camera(camera[0, 0]), "mesh": mesh}
    (out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{name}: {mesh['discs']} discs -> {out / (name + '.glb')}", flush=True)


def main():
    if "--latent" in sys.argv:
        decode = argparse.ArgumentParser()
        decode.add_argument("--latent", type=Path, required=True, help="a safetensors file with `latent` and `camera`")
        decode.add_argument("--sdk", type=Path, required=True)
        decode.add_argument("--pack", type=Path, required=True)
        decode.add_argument("--out", type=Path, required=True)
        decode.add_argument("--name", required=True)
        decode.add_argument("--gaussians", type=int, default=262144)
        decode.add_argument("--seed", type=int, default=42)
        decode.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
        o = decode.parse_args()
        return decode_latent(o.latent, o.sdk, o.pack, o.out, o.name, o.gaussians, o.seed, torch.device(o.device))
    if "--truth" in sys.argv:
        truth = argparse.ArgumentParser()
        truth.add_argument("--truth", type=Path, required=True, help="a recording made with --record")
        truth.add_argument("--sdk", type=Path, required=True)
        truth.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
        options = truth.parse_args()
        return add_truth(options.truth, options.sdk, torch.device(options.device))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdk", type=Path, required=True, help="a clone of github.com/speridlabs/idol-sdk")
    parser.add_argument("--pack", type=Path, required=True, help="the Idol-MLX-Serve-8bit snapshot")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--name", help="the outputs' name (the image's own by default)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                        help="'synthetic': renders and generated pictures (crop 1.2); 'real': photos (crop 1.7)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--rescale_t", type=float, default=3.0)
    parser.add_argument("--gaussians", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--record", type=Path, help="write the denoiser's references for `turbo-engine verify` there")
    args = parser.parse_args()

    sys.path.insert(0, str(args.sdk.resolve()))
    from tensordict import TensorDict

    from idol.camera import decode_camera
    from idol.conditioning import ImageEncoder
    from idol.models import BiRefNet, GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
    from idol.models.layers import RePo3DRotaryEmbedding
    from idol.pipeline import IdolPipeline
    from idol.sampler import sample_flow

    device = torch.device(args.device)
    name = args.name or args.image.stem
    args.out.mkdir(parents=True, exist_ok=True)
    stages = Stages(device)
    print(f"{name}: {args.image} on {device}, {args.mode}, {args.steps} steps, guidance {args.guidance}, "
          f"{args.gaussians} Gaussians, seed {args.seed}", flush=True)

    def load():
        config = json.loads((args.pack / "config.json").read_text())
        pipeline = IdolPipeline.__new__(IdolPipeline)     # its __init__ would fetch the gated encoders
        pipeline.device, pipeline.config, pipeline.optimize = device, config, False
        pipeline.denoiser = load_idol_model(LatentSeqMMFlowModel(**config["denoiser"]["args"]),
                                            args.pack / "denoiser.safetensors", device)
        for module in pipeline.denoiser.modules():
            if isinstance(module, RePo3DRotaryEmbedding):
                module.real_valued = True
        pipeline.decoder = load_idol_model(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]),
                                           args.pack / "decoder.safetensors", device)
        pipeline.decoder_gs = load_idol_model(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]),
                                              args.pack / "decoder_gs.safetensors", device)
        pipeline.background_remover = load_idol_model(BiRefNet(), args.pack / "birefnet.safetensors", device)
        encoder = ImageEncoder.__new__(ImageEncoder)
        encoder.device = device
        encoder.dinov3 = load_dinov3(args.pack / "dinov3", device)
        encoder.flux2_vae = load_flux2_vae(args.pack / "vae", device)
        pipeline.image_encoder = encoder
        return pipeline

    pipeline = stages.run("load", load)
    captured = {}
    if args.record:
        capture_stages(pipeline, captured)
    with torch.no_grad():
        prepared = stages.run("matte", lambda: pipeline.preprocess(Image.open(args.image), mode=args.mode))
        prepared.save(args.out / f"{name}_input.png")
        pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(device)

        # IdolPipeline.__call__, split into its stages, with its seed and its draws in its order
        torch.manual_seed(args.seed)
        encoder = pipeline.image_encoder
        feature1 = stages.run("dinov3", lambda: encoder._dinov3_tokens(pixels))
        feature2 = stages.run("vae", lambda: encoder._flux2_tokens(pixels))
        cond = TensorDict({"feature1": feature1, "feature2": feature2}, batch_size=1)
        denoiser = pipeline.denoiser
        noise = TensorDict({
            "latent": torch.randn(1, denoiser.q_token_length, denoiser.in_channels, device=device),
            "camera": torch.randn(1, 1, denoiser.cam_channels, device=device),
        }, batch_size=1, device=device)
        sample = stages.run("denoise", lambda: sample_flow(
            denoiser, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like), steps=args.steps,
            rescale_t=args.rescale_t, cfg_strength=args.guidance, verbose=True, batch_cfg=False))
        latent = sample["latent"]
        per_anchor = pipeline.decoder_gs.rep_config["num_gaussians"]
        anchors = stages.run("octree", lambda: OctreeProbabilityFixedlenDecoder.sample(
            pipeline.decoder, latent, num_points=max(1, int(args.gaussians) // per_anchor),
            level=pipeline.config["max_voxel_level"]))
        gaussian = stages.run("gaussians", lambda: pipeline.decoder_gs.to_representation(
            anchors, pipeline.decoder_gs(anchors, latent))[0])
        if args.record:
            captured["prepared"] = torch.from_numpy(np.array(prepared))
            captured["anchors"] = anchors
            captured.update(decoder_references(pipeline, latent))
            record(args.record, args, denoiser, cond, noise, sample, sample_flow, captured)

    def write():
        gaussian.save_ply(str(args.out / f"{name}.ply"))
        return write_disc_glb(args.out / f"{name}.glb",
                              gaussian.xyz.float().cpu().numpy(), gaussian.features_dc.float().cpu().numpy(),
                              gaussian.scaling.float().cpu().numpy(), gaussian.rotation.float().cpu().numpy(),
                              gaussian.opacity.float().cpu().numpy())

    mesh = stages.run("write", write)
    generation = sum(v for k, v in stages.seconds.items() if k not in ("load", "write"))
    report = {
        "name": name, "image": str(args.image), "device": str(device), "mode": args.mode, "steps": args.steps,
        "guidance": args.guidance, "rescale_t": args.rescale_t, "gaussians": int(gaussian.xyz.shape[0]),
        "seed": args.seed, "camera": decode_camera(sample["camera"][0, 0].float().cpu()),
        "seconds": stages.seconds, "generation_seconds": round(generation, 2),
        "peak_footprint_gb_after": stages.peak_after, "metal_allocated_gb_after": stages.metal_after,
        "mesh": mesh, "torch": torch.__version__,
    }
    (args.out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{name}: generation {generation:.1f} s, peak footprint {max(stages.peak_after.values()):.2f} GB, "
          f"{mesh['discs']} discs -> {args.out / (name + '.glb')}", flush=True)


if __name__ == "__main__":
    main()
