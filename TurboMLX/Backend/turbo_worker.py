#!/usr/bin/env python3
"""Turbo MLX backend: runs mflux models on behalf of the Swift app.

Both modes speak JSON lines on stdout, one object per line with an "event" key.
Everything else (mflux prints, tqdm bars, warnings, tracebacks) goes to stderr, which the app
shows as the backend log.

  turbo_worker.py serve
      Long-lived worker that keeps the last model loaded. Commands arrive on stdin:
        {"cmd": "generate", "id": "<job>", "model": {...}, "params": {...}}
        {"cmd": "cancel", "id": "<job>"}   stops that job at the next denoising step
        {"cmd": "unload"}                  frees the loaded model
        {"cmd": "shutdown"}

  turbo_worker.py download --repo org/name [--include PATTERN ...]
      Downloads a model into the Hugging Face cache, reporting byte progress.
"""

import argparse
import fnmatch
import json
import os
import queue
import sys
import threading
import time
import traceback


def _reserve_stdout():
    # Keep the real stdout for protocol messages and point fd 1 at stderr, so that anything
    # printed by Python or native code ends up in the log instead of corrupting the protocol.
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


PROTO = _reserve_stdout()
_emit_lock = threading.Lock()


def emit(event: str, **fields) -> None:
    line = json.dumps({"event": event, **fields}, ensure_ascii=False)
    with _emit_lock:
        PROTO.write(line + "\n")
        PROTO.flush()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------------------------


class Cancelled(Exception):
    pass


class MingFamily:
    """inclusionAI Ming-Image-0.1-Design: graphic design, RGBA output."""

    # Its ~16B text encoder is freed after encoding in low-RAM mode (see Worker._generate).
    releases_text_encoder = True

    @staticmethod
    def is_cached(model, prompt: str) -> bool:
        return prompt in model._prompt_cache

    @staticmethod
    def load(spec: dict):
        from mflux.models.common.resolution.config_resolution import ConfigResolution
        from mflux.models.ming_image.variants.ming_image import MingImage

        config = ConfigResolution.resolve_restricted(None, "ming-image-design", model_path=spec["path"])
        return MingImage(model_config=config, model_path=spec["path"], low_ram=bool(spec.get("low_ram")))

    @staticmethod
    def encode(model, prompt: str, spec: dict) -> None:
        # With low_ram, encode_prompt frees the ~16B text encoder after encoding a new prompt;
        # releasing again covers prompts served from the cache while the encoder is resident.
        low_ram = bool(spec.get("low_ram"))
        model.low_ram = low_ram
        model.encode_prompt(prompt)
        if low_ram:
            model.release_text_encoder()

    @staticmethod
    def generate(model, p: dict):
        from mflux.models.ming_image.variants.ming_image import MingImage

        image = model.generate_image(
            seed=int(p["seed"]),
            prompt=p["prompt"],
            width=int(p["width"]),
            height=int(p["height"]),
            num_inference_steps=int(p["steps"]),
            guidance=float(p["guidance"]),
        )
        if p.get("flatten_alpha"):
            image.image = MingImage.to_rgb(image.image)
        return image


class ZImageTurboFamily:
    """Tongyi-MAI Z-Image-Turbo: guidance-distilled, 9 steps."""

    @staticmethod
    def load(spec: dict):
        from mflux.models.common.resolution.config_resolution import ConfigResolution
        from mflux.models.z_image.variants.z_image import ZImage

        config = ConfigResolution.resolve_restricted(None, "z-image-turbo", model_path=spec["path"])
        return ZImage(model_config=config, model_path=spec["path"])

    @staticmethod
    def encode(model, prompt: str, spec: dict) -> None:
        pass  # the prompt is encoded inside generate_image

    @staticmethod
    def generate(model, p: dict):
        return model.generate_image(
            seed=int(p["seed"]),
            prompt=p["prompt"],
            width=int(p["width"]),
            height=int(p["height"]),
            num_inference_steps=int(p["steps"]),
        )


class Flux2KleinFamily:
    """Black Forest Labs FLUX.2 [klein]: distilled 4B/9B (4 steps) or base (CFG)."""

    @staticmethod
    def load(spec: dict):
        from mflux.models.common.config.model_config import AVAILABLE_MODELS
        from mflux.models.common.resolution.config_resolution import ConfigResolution
        from mflux.models.flux2.variants import Flux2Klein

        siblings = tuple(key for key in AVAILABLE_MODELS if key.startswith("flux2-") and key != "flux2-klein-4b")
        # The geometry (4B or 9B, distilled or base) comes from an explicit variant or from the
        # model's name; the cache folder the weights live in is named after a commit hash.
        config = ConfigResolution.resolve_restricted(
            None,
            "flux2-klein-4b",
            model_path=spec.get("name") or spec["path"],
            extra_keys=siblings,
            base_model=spec.get("variant") or None,
        )
        return Flux2Klein(model_config=config, model_path=spec["path"])

    @staticmethod
    def encode(model, prompt: str, spec: dict) -> None:
        pass  # the prompt is encoded inside generate_image

    @staticmethod
    def generate(model, p: dict):
        # Distilled checkpoints only work at guidance 1.0; base checkpoints run real CFG.
        is_base = "base" in model.model_config.model_name.lower()
        return model.generate_image(
            seed=int(p["seed"]),
            prompt=p["prompt"],
            width=int(p["width"]),
            height=int(p["height"]),
            num_inference_steps=int(p["steps"]),
            guidance=float(p["guidance"]) if is_base else 1.0,
        )


class QwenImageFamily:
    """Alibaba Qwen-Image 2512: 20B MMDiT, true CFG (two transformer passes per step)."""

    # Its Qwen2.5-VL text encoder (~14 GB) is freed after encoding in low-RAM mode.
    releases_text_encoder = True
    # Qwen's own pipeline encodes an empty negative prompt as a single space.
    NEGATIVE_PROMPT = " "

    @staticmethod
    def load(spec: dict):
        from mflux.models.common.resolution.config_resolution import ConfigResolution
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        config = ConfigResolution.resolve_restricted(None, "qwen-image", model_path=spec["path"])
        return QwenImage(model_config=config, model_path=spec["path"])

    @staticmethod
    def is_cached(model, prompt: str) -> bool:
        return f"{prompt}|NEG|{QwenImageFamily.NEGATIVE_PROMPT}" in model.prompt_cache

    @staticmethod
    def encode(model, prompt: str, spec: dict) -> None:
        # Encoding ahead of generate_image fills the prompt cache, which generate_image then reads
        # without touching the text encoder, so low-RAM mode can free it before denoising.
        import gc

        import mlx.core as mx
        from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder

        if not QwenImageFamily.is_cached(model, prompt):
            mx.eval(QwenPromptEncoder.encode_prompt(
                prompt=prompt,
                negative_prompt=QwenImageFamily.NEGATIVE_PROMPT,
                prompt_cache=model.prompt_cache,
                qwen_tokenizer=model.tokenizers["qwen"],
                qwen_text_encoder=model.text_encoder,
            ))
        if spec.get("low_ram") and model.text_encoder is not None:
            model.text_encoder = None
            gc.collect()
            mx.clear_cache()

    @staticmethod
    def generate(model, p: dict):
        return model.generate_image(
            seed=int(p["seed"]),
            prompt=p["prompt"],
            negative_prompt=QwenImageFamily.NEGATIVE_PROMPT,
            width=int(p["width"]),
            height=int(p["height"]),
            num_inference_steps=int(p["steps"]),
            guidance=float(p["guidance"]),
        )


FAMILIES = {
    "ming": MingFamily,
    "z-image-turbo": ZImageTurboFamily,
    "flux2-klein": Flux2KleinFamily,
    "qwen-image": QwenImageFamily,
}


class ProgressReporter:
    """mflux callback: turns denoising steps into progress events and honours cancellation."""

    def __init__(self, worker: "Worker"):
        self.worker = worker

    def call_before_loop(self, seed, prompt, latents, config, **_):
        emit("phase", id=self.worker.job_id, phase="denoising", total=config.num_inference_steps)

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        import mlx.core as mx

        # The loop evaluates its lazy graph right after this callback; forcing it here makes the
        # reported step actually finished (the loop's own eval then has nothing left to do).
        mx.eval(latents)
        emit("progress", id=self.worker.job_id, step=t + 1, total=config.num_inference_steps)
        self.worker.raise_if_cancelled()

    def call_after_loop(self, seed, prompt, latents, config):
        emit("phase", id=self.worker.job_id, phase="decoding")


class Worker:
    def __init__(self):
        self.commands: queue.Queue = queue.Queue()
        self.model = None
        self.model_key = None
        # Whether the loaded model has generated: its weights are then resident, not lazy.
        self.model_used = False
        self.job_id = None
        self._default_cache_limit = None
        self._cancelled_ids: set[str] = set()
        self._cancel_lock = threading.Lock()

    # stdin is read on a thread so that "cancel" can arrive while a job runs on the main thread.
    def _read_stdin(self) -> None:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log(f"[turbo] invalid command: {raw[:200]}")
                continue
            if msg.get("cmd") == "cancel":
                with self._cancel_lock:
                    self._cancelled_ids.add(str(msg.get("id")))
            else:
                self.commands.put(msg)
        self.commands.put({"cmd": "shutdown"})  # stdin closed: the app is gone

    def raise_if_cancelled(self) -> None:
        with self._cancel_lock:
            if self.job_id in self._cancelled_ids:
                raise Cancelled()

    def run(self) -> None:
        threading.Thread(target=self._read_stdin, daemon=True).start()
        emit("ready", **_environment_info())
        while True:
            msg = self.commands.get()
            cmd = msg.get("cmd")
            if cmd == "shutdown":
                break
            if cmd == "generate":
                self._generate(msg)
            elif cmd == "unload":
                self._unload()
                emit("unloaded")
            else:
                log(f"[turbo] unknown command: {cmd}")

    def _ensure_model(self, spec: dict):
        family = FAMILIES.get(spec.get("family", ""))
        if family is None:
            raise ValueError(f"Unsupported model family: {spec.get('family')!r}")
        key = (spec["family"], spec["path"], spec.get("variant"))
        if self.model_key != key:
            self._unload()
            emit("phase", id=self.job_id, phase="loading")
            started = time.time()
            model = family.load(spec)
            model.callbacks.register(ProgressReporter(self))
            self.model, self.model_key, self.model_used = model, key, False
            log(f"[turbo] model loaded in {time.time() - started:.1f}s: {spec['path']}")
            emit("model_loaded", path=spec["path"], seconds=round(time.time() - started, 1))
        return family, self.model

    def _unload(self) -> None:
        if self.model is None:
            return
        import gc

        import mlx.core as mx

        self.model, self.model_key, self.model_used = None, None, False
        gc.collect()
        mx.clear_cache()

    def _apply_memory_mode(self, model, low_ram: bool) -> None:
        # What mflux's --low-ram does that suits a long-lived process: decode the image in tiles
        # (only where the VAE shows no seams: not FLUX.2) and keep MLX's buffer cache small.
        import mlx.core as mx
        from mflux.models.common.vae.tiling_config import TilingConfig

        if self._default_cache_limit is None:
            self._default_cache_limit = mx.set_cache_limit(0)
            mx.set_cache_limit(self._default_cache_limit)
        if low_ram:
            model.tiling_config = TilingConfig() if TilingConfig.may_tile_implicitly(model) else None
            mx.set_cache_limit(1 << 30)
        else:
            model.tiling_config = None
            mx.set_cache_limit(self._default_cache_limit)

    def _generate(self, msg: dict) -> None:
        import mlx.core as mx

        self.job_id = str(msg.get("id"))
        params = msg.get("params", {})
        started = time.time()
        try:
            self.raise_if_cancelled()
            mx.reset_peak_memory()
            spec = msg.get("model", {})
            low_ram = bool(spec.get("low_ram"))
            family, model = self._ensure_model(spec)
            if (low_ram and getattr(family, "releases_text_encoder", False) and self.model_used
                    and not family.is_cached(model, params["prompt"])):
                # A new prompt needs the text encoder, but the previous image left the other
                # weights resident: loading it now would stack both. Start from a fresh, lazily
                # loaded model, as the mflux CLI does on every run, so they never overlap.
                self._unload()
                family, model = self._ensure_model(spec)
            self._apply_memory_mode(model, low_ram)
            self.raise_if_cancelled()

            emit("phase", id=self.job_id, phase="encoding")
            family.encode(model, params["prompt"], spec)
            self.raise_if_cancelled()

            self.model_used = True
            image = family.generate(model, params)

            emit("phase", id=self.job_id, phase="saving")
            output = params["output"]
            os.makedirs(os.path.dirname(output), exist_ok=True)
            image.save(path=output, overwrite=True)
            if not os.path.exists(output):
                raise RuntimeError(f"The image was not saved to {output}")

            emit(
                "done",
                id=self.job_id,
                path=output,
                seed=int(params["seed"]),
                width=image.image.width,
                height=image.image.height,
                seconds=round(time.time() - started, 2),
                peak_memory=mx.get_peak_memory(),
            )
        except Cancelled:
            emit("cancelled", id=self.job_id)
        except Exception as exc:  # noqa: BLE001 - report every failure to the app
            traceback.print_exc()
            emit("failed", id=self.job_id, message=f"{type(exc).__name__}: {exc}")
        finally:
            with self._cancel_lock:
                self._cancelled_ids.discard(self.job_id)
            self.job_id = None
            mx.clear_cache()


def _environment_info() -> dict:
    from importlib.metadata import version

    import mlx.core as mx

    info = mx.device_info()
    return {
        "mflux": version("mflux"),
        "mlx": mx.__version__,
        "python": sys.version.split()[0],
        "device": info.get("device_name"),
        "memory": info.get("memory_size"),
    }


def serve() -> None:
    # The app downloads models explicitly (with progress); never let a generation start a
    # silent multi-gigabyte download instead.
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        Worker().run()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        emit("fatal", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)


# --------------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------------


def download(repo: str, patterns: list[str]) -> None:
    import tqdm as tqdm_module
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache

    try:
        files = [
            entry
            for entry in HfApi().list_repo_tree(repo, recursive=True)
            if getattr(entry, "size", None) is not None
            and (not patterns or any(fnmatch.fnmatch(entry.path, p) for p in patterns))
        ]
        total = sum(entry.size for entry in files)
        # snapshot_download reports nothing for files that are already cached.
        cached = sum(
            entry.size for entry in files if isinstance(try_to_load_from_cache(repo, entry.path), str)
        )
        emit("download_started", repo=repo, total=total, cached=cached, files=len(files))

        bars: list = []

        class ByteTracker(tqdm_module.tqdm):
            # snapshot_download aggregates all files into two byte bars: "Downloading bytes"
            # (network, smooth) and "Reconstructing" (written to disk; with hf_xet it only moves
            # when whole files land). Keep both, silenced, and report whichever is further along.
            def __init__(self, *args, **kwargs):
                kwargs["file"] = open(os.devnull, "w")  # noqa: SIM115
                super().__init__(*args, **kwargs)
                if kwargs.get("unit") == "B":
                    bars.append(self)

        finished = threading.Event()

        def report() -> None:
            while not finished.wait(0.4):
                done = max((int(bar.n) for bar in bars), default=0)
                emit("download_progress", bytes=min(total, cached + done), total=total)

        threading.Thread(target=report, daemon=True).start()
        path = snapshot_download(repo_id=repo, allow_patterns=patterns or None, tqdm_class=ByteTracker)
        finished.set()
        emit("download_progress", bytes=total, total=total)
        emit("download_done", repo=repo, path=path)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        emit("failed", message=f"{type(exc).__name__}: {exc}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("serve")
    dl = sub.add_parser("download")
    dl.add_argument("--repo", required=True)
    dl.add_argument("--include", nargs="*", default=[])
    args = parser.parse_args()

    if args.mode == "serve":
        serve()
    else:
        download(args.repo, args.include)


if __name__ == "__main__":
    main()
