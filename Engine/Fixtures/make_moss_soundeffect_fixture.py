"""The MOSS-SoundEffect v2.0 fixture for `turbo-engine verify`.

OpenMOSS's own code (MOSS-TTS, `moss_soundeffect_v2/`) is not on PyPI and is not run here. The DiT, the denoising loop
and the VAE are checked against the float32 golden tensors that xocialize/moss-soundeffect-mlx captured from that
code on its CPU (`tests/fixtures/swift_goldens.safetensors` and `golden_meta.json`, commit 4e8e3e3: the prompt "a heavy
wooden door creaks open slowly" for 10 s, numpy noise, the velocity at t = 1000 and 500, ten guided steps at scale 4,
the decode). Everything else comes from PyPI code on the real weights:

- the prompt text as `MossSoundEffectPipeline` and `WanPrompter` make it (the lines below follow `_format`,
  `basic_clean` and `whitespace_clean`, with ftfy, html and regex), and its tokens from the pack's tokenizer
  (transformers), for a corpus of prompts;
- Qwen3-1.7B's last hidden state in float32 (transformers' Qwen3ForCausalLM, as `Qwen3TextEncoder` loads it) for a
  few prompts: the truth the port's encoder is judged against;
- the sigma schedule of `FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)` (its two lines, in torch).

Usage (Python 3.12 with torch, transformers 4.57, ftfy 6.3, regex, safetensors, numpy):
    python -I make_moss_soundeffect_fixture.py <goldens dir> <mlx-community pack snapshot> <OpenMOSS snapshot> <out dir>
"""

import html
import json
import sys
from pathlib import Path

import ftfy
import numpy as np
import regex
import torch
from safetensors.numpy import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

goldens, pack, official, out = (Path(a) for a in sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)


def pipeline_text(prompt, seconds):
    """`MossSoundEffectPipeline.__call__`'s `_format`."""
    seconds = round(float(seconds), 1)
    return f"{prompt.strip()} duration: {seconds:.1f}s"


def clean(text):
    """`HuggingfaceTokenizer._clean` with clean='whitespace'."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()
    text = regex.sub(r"\s+", " ", text)
    return text.strip()


tokenizer = AutoTokenizer.from_pretrained(str(pack / "tokenizer"))


def tokens(text):
    batch = tokenizer([clean(text)], return_tensors="np", padding="max_length", truncation=True, max_length=512,
                      add_special_tokens=True)
    return batch.input_ids[0].astype(np.int32), batch.attention_mask[0].astype(np.int32)


corpus = [
    "a heavy wooden door creaks open slowly",
    "  Rain on a tin roof,\tthunder far away.\n",
    "The crisp, rhythmic click-clack of fast typing on a mechanical keyboard.",
    "A dog’s bark echoes — “woof!” — in an empty hall…",
    "清晨小鸟叽叽喳喳地叫着，叫声清脆悦耳。",
    "刷牙的声音，牙刷毛摩擦牙齿的那种沙沙声！（近距离）",
    "Ｆｕｌｌｗｉｄｔｈ　ＬＯＵＤ　ｎｏｉｓｅｓ ｶﾀｶﾅ",
    "ﬁre crackling in a ﬁreplace, ﬂames",
    "Tom &amp; Jerry chase &lt;fast&gt; &#39;boing&#39; &#x41;&#65 &nbsp;spring",
    "a <b>bold</b> &amp; loud horn",
    "café vs café, Å vs Å",
    "beep\x07 boop\x1b[31m red\x1b[0m ​zero width　space line",
    "a C1 \x93quoted\x94 \x85 bit",
    "emoji 🔔🐕‍🦺 and a tag \U000E0067",
    "&AMP; &QUOT;x&QUOT; &NBSP; &APOS;s &amp;amp;amp; &ampx &ltb&gt",
    "",
    "   ",
    "wind " * 300,
]
durations = [10, 5, 2.5, 0.25, 0.35, 2.35, 1.05, 29.95, 30, 0.1, 7.77]

prompts = []
for text in corpus:
    for seconds in ([10] if text else [None]):
        full = pipeline_text(text, seconds) if seconds is not None else text
        ids, mask = tokens(full)
        prompts.append({"prompt": text, "seconds": seconds, "text": full, "clean": clean(full),
                        "ids": ids[: int(mask.sum())].tolist()})
# The negative prompt goes in without a duration.
for text in ["", "music, speech", "低沉的嗡嗡声"]:
    ids, mask = tokens(text)
    prompts.append({"prompt": text, "seconds": None, "text": text, "clean": clean(text), "ids": ids[: int(mask.sum())].tolist()})
durations_json = [{"seconds": s, "text": pipeline_text("x", s), "samples": int(48000 * round(float(s), 1))} for s in durations]
(out / "prompts.json").write_text(json.dumps({"prompts": prompts, "durations": durations_json}, ensure_ascii=False, indent=1))

# The encoder's truth: Qwen3 in float32, the last of `hidden_states` as `Qwen3TextEncoder.forward` returns it.
encoder = AutoModelForCausalLM.from_pretrained(str(pack / "text_encoder"), torch_dtype=torch.float32)
encoder.eval()
refs = {}
encoder_prompts = [pipeline_text("a heavy wooden door creaks open slowly", 10),
                   pipeline_text("清晨小鸟叽叽喳喳地叫着，叫声清脆悦耳。", 5),
                   "music, speech"]
for index, text in enumerate(encoder_prompts):
    ids, mask = tokens(text)
    result = encoder(input_ids=torch.from_numpy(ids[None].astype(np.int64)), attention_mask=torch.from_numpy(mask[None].astype(np.int64)),
                     output_hidden_states=True, use_cache=False)
    states = result.hidden_states[-1][0].float().numpy().copy()
    valid = int(mask.sum())
    states[valid:] = 0
    refs[f"encoder{index}_ids"] = ids
    refs[f"encoder{index}_mask"] = mask
    refs[f"encoder{index}_states"] = states[None]
    # Which state that is: transformers puts the final norm's output last.
    final = encoder.model(input_ids=torch.from_numpy(ids[None].astype(np.int64)),
                          attention_mask=torch.from_numpy(mask[None].astype(np.int64))).last_hidden_state[0].float().numpy()
    same = np.array_equal(final, result.hidden_states[-1][0].float().numpy())
    print(f"encoder prompt {index}: {valid} tokens; hidden_states[-1] is the final norm's output: {same}")
del encoder

# The same in bf16, as the pipeline loads it: how far the reference's own rounding takes it from float32.
encoder = AutoModelForCausalLM.from_pretrained(str(pack / "text_encoder"), torch_dtype=torch.bfloat16)
encoder.eval()
for index, text in enumerate(encoder_prompts):
    ids, mask = tokens(text)
    result = encoder(input_ids=torch.from_numpy(ids[None].astype(np.int64)), attention_mask=torch.from_numpy(mask[None].astype(np.int64)),
                     output_hidden_states=True, use_cache=False)
    states = result.hidden_states[-1][0].float().numpy().copy()
    states[int(mask.sum()):] = 0
    refs[f"encoder{index}_states_bf16"] = states[None]
del encoder

# The schedule.
for steps in [1, 2, 10, 50, 100, 200]:
    sigmas = torch.linspace(1.0, 0.0, steps + 1)[:-1]
    sigmas = 5.0 * sigmas / (1 + (5.0 - 1) * sigmas)
    refs[f"sigmas{steps}"] = sigmas.numpy().astype(np.float32)

# The goldens.
golden = load_file(str(goldens / "swift_goldens.safetensors"))
for name, value in golden.items():
    refs[name] = value
meta = json.loads((goldens / "golden_meta.json").read_text())

save_file(refs, str(out / "references.safetensors"))
(out / "fixture.json").write_text(json.dumps({
    "family": "moss-soundeffect",
    "pack": str(pack),
    "official_transformer": str(official / "transformer" / "diffusion_pytorch_model.safetensors"),
    "golden": meta,
    "encoder_prompts": encoder_prompts,
}, ensure_ascii=False, indent=1))
print("wrote", out)
