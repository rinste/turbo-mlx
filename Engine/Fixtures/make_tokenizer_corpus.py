"""References for `turbo-engine verify-tokenizers`: what the reference tokenizers make of a corpus of
prompts, for every family's prompt pipeline, so the engine's own tokenizer (`Tokenizer/`) can be
checked without Python.

    $PY Engine/Fixtures/make_tokenizer_corpus.py Engine/Fixtures/tokenizers.json

It reads the tokenizers of the catalog's checkpoints from the Hugging Face cache (the commits the
catalog names) with `transformers`' fast tokenizers, which mflux and ltx-2-mlx call, and records,
for each pipeline, the folder it read and the ids of every prompt, built as the families build
them: FLUX.2 Klein and Z-Image with Qwen3's chat template (thinking off and on), Qwen-Image and
Qwen-Image Edit with their templates around the prompt, Ming-Image with its template and no
special tokens, LTX-2's Gemma 3 with `<bos>` and the prompt stripped.
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HUB = Path.home() / ".cache/huggingface/hub"

QWEN_IMAGE_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
MING_TEMPLATE = (
    "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>{}"
    "<|role_end|><role>ASSISTANT</role>"
)


def edit_text(prompt: str, image_tokens: int) -> str:
    return (
        "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, "
        "background), then explain how the user's text instruction should alter or modify the image. Generate a new "
        "image that meets the user's requirements while maintaining consistency with the original input where "
        "appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|>" + "<|image_pad|>" * image_tokens
        + f"<|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )


# (name, repo, commit, sub-folder)
PIPELINES = [
    ("qwen3-chat-no-thinking", "mflux-community/flux2-klein-4b-mflux-q4", "794cd159538149ad9830848508c31f0ea7088e58", "tokenizer"),
    ("qwen3-chat-thinking", "mflux-community/z-image-turbo-mflux-q4", "f427e257d8e6ffa03edd4d9ac554a05809da456c", "tokenizer"),
    ("qwen-image", "mflux-community/qwen-image-2512-mflux-q4", "ec35d366eeb701838007c1720c3d80f2f7fbf9f4", "tokenizer"),
    ("qwen-image-edit", "mflux-community/qwen-image-edit-2511-mflux-q4", "720ad94d982b3dc22f9122ee96af31221d542d47", "tokenizer"),
    ("ming", "joeynyc/Ming-Image-0.1-Design-mflux-q8-te5", "3adb8aaef779f9b3fa4621acebeba97bb4010d42", "mllm"),
    ("gemma3", "mlx-community/gemma-3-12b-it-4bit", "86cc6a8dedbc456dd0e4af01a9d09f396f77e558", ""),
]

PROMPTS = [
    "",
    " ",
    "a cat",
    "A cozy reading nook with a cat asleep on a pile of books, warm afternoon light, photo",
    "A lighthouse on a rocky coast at sunset, a wooden sign that says 'TURBO MLX', photo",
    "Make it winter: snow on the street and on the car",
    "  leading and trailing spaces  ",
    "tabs\tand\ttabs\t\t",
    "line one\nline two\n\nline four\n\n\n",
    "windows\r\nline\r\nendings\r\n",
    "trailing spaces before a newline   \nnext",
    "\n\nstarts with newlines",
    "multiple     spaces     between     words",
    "It's, IT'S, it's, we'LL, you'RE, they've, I'm, she'd, don't, CAN'T",
    "'s 't 're 've 'm 'll 'd alone",
    "Numbers: 0 1 12 123 1234 12345 123456 3.14159 1,000,000 1e10 -42 +7 50% 1/2",
    "Dates 2026-09-29, times 21:30:05, phones +39 02 1234 5678",
    "Punctuation!!! ??? ... --- ___ *** ### @@@ $$$ %%% ^^^ &&& ((( ))) [[[ ]]] {{{ }}}",
    "Quotes: “curly” ‘single’ «guillemets» „low“ 'straight' \"double\" `backtick`",
    "Una gatta bianca che dorme su una poltrona di velluto verde, luce del pomeriggio, foto",
    "Perché è così? Più città, università, caffè, però, già, là, giù, così è",
    "Ä Ö Ü ä ö ü ß straße Größe",
    "Élégant café à Paris, crème brûlée, garçon, naïve, cœur",
    "El niño español, mañana, pingüino, ¿qué? ¡sí!",
    "Português: ação, coração, não, pão",
    "Polski: zażółć gęślą jaźń; Čeština: příliš žluťoučký kůň",
    "Русский текст: съешь же ещё этих мягких французских булок",
    "Ελληνικά: γαζέες καὶ μυρτιὲς",
    "中文提示词：一只在雪地里奔跑的红色狐狸，清晨的柔和光线。",
    "海報上寫著「繁體中文」四個字",
    "日本語のプロンプト：桜の木の下で本を読む少女、カタカナとひらがな",
    "한국어 프롬프트: 서울의 밤거리, 네온사인",
    "العربية: قطة تنام على الأريكة",
    "עברית: חתול ישן על הספה",
    "हिन्दी: बारिश में भीगता हुआ शहर",
    "ภาษาไทย: แมวนอนหลับบนโซฟา",
    "Tiếng Việt: con mèo ngủ trên ghế sô-pha",
    "Emoji 😀 😃 🐱 🚀 ❤️ 👍🏽 👨‍👩‍👧‍👦 🏳️‍🌈 🇮🇹 🇯🇵",
    "Math ∑ ∫ √ ∞ ≈ ≠ ≤ ≥ ± × ÷ π µ Ω °C €100 £50 ¥1000 ₿",
    "Arrows → ← ↑ ↓ ⇒ and boxes ■ □ ▲ ● ★ ☆ ✓ ✗",
    "Combining: é à ñ ö (decomposed accents)",
    "Precomposed: é à ñ ö",
    "Zero width​space, joiner‍, non-breaking space, soft­hyphen",
    "BOM﻿ inside and control \u0007 bell",
    "Private use  and surrogates-free text \U0001f600",
    "URL https://github.com/rinste/turbo-mlx?tab=readme#install and email someone@example.com",
    "Code: def f(x): return x ** 2  # comment\n    if x > 0: print(f'{x=}')",
    "JSON {\"key\": [1, 2, 3], \"nested\": {\"a\": null}}",
    "HTML <div class=\"a\">text</div> &amp; &lt;tag&gt;",
    "Special tokens typed by hand: <|im_start|> <|im_end|> <|endoftext|> <think> </think>",
    "More: <|vision_start|><|image_pad|><|vision_end|> <bos> <eos> <pad> <unk>",
    "Ming's own: <role>HUMAN</role> <|role_end|>",
    "Almost special: <|im_start> <im_end|> < | im_start | >",
    "UPPERCASE PROMPT WITH SHOUTING AND ACRONYMS LIKE NASA, HTTP, MLX, GPU",
    "camelCase snake_case kebab-case PascalCase SCREAMING_SNAKE",
    "hyphenated-words, well-known, state-of-the-art, 8-bit, 4K, 16:9, f/1.8, 50mm",
    "Photo, 85mm lens, shallow depth of field, bokeh, golden hour, cinematic lighting, 8k, ultra detailed",
    "masterpiece, best quality, (detailed:1.2), [background], {style}, <lora:foo:0.8>",
    "a" * 300,
    "word " * 200,
    "日本" * 150,
    "🐱" * 60,
    " ".join(f"token{i}" for i in range(400)),
    "A very long prompt. " + "It describes a scene in painstaking detail, with many clauses, colors and objects; " * 30,
]


def pipeline_ids(name: str, tokenizer, prompt: str) -> list[int]:
    if name == "qwen3-chat-no-thinking":
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    if name == "qwen3-chat-thinking":
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    if name == "qwen-image":
        return tokenizer(QWEN_IMAGE_TEMPLATE.format(prompt), add_special_tokens=True)["input_ids"]
    if name == "qwen-image-edit":
        return tokenizer(edit_text(prompt, 16), add_special_tokens=True)["input_ids"]
    if name == "ming":
        return tokenizer(MING_TEMPLATE.replace("{}", prompt), add_special_tokens=False)["input_ids"]
    if name == "gemma3":
        return tokenizer(prompt.strip(), add_special_tokens=True)["input_ids"]
    raise ValueError(name)


def main() -> None:
    out = Path(sys.argv[1])
    corpus = {"prompts": PROMPTS, "pipelines": []}
    for name, repo, commit, sub in PIPELINES:
        folder = HUB / f"models--{repo.replace('/', '--')}" / "snapshots" / commit / sub
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
        ids = [pipeline_ids(name, tokenizer, p) for p in PROMPTS]
        corpus["pipelines"].append({"name": name, "repo": repo, "commit": commit, "folder": sub, "ids": ids})
        print(f"{name:24} {sum(map(len, ids)):7d} ids  ({type(tokenizer).__name__})")
    out.write_text(json.dumps(corpus, ensure_ascii=False, separators=(",", ":")))
    print("wrote", out, f"{out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
