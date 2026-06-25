# SwarmUI WD14 Tagger Extension

A [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) extension that adds automatic image tagging using HuggingFace-hosted booru taggers, including models from [SmilingWolf](https://huggingface.co/SmilingWolf), [deepghs](https://huggingface.co/deepghs), [fancyfeast](https://huggingface.co/fancyfeast), [Camais03](https://huggingface.co/Camais03), and [lodestones](https://huggingface.co/lodestones).

Generate tags from any image in the viewer with one click, or use the `<wd14tagger>` prompt tag to auto-tag at generation time.

---

## Features

- **Generate Tags** button in the image viewer (via the standard media button bar)
- `<wd14tagger>` prompt tag — automatically tags the init image at generation time and injects the tags into the prompt
- Optional prompt-tag positional overrides: `<wd14tagger:model-id,general-threshold,character-threshold>`
- Settings managed as a **WD14 Tagger** parameter group — model, general/character thresholds, filter tags, and insert mode are saved and loaded like any other SwarmUI parameter
- Multiple WD14 model options
- Separate configurable confidence thresholds for **general** tags (default: 0.35) and **character** tags (default: 0.85)
- Each threshold category can be independently toggled off to exclude that tag type entirely
- Tag filter list — exclude or replace exact tags, plus boundary-aware wildcard matching for start / end / contains phrase rules
- Models are downloaded automatically on first use and cached locally under `Models/wd14_tagger/`
- All tagging runs through the self-start ComfyUI backend queue via a custom Comfy node, so heavy tagger execution shares the backend slot with normal generations

## Supported Models

| Label | HuggingFace Repo |
|---|---|
| WD EVA02 Large v3 *(default)* | `SmilingWolf/wd-eva02-large-tagger-v3` |
| WD ViT Large v3 | `SmilingWolf/wd-vit-large-tagger-v3` |
| WD ViT v3 | `SmilingWolf/wd-vit-tagger-v3` |
| WD SwinV2 v3 | `SmilingWolf/wd-swinv2-tagger-v3` |
| WD ConvNext v3 | `SmilingWolf/wd-convnext-tagger-v3` |
| WD SwinV2 v2 | `SmilingWolf/wd-v1-4-swinv2-tagger-v2` |
| WD ViT v2 | `SmilingWolf/wd-v1-4-vit-tagger-v2` |
| WD ConvNext v2 | `SmilingWolf/wd-v1-4-convnext-tagger-v2` |
| PixAI Tagger v0.9 | `deepghs/pixai-tagger-v0.9-onnx` |
| JoyTag | `fancyfeast/joytag` |
| Camie Tagger v1 | `Camais03/camie-tagger` |
| Camie Tagger v2 | `Camais03/camie-tagger-v2` |
| Taggerine (DINOv3 ViT-H/16+) | `lodestones/taggerine` |

### Model Comparison

#### WD Series (SmilingWolf)

Trained on Danbooru. The gold standard for anime/illustration tagging — well-tested, widely used, and the most reliable for general-purpose use. All v3 models cover ~10,800 tags; v2 models cover ~9,000.

**Which v3 to pick:**

| Model | Notes |
|---|---|
| **EVA02 Large** *(default)* | Best overall accuracy in the v3 family. Recommended for most use cases. |
| **ViT Large** | Marginally lower accuracy than EVA02, otherwise comparable. |
| **ViT / SwinV2 / ConvNext** | Slightly lower accuracy than the Large variants. |

**v2 models** are less accurate than their v3 counterparts across the board. Only worth using if you need to match output from an existing v2-based workflow.

**Excels at:** general visual tags (composition, clothing, expressions, colour, style), content ratings, common characters.  
**Struggles with:** recently introduced characters or series, niche/rare tags, non-anime art styles.

---

#### PixAI Tagger v0.9

Trained by pixai labs https://pixai.art/

This extension uses the ONNX export published by [DeepGHS](https://huggingface.co/deepghs/pixai-tagger-v0.9-onnx) for local inference.

Also trained on Danbooru, but from a snapshot through early 2025 — more recent than the WD v3 training data. Output format and threshold behaviour are compatible with WD v3.

**Excels at:** newer characters and tags that post-date the WD v3 training cutoff.  
**Struggles with:** the same failure modes as WD otherwise; still Danbooru-scoped.

---

#### JoyTag

JoyTag is a ViT-B/16 multi-label tagger trained on Danbooru 2021 plus additional hand-tagged images to improve non-anime and photographic coverage. It ships an ONNX export with an ordered `top_tags.txt` list, so it integrates well with the extension's existing ONNX execution path.

**Excels at:** broader cross-domain tagging than pure Danbooru-only WD-family models, including stronger photographic coverage.  
**Tradeoffs:** upstream does not publish category splits alongside the ONNX export, so this extension applies the **General Threshold** to all JoyTag tags. If General Threshold is toggled off, it falls back to Character Threshold as a single global threshold.

---

#### Camie Tagger (v1 / v2)

A community model with a different goal: breadth over precision. Covers **70,000+ tags** across general, character, copyright, artist, meta, and rating categories — far more than any WD model. The macro F1 is lower than WD, meaning it's less consistent on rare tags, but micro F1 (common tags) is competitive. v2 improves on v1's accuracy.

**Excels at:** catching niche tags, copyright/series identification, and artist tags where WD returns sparse results.
**Struggles with:** consistency on uncommon tags; higher false-positive rate at lower thresholds. Consider raising the general threshold slightly (0.4–0.5) compared to WD defaults.

---

#### Taggerine

Taggerine is a large DINOv3 ViT-H/16+ tagger trained on combined Danbooru and e621 annotations. It covers **74,000+ tags** and is especially strong when you want broader booru-style coverage than standard WD models. In this extension it reuses the existing threshold UI by treating character-category tags with the character threshold and all other non-rating categories with the general threshold.

**Excels at:** broad tag coverage, character detection, and non-WD vocab breadth.  
**Tradeoffs:** much heavier first-run setup than the ONNX models. The checkpoint is about 5.3 GB and requires PyTorch-based inference, so first use is slower and disk usage is substantially higher.

---

**Not sure which to use?** Start with **WD EVA02 Large v3**. If you want broader vocab coverage or stronger non-WD-style booru tagging and you do not mind the heavier model download, try **Taggerine**.

---

## Requirements

- [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) (tested on v0.9.x / v1.x)
- A self-start ComfyUI backend, since the extension now routes all tagging through a custom Comfy node
- Python 3.10+ with the following packages:
  - `onnxruntime` (or `onnxruntime-gpu` for GPU inference)
  - `Pillow`
  - `numpy`
  - `huggingface_hub`

The extension will automatically install dependencies, but if you need to manually install run:

```bash
pip install onnxruntime Pillow numpy huggingface_hub
```

> For GPU acceleration replace `onnxruntime` with `onnxruntime-gpu`.
>
> **Note:** Taggerine uses the existing Comfy Python runtime for `torch`, `torchvision`, `requests`, and `safetensors` rather than installing its own copies. The model checkpoint is still roughly 5.3 GB on first use.

---

## Installation

1. Clone this repository into SwarmUI's `src/Extensions/` folder:

   ```bash
   cd SwarmUI/src/Extensions
   git clone https://github.com/GlenCarpenter/SwarmUI-WD14Tagger.git
   ```

2. Rebuild and restart SwarmUI. The extension will be picked up automatically.

---

## General usage

<img width="1376" height="801" alt="image" src="https://github.com/user-attachments/assets/2257fa2c-bdbd-4003-b8e1-b0e0e577e320" />

### Generate Tags Button

The **Generate Tags** button appears in the image viewer's media button bar (the same bar as Save, Send to Init Image, etc.).

1. Generate or load an image so it appears in the viewer.
2. Click **Generate Tags**.
3. The prompt box is populated with the detected tags according to the current **Insert Mode** setting.

If `<wd14tagger>` is already present anywhere in the prompt when the button is clicked, the tag is replaced in-place with the generated tags. Otherwise the **Insert Mode** (Replace / Prepend / Append) controls where tags are inserted.

### `<wd14tagger>` Prompt Tag

Place `<wd14tagger>` anywhere in your prompt text and SwarmUI will automatically tag your init image (or first prompt image) at generation time, replacing the tag with the detected WD14 tags before the generation starts. No button click needed.

```
masterpiece, <wd14tagger>, best quality
```

To override the normal **[WD14 Tagger] Model** setting for just one prompt tag, add a model ID after a colon:

```
masterpiece, <wd14tagger:fancyfeast/joytag>, best quality
```

You can also override the **General Threshold** and **Character Threshold** for that specific prompt tag by adding positional arguments after the model ID:

```
masterpiece, <wd14tagger:deepghs/pixai-tagger-v0.9-onnx,0.5,0.85>, best quality
```

Spaces around commas are ignored, so this is equivalent:

```
masterpiece, <wd14tagger:deepghs/pixai-tagger-v0.9-onnx, 0.5, 0.85>, best quality
```

When one of those positional values is omitted, the prompt tag falls back to the value from the **WD14 Tagger** parameter group. The filter list is always taken from the parameter group. By default the prompt tag also uses that group's selected model, unless you explicitly override it in the tag. The `<wd14tagger>` prompt tag requires an init image or prompt image to be set — if none is available a warning is shown and the tag is removed silently.

> **Tip:** Start typing `<wd14` in the prompt box to find the tag in the autocomplete dropdown, then type `:` after `wd14tagger` to see the available model IDs and the per-tag syntax reminder.

### Filter tags usage (exclude or replace tags)

Build your filter list as a single comma-separated string, where each entry is one rule. You can mix exclude and replace rules in the same list. SwarmUI reads entries left to right as separate rules, for example:

`solo, simple background, *hair*:wig, red*:blue, 1girl:person`

In that example, each comma-separated item is processed as its own rule. Use the exclude and replace formats in the tables below to decide what each rule should do.

Wildcard matching is boundary-aware rather than generic globbing. Boundaries are any non-alphanumeric characters (for example spaces, `+`, and `-`).

Within each category, exact rules are applied before wildcard rules. When both replace and exclude rules are present together, replace rules run before exclude rules.

#### Exclude usage

| Rule Pattern | Type | Result | Example | Matches | Does Not Match | Output |
|---|---|---|---|---|---|---|
| `<phrase>` | Exact exclude | Removes only tags that exactly equal `<phrase>` | `hair` | `hair` | `black hair`, `facial-hair` | `hair -> [removed]`<br>`black hair -> black hair [ignored]`<br>`facial-hair -> facial-hair [ignored]` |
| `<phrase>*` | Prefix exclude | Removes tags that start with `<phrase>` on boundaries | `red*` | `red`, `red eyes`, `red+vehicle` | `orange red`, `redirect` | `red -> [removed]`<br>`red eyes -> [removed]`<br>`red+vehicle -> [removed]`<br>`orange red -> orange red [ignored]`<br>`redirect -> redirect [ignored]` |
| `*<phrase>` | Suffix exclude | Removes tags that end with `<phrase>` on boundaries | `*hair` | `hair`, `black hair`, `facial-hair`, `6+vehicles` (with `*vehicles`) | `hair style`, `chair` | `hair -> [removed]`<br>`black hair -> [removed]`<br>`facial-hair -> [removed]`<br>`6+vehicles -> 6+vehicles [ignored for *hair]`<br>`hair style -> hair style [ignored]`<br>`chair -> chair [ignored]` |
| `*<phrase>*` | Contains exclude | Removes tags containing `<phrase>` on boundaries | `*hop*` | `hop scotch`, `my hop`, `my-hop-tag` | `hope` | `hop scotch -> [removed]`<br>`my hop -> [removed]`<br>`my-hop-tag -> [removed]`<br>`hope -> hope [ignored]` |

#### Replace usage

| Rule Pattern | Type | Result | Example | Matches | Does Not Match | Output |
|---|---|---|---|---|---|---|
| `<phrase>:<replacement>` | Exact replace | Replaces the full tag when it exactly equals `<phrase>` | `hair:wig` | `hair` | `black hair`, `facial-hair` | `hair -> wig [changed]`<br>`black hair -> black hair [ignored]`<br>`facial-hair -> facial-hair [ignored]` |
| `<phrase>*:<replacement>` | Prefix replace | Replaces only the matched prefix phrase | `red*:blue` | `red eyes` | `orange red`, `redirect` | `red eyes -> blue eyes [changed]`<br>`orange red -> orange red [ignored]`<br>`redirect -> redirect [ignored]` |
| `*<phrase>:<replacement>` | Suffix replace | Replaces only the matched suffix phrase | `*hair:wig` | `black hair` | `hair style`, `chair` | `black hair -> black wig [changed]`<br>`hair style -> hair style [ignored]`<br>`chair -> chair [ignored]` |
| `*<phrase>*:<replacement>` | Contains replace | Replaces only the matched phrase inside the tag | `*hair*:wig` | `black hair`, `hair style`, `big hair style` | `chair`, `hairstyle` | `black hair -> black wig [changed]`<br>`hair style -> wig style [changed]`<br>`big hair style -> big wig style [changed]`<br>`chair -> chair [ignored]`<br>`hairstyle -> hairstyle [ignored]` |
#### Apply to current prompt

The **Apply to current prompt** button (under the Filter Tags field) runs these same rules against the text already in your prompt box and replaces it with the filtered result, making it convenient to clean up a hand-written, weighted prompt without re-tagging an image.

#### Note on prompt-weighting syntax

When using the **Apply to current prompt** button, you may use/encounter usage of prompt-weighting syntax. The tagger models themselves do not apply this syntax.

Filter rules will match the underlying tag, so standard prompt-weighting decoration is ignored when rules are evaluated. Surrounding parentheses and a trailing `:<number>` weight are peeled off before matching, then re-applied to tags that are kept or replaced:

- `((sweater))` is matched as `sweater`
- `realistic:1.3` is matched as `realistic`
- `(blue eyes:1.2)` is matched as `blue eyes`

This means a rule like `sweater` removes `((sweater))`, and a rule like `realistic:photographic` turns `(realistic:1.3)` into `(photographic:1.3)` — the weight and parentheses are preserved on the result. Only a colon immediately followed by a number is treated as a weight, so non-numeric uses of `:` are unaffected.

---

### WD14 Tagger Parameter Group

All settings live in the **WD14 Tagger** group in the parameter sidebar. They are saved and restored by SwarmUI's normal parameter save/load system, including the built-in parameter memory and presets.

> **Recommended:** Use SwarmUI **Presets** to store your preferred WD14 settings (model, thresholds, filter tags, insert mode) for reliable reuse across sessions.

| Setting | Description |
|---|---|
| **[WD14 Tagger] Model** | Tagger model to use for inference |
| **[WD14 Tagger] General Threshold** | Minimum confidence score (0.0–1.0) for a general tag to be included (default: 0.35). Toggle off to suppress all general tags. |
| **[WD14 Tagger] Character Threshold** | Minimum confidence score (0.0–1.0) for a character tag to be included (default: 0.85). Toggle off to suppress all character tags. |
| **[WD14 Tagger] Filter Tags** | Comma-separated tag rules: use `tag` to remove it, `source:target` to replace an exact tag, or wildcard forms like `tag*:new`, `*tag:new`, and `*tag*:new` to substitute only the matching phrase on word boundaries. Non-alphanumeric separators like spaces, `+`, and `-` count as boundaries. Example: `solo, *background*, asian:person, red*:primary color, *people:crowd, *hair*:wig` |
| **[WD14 Tagger] Insert Mode** | How tags are inserted into the prompt: Replace, Prepend, or Append |

## How It Works

1. The current image is read from the SwarmUI viewer and base64-encoded in the browser.
2. It is sent to the C# API endpoint (`WD14TaggerGenerateTags`).
3. The API submits a tiny workflow to a self-start ComfyUI backend using the extension's `WD14TaggerGenerate` custom node.
4. That custom node installs dependencies in the Comfy Python environment if needed, invokes `wd14_tagger_inference.py`, and writes the resulting tags to a temporary text file.
5. The C# layer reads the result, applies any exact or wildcard tag substitutions/exclusions, and returns the final comma-separated tag string to the frontend.
6. Tags are inserted into the prompt text box.

---

## License

MIT License

Copyright (c) 2026 Glen Carpenter

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
