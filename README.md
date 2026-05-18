# SwarmUI WD14 Tagger Extension

A [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) extension that adds automatic image tagging using WD14-style ONNX models from HuggingFace, including models from [SmilingWolf](https://huggingface.co/SmilingWolf), [deepghs](https://huggingface.co/deepghs), and [Camais03](https://huggingface.co/Camais03).

Generate tags from any image in the viewer with one click, or use the `<wd14tagger>` prompt tag to auto-tag at generation time.

---

## Features

- **Generate Tags** button in the image viewer (via the standard media button bar)
- `<wd14tagger>` prompt tag — automatically tags the init image at generation time and injects the tags into the prompt
- Settings managed as a **WD14 Tagger** parameter group — model, general/character thresholds, filter tags, and insert mode are saved and loaded like any other SwarmUI parameter
- Multiple WD14 model options
- Separate configurable confidence thresholds for **general** tags (default: 0.35) and **character** tags (default: 0.85)
- Each threshold category can be independently toggled off to exclude that tag type entirely
- Tag filter list — exclude specific tags from the output and persist through SwarmUI's built-in parameter memory
- Models are downloaded automatically on first use and cached locally under `Models/wd14_tagger/`

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
| Camie Tagger v1 | `Camais03/camie-tagger` |
| Camie Tagger v2 | `Camais03/camie-tagger-v2` |

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

Also trained on Danbooru, but from a snapshot through early 2025 — more recent than the WD v3 training data. Output format and threshold behaviour are compatible with WD v3.

**Excels at:** newer characters and tags that post-date the WD v3 training cutoff.  
**Struggles with:** the same failure modes as WD otherwise; still Danbooru-scoped.

---

#### Camie Tagger (v1 / v2)

A community model with a different goal: breadth over precision. Covers **70,000+ tags** across general, character, copyright, artist, meta, and rating categories — far more than any WD model. The macro F1 is lower than WD, meaning it's less consistent on rare tags, but micro F1 (common tags) is competitive. v2 improves on v1's accuracy.

**Excels at:** catching niche tags, copyright/series identification, and artist tags where WD returns sparse results.
**Struggles with:** consistency on uncommon tags; higher false-positive rate at lower thresholds. Consider raising the general threshold slightly (0.4–0.5) compared to WD defaults.

---

**Not sure which to use?** Start with **WD EVA02 Large v3**. If you find it missing characters, series names, or artist tags, try **PixAI Tagger v0.9** as a complement or replacement.

---

## Requirements

- [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) (tested on v0.9.x / v1.x)
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

---

## Installation

1. Clone this repository into SwarmUI's `src/Extensions/` folder:

   ```bash
   cd SwarmUI/src/Extensions
   git clone https://github.com/GlenCarpenter/SwarmUI-WD14Tagger.git
   ```

2. Rebuild and restart SwarmUI. The extension will be picked up automatically.

---

## Usage

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

The model, thresholds, and filter list used are taken from the **WD14 Tagger** parameter group. The `<wd14tagger>` prompt tag requires an init image or prompt image to be set — if none is available a warning is shown and the tag is removed silently.

> **Tip:** Start typing `<wd14` in the prompt box to find the tag in the autocomplete dropdown.

### WD14 Tagger Parameter Group

All settings live in the **WD14 Tagger** group in the parameter sidebar. They are saved and restored by SwarmUI's normal parameter save/load system, including the built-in parameter memory and presets.

> **Recommended:** Use SwarmUI **Presets** to store your preferred WD14 settings (model, thresholds, filter tags, insert mode) for reliable reuse across sessions.

| Setting | Description |
|---|---|
| **[WD14 Tagger] Model** | WD14 ONNX model to use for inference |
| **[WD14 Tagger] General Threshold** | Minimum confidence score (0.0–1.0) for a general tag to be included (default: 0.35). Toggle off to suppress all general tags. |
| **[WD14 Tagger] Character Threshold** | Minimum confidence score (0.0–1.0) for a character tag to be included (default: 0.85). Toggle off to suppress all character tags. |
| **[WD14 Tagger] Filter Tags** | Comma-separated exact tag rules: use `tag` to remove it, or `source:target` to replace it (e.g. `solo, simple background, asian:person`) |
| **[WD14 Tagger] Insert Mode** | How tags are inserted into the prompt: Replace, Prepend, or Append |

---

## How It Works

1. The current image is read from the SwarmUI viewer and base64-encoded in the browser.
2. It is sent to the C# API endpoint (`WD14TaggerGenerateTags`).
3. The API writes a temporary file and invokes `wd14_tagger_inference.py` using the configured Python executable.
4. The Python script downloads the selected model from HuggingFace (if not already cached), runs ONNX inference, and returns a JSON result.
5. The C# layer applies any exact-match tag substitutions/exclusions and returns the final comma-separated tag string to the frontend.
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
