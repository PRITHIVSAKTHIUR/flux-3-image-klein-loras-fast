# **[flux-3-image-klein-loras-fast](https://huggingface.co/spaces/prithivMLmods/flux-3-image-klein-loras-fast)**

flux-3-image-klein-loras-fast is an interactive image editing, text-to-image synthesis, and dynamic LoRA execution studio powered by the `black-forest-labs/flux-3-image-klein` Distilled architecture (`ModularPipeline`). The workspace couples high-resolution generation (across `1K`, `2K`, and `4K` tiers) with `black-forest-labs/flux-3-image-klein-pup`, a Qwen3.5-based vision-language prompt upsampler that enriches raw instructions and multi-image references prior to 16-step distilled sampling.

The platform supports pre-configured adapters (such as `360-Deg-LoRA`) alongside on-the-fly custom LoRA loading from any Hugging Face repository, complete with an in-memory adapter cache (`CUSTOM_LORA_CACHE`) to prevent redundant downloads. The backend is served via `gradio.Server` and FastAPI, driving a dark-mode single-page web application (SPA) with A/B comparison sliders, input/history filmstrips, and quick prompt suggestions.

<img width="1918" height="900" alt="Screenshot From 2026-10-24 18-12-17" src="https://github.com/user-attachments/assets/a484a1c3-ea63-400e-bd11-db8d210a2ef8" />
<img width="1918" height="900" alt="Screenshot From 2026-10-24 18-11-30" src="https://github.com/user-attachments/assets/88999e24-9647-4ea7-8ec7-41f19568e8c4" />

### **Key Features**

* **FLUX.3 Image Klein Modular Pipeline:** Executes 16-step distilled image-to-image editing, multi-reference composition, and text-to-image generation in `bfloat16` precision.
* **Integrated Pup Prompt Upsampler:** Uses `black-forest-labs/flux-3-image-klein-pup` (`Qwen3_5ForConditionalGeneration`) to automatically expand user instructions and visual reference contexts into detailed captions.
* **Plug-and-Play Custom LoRA Engine:** Supports built-in presets (`360-Deg-LoRA`) and arbitrary Hugging Face LoRA repositories (`user/repo` + optional weight path) with automatic runtime caching.
* **Multi-Tier Resolution & Aspect Scaling:** Supports `1K` ($1024$), `2K` ($2048$), and `4K` ($4096$) long-edge targets across `Auto`, `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, and `9:16` aspect ratios, snapping dimensions to multiples of 16.
* **Studio SPA Interface:** A responsive dark-mode web workspace featuring an interactive before/after comparison slider, chain-edit feedback loops (*Use as Input*), seed copying, and drag-and-drop or clipboard paste support.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── C.jpg
│   ├── cloth.jpg
│   ├── face-swap.jpg
│   ├── image.webp
│   ├── mc.png
│   └── Snow-Klein-consistency.png
├── app.py
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the `flux-3-image-klein-loras-fast` environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU with `bfloat16` support is required.

* **Python Version:** Minimum Python **3.12.12** is required; Python **3.14** is recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for optimal compatibility.
* **CUDA Version:** **CUDA 13.0** is recommended (`--extra-index-url [https://download.pytorch.org/whl/cu130](https://download.pytorch.org/whl/cu130)`), matching the environment running on the live Hugging Face Space.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on `uv.lock`.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/flux-3-image-klein-loras-fast.git
cd flux-3-image-klein-loras-fast
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the application**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install "pip>=26.2.2"
```

**2. Install Core Dependencies**
Install the primary deep learning stack, bleeding-edge `diffusers` main branch, and supporting libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130

git+https://github.com/huggingface/diffusers.git
torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
peft==0.19.1
gradio==6.30.0
av==17.1.0
spaces>=0.51.1
huggingface-hub>=1.24.0
pillow>=12.3.0
```

### **Usage**

Once the FastAPI server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Upload Reference Media (Optional):** Drag and drop one or more images onto the canvas, paste from your clipboard (⌘/Ctrl+V), or use the upload button in the left rail. Leave the input filmstrip empty for text-to-image synthesis.
2. **Configure LoRA / Style:**
* Choose `None` for the base Distilled model, or select a pre-registered adapter such as `360-Deg-LoRA`.
* Select `Custom LoRA` to expose the **Repo id** (`user/repo`) and optional **Weight file path** fields for on-the-fly Hub loading.


3. **Set Output & Upsampling:** Select your target **Aspect** (`Auto` or fixed ratio) and **Resolution** (`1K`, `2K`, or `4K`). Toggle **Upsample prompt (Pup)** to expand your instruction via the Qwen3.5 prompt upsampler.
4. **Generate:** Enter your instruction in the prompt box and click **Generate** (or press ⌘/Ctrl + Enter).
5. **Compare & Chain:** Use the **Compare** icon on the left rail for an interactive before/after split view, or click **Use as Input** to feed the output directly into a subsequent edit pass.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/flux-3-image-klein-loras-fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/flux-3-image-klein-loras-fast.git](https://github.com/PRITHIVSAKTHIUR/flux-3-image-klein-loras-fast.git)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/flux-3-image-klein-loras-fast](https://huggingface.co/spaces/prithivMLmods/flux-3-image-klein-loras-fast)
