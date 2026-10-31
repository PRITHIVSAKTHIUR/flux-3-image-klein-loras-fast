import os
import gc
import json
import random
import base64
from io import BytesIO

import numpy as np
import spaces
import torch
from PIL import Image
from gradio import Server
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from diffusers import ModularPipeline
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Using device:", device)

# ── Mode / Resolution / Aspect configuration ──────────────────────────────────
# FLUX.3 Image Klein ships a Distilled variant that runs in 16 steps.
# Only Distilled is offered in this studio; Base at 2K/4K is intentionally not exposed.
DISTILLED = "Distilled (16 steps)"
DEFAULT_MODE = DISTILLED
DEFAULT_STEPS = 16
DEFAULT_GUIDANCE = 1.0

RESOLUTIONS = {
    "1K": 1024,
    "2K": 2048,
    "4K": 4096,
}
MODE_RESOLUTIONS = {
    DISTILLED: ["1K", "2K", "4K"],
}
DEFAULT_RESOLUTION = {
    DISTILLED: "2K",
}

ASPECT_RATIOS = {
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
}
AUTO = "Auto"
ASPECT_CHOICES = [AUTO] + list(ASPECT_RATIOS.keys())

# ── LoRA adapter registry ────────────────────────────────────────────────────
# Add new entries following the shape below. The frontend picks them up
# automatically from /api/config. The "Custom LoRA" entry is special — when
# selected, the frontend surfaces two text inputs (repo id + weight filename)
# and the backend loads it on-the-fly on the running instance.
ADAPTER_SPECS = {
    "None": {
        "repo": None,
        "weights": None,
        "adapter_name": None,
    },
    "360-Deg-LoRA": {
        "repo": "DenRakEiw/Flux_3_klein_360_Degree",
        "weights": "360_f3.safetensors",
        "adapter_name": "360-panorama",
    },
    "Font-Box": {
        "repo": "lilylilith/FLUX.3_Klein_FontBox",
        "weights": "F.3K_FontBox_v2.safetensors",
        "adapter_name": "font-box",
    },
    "Custom LoRA": {
        # Marker — repo/weights are supplied per-request via the API.
        "repo": None,
        "weights": None,
        "adapter_name": "custom-lora",
        "is_custom": True,
    },
}
LOADED_ADAPTERS: set = set()
ADAPTER_NAMES = list(ADAPTER_SPECS.keys())

# Cache of previously-loaded Custom LoRAs, keyed by (repo, weights).
# Avoids re-downloading the same custom adapter on every run.
CUSTOM_LORA_CACHE: dict = {}

# ── Examples ─────────────────────────────────────────────────────────────────
# Each example: {"images": ["path/1.jpg", ...], "prompt": "...", "lora": "None"}
EXAMPLES_CONFIG = [
    {
        "images": ["examples/C.jpg"],
        "prompt": "Change the weather to stormy.",
        "lora": "None",
    },
    {
        "images": ["examples/2.jpg"],
        "prompt": (
            "Relight the image to remove all existing lighting conditions and "
            "replace them with neutral, uniform illumination. Apply soft, evenly "
            "distributed lighting with no directional shadows, no harsh highlights, "
            "and no dramatic contrast. Maintain the original identity of all "
            "subjects exactly, preserving facial structure, skin tone, proportions, "
            "expressions, hair, clothing, and textures. Do not alter pose, camera "
            "angle, background geometry, or image composition. Lighting should "
            "appear balanced and studio-neutral, similar to diffuse overcast or a "
            "soft lightbox setup. Ensure consistent exposure across the entire "
            "image with realistic depth and subtle shading only where necessary "
            "for form."
        ),
        "lora": "None",
    },
    {
        "images": ["examples/1.jpg", "examples/2.jpg"],
        "prompt": (
            "Transfer the background from Image 1 to Image 2. Preserve the subject "
            "in Image 2 exactly, including identity, facial features, skin tone, "
            "hair, clothing, pose, proportions, and details. Replace only the "
            "background of Image 2 with the background from Image 1. Match "
            "perspective, scale, lighting, color temperature, shadows, and depth "
            "of field for a seamless, photorealistic composite. Keep the subject "
            "in Image 2 unchanged and ensure natural edges and consistent blending."
        ),
        "lora": "None",
    },
    {
        "images": ["examples/1.jpg"],
        "prompt": (
            "cinematic polaroid with soft grain subtle vignette gentle lighting "
            "white frame handwritten photographed by prithivMLmods preserving "
            "realistic texture and details"
        ),
        "lora": "None",
    },
    {
        "images": ["examples/cloth.jpg"],
        "prompt": (
            "Remove all visible wrinkles, creases, and folds from the cloth to "
            "achieve a smooth, neat, and professionally finished appearance. "
            "Preserve the original garment design, fabric texture, color, "
            "stitching, seams, shape, proportions, and details. Maintain realistic "
            "fabric material and natural shading. Do not change the garment style, "
            "fit, or structure, and avoid an artificial or plastic-like appearance."
        ),
        "lora": "None",
    },
]

# ── Load FLUX.3 Image Klein base model (ModularPipeline) ────────────────────
print("Loading FLUX.3 Image Klein (ModularPipeline)…")
pipe = ModularPipeline.from_pretrained("black-forest-labs/flux-3-image-klein")
pipe.load_components(dtype=torch.bfloat16)
pipe.to(device)
print("Base model loaded successfully.")

# ── Load FLUX.3 Image Klein Pup (Qwen3.5 prompt upsampler) ──────────────────
print("Loading Pup (Qwen3.5 prompt upsampler)…")
PUP_ID = "black-forest-labs/flux-3-image-klein-pup"
pup_processor = AutoProcessor.from_pretrained(PUP_ID)
upsampler = Qwen3_5ForConditionalGeneration.from_pretrained(
    PUP_ID, dtype=torch.bfloat16, device_map="auto"
).eval()
pup_system_prompts = json.load(open(hf_hub_download(PUP_ID, "system_prompts.json")))
print("Pup loaded successfully.")


def upsample_prompt(prompt: str, images=()) -> str:
    """Run the Qwen3.5 Pup upsampler on the user's prompt (with optional references)."""
    if images:
        text = (
            "Reference images follow in order (the first image, the second image, ...).\n"
            f"User request: {prompt}\nOutput the caption."
        )
        content = (
            [{"type": "image", "image": image} for image in images]
            + [{"type": "text", "text": text}]
        )
        messages = [
            {"role": "system", "content": pup_system_prompts["i2i"]},
            {"role": "user", "content": content},
        ]
    else:
        messages = [
            {"role": "system", "content": pup_system_prompts["t2i"]},
            {"role": "user", "content": prompt},
        ]
    inputs = pup_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    # Move inputs to the upsampler's actual parameter device (works with device_map="auto").
    try:
        target_device = next(upsampler.parameters()).device
    except StopIteration:
        target_device = device
    inputs = inputs.to(target_device)
    with torch.no_grad():
        output = upsampler.generate(**inputs, max_new_tokens=3584, do_sample=False)
    return pup_processor.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def compute_dimensions(aspect_choice: str, resolution_choice: str, input_image=None):
    """Compute (width, height) divisible by 16 from aspect + resolution.

    The resolution defines the longest side. When aspect is "Auto", the input
    image's aspect ratio is used (falls back to 1:1 when there is no input).
    """
    if resolution_choice not in RESOLUTIONS:
        resolution_choice = DEFAULT_RESOLUTION[DISTILLED]
    target_long = RESOLUTIONS[resolution_choice]

    if aspect_choice == AUTO:
        if input_image is not None:
            w, h = input_image.size
            aspect = (w / h) if h > 0 else 1.0
        else:
            aspect = 1.0
    else:
        aspect = ASPECT_RATIOS.get(aspect_choice, 1.0)

    if aspect >= 1.0:
        width = target_long
        height = int(round(target_long / aspect))
    else:
        height = target_long
        width = int(round(target_long * aspect))

    # FLUX requires dimensions divisible by 16
    width = max((width // 16) * 16, 16)
    height = max((height // 16) * 16, 16)
    return width, height


# ── Image helpers ────────────────────────────────────────────────────────────
def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""


def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""


def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGB"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images


def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Client config ────────────────────────────────────────────────────────────
def build_client_config():
    """Static config consumed by the frontend: LoRA list + examples + output options."""
    examples = []
    for i, ex in enumerate(EXAMPLES_CONFIG):
        examples.append({
            "idx": i,
            "thumbs": [make_thumb_b64(p) for p in ex["images"]],
            "n_images": len(ex["images"]),
            "lora": ex.get("lora", "None"),
            "prompt": ex["prompt"],
        })
    return {
        "loras": ADAPTER_NAMES,
        "default_lora": "None",
        "custom_lora_key": "Custom LoRA",
        "examples": examples,
        "modes": list(MODE_RESOLUTIONS.keys()),
        "default_mode": DEFAULT_MODE,
        "mode_resolutions": MODE_RESOLUTIONS,
        "default_resolution": DEFAULT_RESOLUTION,
        "aspects": ASPECT_CHOICES,
        "default_aspect": AUTO,
        "default_steps": DEFAULT_STEPS,
        "default_guidance": DEFAULT_GUIDANCE,
        "default_upsample": False,
    }


print("Building client config…")
CLIENT_CONFIG = build_client_config()
print(
    f"Built config: {len(EXAMPLES_CONFIG)} examples, {len(ADAPTER_NAMES)} LoRAs, "
    f"mode={DEFAULT_MODE}, default resolution={DEFAULT_RESOLUTION[DISTILLED]}."
)


# ── Gradio Server (FastAPI + Gradio queue/API engine) ────────────────────────
app = Server(title="flux-3-image-klein-lora-studio")


@app.mcp.tool(name="edit_image")
@app.api(name="edit_image")
@spaces.GPU(size="xlarge")
def infer(
    images_b64_json: str,
    prompt: str,
    lora_adapter: str,
    aspect: str,
    resolution: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    steps: int,
    upsample: bool = False,
    custom_lora_repo: str = "",
    custom_lora_weights: str = "",
) -> dict:
    """Edit one or more images with FLUX.3 Image Klein (Distilled, 16-step default).

    When ``lora_adapter == "Custom LoRA"``, the LoRA at ``custom_lora_repo``
    (e.g. ``user/repo``) is downloaded and loaded on the running instance.
    ``custom_lora_weights`` is the optional weight file name/path inside the
    repo (e.g. ``pytorch_lora_weights.safetensors`` or
    ``subdir/file.safetensors``); if left empty, the loader auto-detects.

    Returns:
        {"image": <base64 PNG data URL>, "seed": <seed used>,
         "caption": <prompt used by the model>, "width": int, "height": int,
         "lora": <adapter actually used>}
    """
    gc.collect()
    torch.cuda.empty_cache()

    pil_images = b64_to_pil_list(images_b64_json)
    has_images = len(pil_images) > 0

    if not prompt or prompt.strip() == "":
        raise Exception("Please enter an edit prompt.")

    spec = ADAPTER_SPECS.get(lora_adapter, ADAPTER_SPECS["None"])
    is_custom = bool(spec.get("is_custom"))

    # ── LoRA handling ────────────────────────────────────────────────────────
    if is_custom:
        # Custom LoRA path: load whatever the user supplied on this instance.
        repo = (custom_lora_repo or "").strip()
        weights = (custom_lora_weights or "").strip()
        if not repo:
            raise Exception(
                "Custom LoRA selected — please enter a repo id "
                "(e.g. user/repo) in the inspector."
            )
        cache_key = (repo, weights or "*auto*")
        if cache_key in CUSTOM_LORA_CACHE:
            adapter_name = CUSTOM_LORA_CACHE[cache_key]
            print(f"Custom LoRA cache hit: {repo} / '{weights}' -> {adapter_name}")
        else:
            adapter_name = "custom-lora-" + str(len(CUSTOM_LORA_CACHE))
            print(f"Loading Custom LoRA: repo={repo}, weights='{weights or '(auto)'}', adapter={adapter_name}")
            try:
                load_kwargs = {"adapter_name": adapter_name}
                if weights:
                    load_kwargs["weight_name"] = weights
                pipe.load_lora_weights(repo, **load_kwargs)
                CUSTOM_LORA_CACHE[cache_key] = adapter_name
                LOADED_ADAPTERS.add(adapter_name)
            except Exception as e:
                raise Exception(f"Failed to load Custom LoRA '{repo}': {e}")
        try:
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters([adapter_name], adapter_weights=[1.0])
        except Exception as e:
            print(f"set_adapters warning: {e}")
        used_lora_label = f"Custom: {repo}" + (f" / {weights}" if weights else "")
    elif spec["adapter_name"] is None:
        # "None" — disable any active LoRA.
        try:
            if hasattr(pipe, "disable_lora"):
                pipe.disable_lora()
        except Exception as e:
            print(f"disable_lora warning: {e}")
        used_lora_label = "None"
    else:
        # Pre-registered LoRA (e.g. 360-Deg-LoRA).
        adapter_name = spec["adapter_name"]
        if adapter_name not in LOADED_ADAPTERS:
            try:
                pipe.load_lora_weights(
                    spec["repo"],
                    weight_name=spec["weights"],
                    adapter_name=adapter_name,
                )
                LOADED_ADAPTERS.add(adapter_name)
            except Exception as e:
                raise Exception(f"Failed to load adapter {lora_adapter}: {e}")
        try:
            if hasattr(pipe, "set_adapters"):
                pipe.set_adapters([adapter_name], adapter_weights=[1.0])
        except Exception as e:
            print(f"set_adapters warning: {e}")
        used_lora_label = lora_adapter

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)

    # ── Compute target dimensions from aspect + resolution ──────────────────
    input_for_dim = pil_images[0] if has_images else None
    width, height = compute_dimensions(aspect, resolution, input_for_dim)

    processed_images = [
        img.resize((width, height), LANCZOS).convert("RGB")
        for img in pil_images
    ]

    # ── Prompt upsampling via Pup ───────────────────────────────────────────
    if upsample:
        try:
            caption = upsample_prompt(prompt, tuple(processed_images))
        except Exception as e:
            print(f"Pup upsampler failed, using raw prompt: {e}")
            caption = prompt
    else:
        caption = prompt

    try:
        kwargs = dict(
            prompt=caption,
            generator=generator,
            output="images",
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
        )
        if has_images:
            image_input = (
                processed_images if len(processed_images) > 1 else processed_images[0]
            )
            kwargs["image"] = image_input
        else:
            kwargs["width"] = width
            kwargs["height"] = height

        result = pipe(**kwargs)

        # ModularPipeline with output="images" returns a list; fall back to .images[0]
        if isinstance(result, (list, tuple)):
            result_image = result[0]
        elif hasattr(result, "images") and result.images:
            result_image = result.images[0]
        else:
            result_image = result

        return {
            "image": pil_to_b64_png(result_image),
            "seed": seed,
            "caption": caption,
            "width": width,
            "height": height,
            "lora": used_lora_label,
        }
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Return base64-encoded example images + prompt + LoRA for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"images": [], "prompt": "", "lora": "", "names": [], "status": "error"}
    ex = EXAMPLES_CONFIG[i]
    b64_list, names = [], []
    for path in ex["images"]:
        b64 = encode_full_image(path)
        if b64:
            b64_list.append(b64)
            names.append(os.path.basename(path))
    return {
        "images": b64_list,
        "prompt": ex["prompt"],
        "lora": ex.get("lora", "None"),
        "names": names,
        "status": "ok",
    }


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: LoRA choices + example cards + output options for the frontend."""
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)
