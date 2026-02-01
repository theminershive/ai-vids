import os
import json
import time
import logging
import random
import shutil
import requests
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# Optional: only required for legacy OpenAI moderation rewrite
try:
    import openai
except Exception:
    openai = None

# ------------------- CONFIG -------------------
load_dotenv()

VISUAL_BACKEND = os.getenv("VISUAL_BACKEND", "comfyui").strip().lower()  # comfyui | leonardo

# Leonardo vars
LEONARDO_API_KEY = os.getenv("LEONARDO_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_PROMPT_REWRITE_URL = os.getenv("LOCAL_PROMPT_REWRITE_URL", "http://192.168.1.176:11434/api/generate").strip()
LOCAL_PROMPT_REWRITE_MODEL = os.getenv("LOCAL_PROMPT_REWRITE_MODEL", "qwen2.5:14b-instruct-q4_K_M").strip()
LEONARDO_API_ENDPOINT = "https://cloud.leonardo.ai/api/rest/v1"

# ComfyUI vars
COMFYUI_BASE_URL = os.getenv("COMFYUI_BASE_URL", "http://192.168.1.176:8188").rstrip("/")
COMFYUI_TIMEOUT_S = int(os.getenv("COMFYUI_TIMEOUT_S", "1800"))
COMFYUI_POLL_S = float(os.getenv("COMFYUI_POLL_S", "2.0"))
COMFYUI_WIDTH = os.getenv("COMFYUI_WIDTH", "").strip()
COMFYUI_HEIGHT = os.getenv("COMFYUI_HEIGHT", "").strip()
COMFYUI_LORA_NAME = os.getenv("COMFYUI_LORA_NAME", "Flux_2-Turbo-LoRA_comfyui.safetensors").strip()

# Output
OUTPUT_DIR = os.getenv("VISUAL_OUTPUT_DIR", "downloaded_content")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Retry configuration
MAX_RETRIES = int(os.getenv("VISUAL_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("VISUAL_RETRY_DELAY", "2"))

# Default negative prompt to avoid unwanted text in generated images (Leonardo only)
NEGATIVE_PROMPT_DEFAULT = os.getenv("NEGATIVE_PROMPT_DEFAULT", "text")

# Strong "no text" guidance appended to the POSITIVE prompt (for workflows without an explicit negative node)
NO_TEXT_POSITIVE_GUIDANCE = os.getenv(
    "NO_TEXT_POSITIVE_GUIDANCE",
    (
        "NO TEXT: no letters, no words, no typography, no logos, no watermarks, no signage, "
        "no captions, no UI overlays, no labels, no QR codes, no barcodes, no license plates, "
        "no posters, no newspaper, no book text."
    )
).strip()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("visuals.log")],
)

if VISUAL_BACKEND not in ("comfyui", "leonardo"):
    raise ValueError("VISUAL_BACKEND must be 'comfyui' or 'leonardo'")

# ------------------- Leonardo headers -------------------
AUTHORIZATION = f"Bearer {LEONARDO_API_KEY}" if LEONARDO_API_KEY else ""
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": AUTHORIZATION,
}

# ---------------------------------------------------------------------------- #
# Leonardo Model Configuration                                                  #
# ---------------------------------------------------------------------------- #

def get_model_config():
    return {
        "id": "de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3",
        "width": 576,
        "height": 1024,
        "num_images": 1,
        "alchemy": True,
        "enhancePrompt": False,
        "photoReal": False,
        "photoRealVersion": "",
        "presetStyle": "CINEMATIC",
        "negative_prompt": NEGATIVE_PROMPT_DEFAULT,
    }


def get_model_config_by_style(style_name=None):
    return get_model_config()

# ---------------------------------------------------------------------------- #
# Prompt rewriting (Leonardo moderation helper)                                 #
# ---------------------------------------------------------------------------- #

def rewrite_prompt(original_prompt: str) -> str:
    if not original_prompt:
        return original_prompt

    # Local rewrite (preferred)
    try:
        payload = {
            "model": LOCAL_PROMPT_REWRITE_MODEL,
            "prompt": (
                "Rewrite this image prompt to be safe for moderation while preserving the scene and details. "
                "Return only the rewritten prompt text.\n\n"
                f"{original_prompt}"
            ),
            "stream": False,
        }
        resp = requests.post(LOCAL_PROMPT_REWRITE_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        new_prompt = (data.get("response") or data.get("text") or data.get("message") or "").strip()
        if new_prompt:
            logging.info(f"Rewritten prompt (local): {new_prompt}")
            return new_prompt
        logging.warning("Local prompt rewrite returned empty response; using original.")
    except Exception as e:
        logging.warning(f"Local prompt rewrite failed, using original. Error: {e}")

    if openai and OPENAI_API_KEY:
        try:
            openai.api_key = OPENAI_API_KEY
            resp = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rewrite image prompts to remove or soften any content that might be blocked "
                            "by moderation, while preserving the core scene and details."
                        ),
                    },
                    {"role": "user", "content": f"Rewrite this image prompt to be safe:\n{original_prompt}"},
                ],
                max_tokens=200,
            )
            new_prompt = resp.choices[0].message.content.strip()
            logging.info(f"Rewritten prompt (openai): {new_prompt}")
            return new_prompt
        except Exception as e:
            logging.warning(f"OpenAI prompt rewrite failed, using original. Error: {e}")

    return original_prompt

# ---------------------------------------------------------------------------- #
# Prompt cleanup / augmentation for Flux2                                       #
# ---------------------------------------------------------------------------- #

def normalize_flux2_prompt(prompt: str) -> str:
    if not prompt:
        return prompt

    p = prompt.strip()
    lower = p.lower()

    prefixes = [
        "vivid scene for:",
        "vivid scene for :",
        "vivid scene:",
        "vivid scene :",
    ]
    for pref in prefixes:
        if lower.startswith(pref):
            p = p[len(pref):].strip()
            break

    if "no text" not in p.lower():
        p = f"{p}\n\n{NO_TEXT_POSITIVE_GUIDANCE}"

    return p

# ---------------------------------------------------------------------------- #
# SMART download helper: URL OR LOCAL PATH                                      #
# ---------------------------------------------------------------------------- #

def download_content(url_or_path: str, filename: str):
    parsed = urlparse(url_or_path)

    # Real URL (http/https)
    if parsed.scheme in ("http", "https"):
        resp = requests.get(url_or_path, stream=True, timeout=300)
        resp.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in resp.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)
        logging.info(f"Downloaded remote image -> {filename}")
        return filename

    # Local path
    src = Path(url_or_path)
    dst = Path(filename)
    if not src.exists():
        raise FileNotFoundError(f"Local image not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    logging.info(f"Copied local image -> {filename}")
    return filename

# ---------------------------------------------------------------------------- #
# Leonardo pipeline                                                             #
# ---------------------------------------------------------------------------- #

def generate_image_leonardo(prompt: str, model_config: dict = None) -> str:
    if not LEONARDO_API_KEY:
        raise RuntimeError("LEONARDO_API_KEY missing but VISUAL_BACKEND=leonardo")

    config = model_config if model_config is not None else get_model_config()
    payload = {
        "modelId": config["id"],
        "height": config["height"],
        "width": config["width"],
        "num_images": config["num_images"],
        "alchemy": config["alchemy"],
        "photoReal": config["photoReal"],
        "photoRealVersion": config["photoRealVersion"],
        "enhancePrompt": config["enhancePrompt"],
        "presetStyle": config["presetStyle"],
        "prompt": prompt,
        "negative_prompt": config.get("negative_prompt", ""),
    }
    logging.info(f"Leonardo requesting generation: {prompt}")
    resp = requests.post(
        f"{LEONARDO_API_ENDPOINT}/generations",
        json=payload,
        headers=HEADERS,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    gen = data.get("generations_by_pk") or data.get("sdGenerationJob")
    generation_id = gen.get("id") or gen.get("generationId")
    if not generation_id:
        raise RuntimeError(f"No generation ID returned: {data}")
    logging.info(f"Leonardo generation initiated: {generation_id}")
    return generation_id


def poll_generation_status_leonardo(generation_id: str, wait_time: float = 10) -> dict:
    for attempt in range(1, 31):
        try:
            resp = requests.get(
                f"{LEONARDO_API_ENDPOINT}/generations/{generation_id}",
                headers=HEADERS,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json() if resp.content else {}

            status = (
                data.get("status")
                or data.get("generations_by_pk", {}).get("status")
                or data.get("sdGenerationJob", {}).get("status", "")
            ).lower()

            logging.info(f"Leonardo poll {attempt}/30: {status}")

            if status == "complete":
                return data
            if status == "failed":
                raise RuntimeError("Leonardo generation failed.")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Leonardo poll error attempt {attempt}: {e}")

        delay = wait_time + (attempt * 0.5)
        time.sleep(delay)

    raise RuntimeError("Leonardo generation timed out after 30 polling attempts.")


def extract_image_url_leonardo(generation_data: dict) -> str:
    img_list = generation_data.get("generations_by_pk", {}).get("generated_images", []) or \
               generation_data.get("sdGenerationJob", {}).get("generated_images", [])
    if img_list:
        return img_list[0].get("url")
    return None


def generate_image_with_retry_leonardo(prompt: str, model_config: dict = None) -> (str, str):
    attempt_prompt = prompt
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            gen_id = generate_image_leonardo(attempt_prompt, model_config)
            return gen_id, attempt_prompt
        except requests.exceptions.HTTPError as http_err:
            code = getattr(http_err.response, "status_code", None)
            logging.warning(f"Leonardo HTTP {code} attempt {attempt}")
            if code == 403:
                attempt_prompt = rewrite_prompt(attempt_prompt)
        except Exception as err:
            logging.error(f"Leonardo attempt {attempt} error: {err}")
        time.sleep(RETRY_DELAY * attempt)

    logging.error(f"Leonardo retries exhausted for prompt: {prompt}")
    return None, prompt

# ---------------------------------------------------------------------------- #
# ComfyUI workflow (Flux2 Turbo LoRA)                                           #
# ---------------------------------------------------------------------------- #

COMFYUI_WORKFLOW_TEMPLATE = {
  "9": {
    "inputs": {
      "filename_prefix": "Flux2_Turbo",
      "images": ["68:8", 0]
    },
    "class_type": "SaveImage",
    "_meta": {"title": "Save Image"}
  },
  "68:10": {
    "inputs": {"vae_name": "flux2-vae.safetensors"},
    "class_type": "VAELoader",
    "_meta": {"title": "Load VAE"}
  },
  "68:12": {
    "inputs": {"unet_name": "flux2_dev_fp8mixed.safetensors", "weight_dtype": "default"},
    "class_type": "UNETLoader",
    "_meta": {"title": "Load Diffusion Model"}
  },
  "68:16": {
    "inputs": {"sampler_name": "euler"},
    "class_type": "KSamplerSelect",
    "_meta": {"title": "KSamplerSelect"}
  },
  "68:25": {
    "inputs": {"noise_seed": 649422536169327},
    "class_type": "RandomNoise",
    "_meta": {"title": "RandomNoise"}
  },
  "68:38": {
    "inputs": {"clip_name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2", "device": "default"},
    "class_type": "CLIPLoader",
    "_meta": {"title": "Load CLIP"}
  },
  "68:70": {
    "inputs": {
      "model": ["68:12", 0],
      "clip": ["68:38", 0],
      "lora_name": "Flux_2-Turbo-LoRA_comfyui.safetensors",
      "strength_model": 1.0,
      "strength_clip": 1.0
    },
    "class_type": "LoraLoader",
    "_meta": {"title": "Load LoRA (Turbo)"}
  },
  "68:6": {
    "inputs": {
      "text": "PROMPT_WILL_BE_INJECTED",
      "clip": ["68:70", 1]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
  },
  "68:26": {
    "inputs": {"guidance": 4, "conditioning": ["68:6", 0]},
    "class_type": "FluxGuidance",
    "_meta": {"title": "FluxGuidance"}
  },
  "68:22": {
    "inputs": {"model": ["68:70", 0], "conditioning": ["68:26", 0]},
    "class_type": "BasicGuider",
    "_meta": {"title": "BasicGuider"}
  },
  "68:47": {
    "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    "class_type": "EmptyFlux2LatentImage",
    "_meta": {"title": "Empty Flux 2 Latent"}
  },
  "68:48": {
    "inputs": {"steps": 10, "width": 1024, "height": 1024},
    "class_type": "Flux2Scheduler",
    "_meta": {"title": "Flux2Scheduler"}
  },
  "68:13": {
    "inputs": {
      "noise": ["68:25", 0],
      "guider": ["68:22", 0],
      "sampler": ["68:16", 0],
      "sigmas": ["68:48", 0],
      "latent_image": ["68:47", 0]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {"title": "SamplerCustomAdvanced"}
  },
  "68:8": {
    "inputs": {"samples": ["68:13", 0], "vae": ["68:10", 0]},
    "class_type": "VAEDecode",
    "_meta": {"title": "VAE Decode"}
  }
}


def _remap_node_ids_for_comfyui_api(workflow: dict) -> dict:
    mapping = {}
    for k in workflow.keys():
        if isinstance(k, str) and ":" in k:
            suffix = k.split(":")[-1]
            mapping[k] = suffix
        else:
            mapping[k] = k

    new_wf = {}
    for old_k, node in workflow.items():
        new_k = mapping[old_k]
        new_wf[new_k] = node

    def rewrite_value(v):
        if isinstance(v, list):
            if len(v) == 2 and isinstance(v[0], str) and v[0] in mapping and isinstance(v[1], int):
                return [mapping[v[0]], v[1]]
            return [rewrite_value(x) for x in v]
        if isinstance(v, dict):
            return {kk: rewrite_value(vv) for kk, vv in v.items()}
        return v

    return rewrite_value(new_wf)


def comfyui_build_workflow(prompt: str, section_idx: int) -> dict:
    wf = json.loads(json.dumps(COMFYUI_WORKFLOW_TEMPLATE))

    normalized = normalize_flux2_prompt(prompt)
    wf["68:6"]["inputs"]["text"] = normalized

    wf["68:70"]["inputs"]["lora_name"] = COMFYUI_LORA_NAME

    wf["9"]["inputs"]["filename_prefix"] = f"Flux2_Turbo_section_{section_idx}"
    wf["68:25"]["inputs"]["noise_seed"] = random.randint(1, 2_000_000_000)

    if COMFYUI_WIDTH.isdigit():
        w = int(COMFYUI_WIDTH)
        wf["68:47"]["inputs"]["width"] = w
        wf["68:48"]["inputs"]["width"] = w
    if COMFYUI_HEIGHT.isdigit():
        h = int(COMFYUI_HEIGHT)
        wf["68:47"]["inputs"]["height"] = h
        wf["68:48"]["inputs"]["height"] = h

    wf = _remap_node_ids_for_comfyui_api(wf)
    return wf


def comfyui_queue_prompt(workflow: dict) -> str:
    url = f"{COMFYUI_BASE_URL}/prompt"
    payload = {"prompt": workflow}
    resp = requests.post(url, json=payload, timeout=120)

    if resp.status_code >= 400:
        logging.error(f"ComfyUI /prompt error {resp.status_code}: {resp.text}")
        resp.raise_for_status()

    data = resp.json()
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id returned from ComfyUI: {data}")
    return prompt_id


def comfyui_poll_history(prompt_id: str) -> dict:
    url = f"{COMFYUI_BASE_URL}/history/{prompt_id}"
    deadline = time.time() + COMFYUI_TIMEOUT_S

    while time.time() < deadline:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json() or {}
        job = data.get(prompt_id) or {}
        outputs = job.get("outputs") or {}
        if outputs:
            return job
        time.sleep(COMFYUI_POLL_S)

    raise RuntimeError(f"ComfyUI job timed out (prompt_id={prompt_id})")


def comfyui_extract_first_image_file(history_job: dict) -> dict:
    outputs = history_job.get("outputs") or {}
    for _, node_out in outputs.items():
        imgs = node_out.get("images") or []
        if imgs:
            return imgs[0]
    return {}


def comfyui_download_view(file_info: dict, out_path: str):
    filename = file_info.get("filename")
    subfolder = file_info.get("subfolder", "")
    ftype = file_info.get("type", "output")

    if not filename:
        raise RuntimeError(f"Missing filename in file_info: {file_info}")

    url = f"{COMFYUI_BASE_URL}/view"
    params = {"filename": filename, "subfolder": subfolder, "type": ftype}
    resp = requests.get(url, params=params, stream=True, timeout=300)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(1024 * 64):
            if chunk:
                f.write(chunk)

    logging.info(f"Downloaded ComfyUI image -> {out_path}")


def generate_image_with_retry_comfyui(prompt: str, section_idx: int = 1) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            wf = comfyui_build_workflow(prompt, section_idx)
            prompt_id = comfyui_queue_prompt(wf)
            logging.info(f"ComfyUI queued prompt_id={prompt_id} section={section_idx}")
            job = comfyui_poll_history(prompt_id)
            file_info = comfyui_extract_first_image_file(job)
            if not file_info:
                raise RuntimeError(f"No images found in ComfyUI history outputs. job={job}")

            ext = os.path.splitext(file_info.get("filename", ""))[1] or ".png"
            out_path = os.path.join(OUTPUT_DIR, f"section_{section_idx}{ext}")
            comfyui_download_view(file_info, out_path)
            return out_path

        except Exception as err:
            logging.error(f"ComfyUI attempt {attempt} error: {err}")
            time.sleep(RETRY_DELAY * attempt)

    logging.error(f"ComfyUI retries exhausted for section {section_idx}")
    return None

# ---------------------------------------------------------------------------- #
# MODERN helper (recommended for new code)                                      #
# ---------------------------------------------------------------------------- #

def generate_visual(prompt: str, section_idx: int = 1, style_name: str = None) -> str:
    backend = VISUAL_BACKEND

    if backend == "comfyui":
        return generate_image_with_retry_comfyui(prompt, section_idx=section_idx)

    config = get_model_config_by_style(style_name)
    gen_id, _ = generate_image_with_retry_leonardo(prompt, config)
    if not gen_id:
        return None

    result = poll_generation_status_leonardo(gen_id)
    img_url = extract_image_url_leonardo(result)
    if not img_url:
        return None

    ext = os.path.splitext(urlparse(img_url).path)[1] or ".jpg"
    out_path = os.path.join(OUTPUT_DIR, f"section_{section_idx}{ext}")
    download_content(img_url, out_path)
    return out_path
