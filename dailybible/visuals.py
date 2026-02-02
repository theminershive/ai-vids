import os
import json
import time
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
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

_image_backend_env = os.getenv("IMAGE_BACKEND", "").strip().lower()
_visual_backend_env = os.getenv("VISUAL_BACKEND", "comfyui").strip().lower()
if _image_backend_env in ("flux", "comfyui"):
    VISUAL_BACKEND = "comfyui"
elif _image_backend_env == "leonardo":
    VISUAL_BACKEND = "leonardo"
else:
    VISUAL_BACKEND = _visual_backend_env  # comfyui | leonardo

# Leonardo vars
LEONARDO_API_KEY = os.getenv("LEONARDO_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LOCAL_PROMPT_REWRITE_URL = os.getenv("LOCAL_PROMPT_REWRITE_URL", "http://192.168.1.176:11434/api/generate").strip()
LOCAL_PROMPT_REWRITE_MODEL = os.getenv("LOCAL_PROMPT_REWRITE_MODEL", "qwen2.5:14b-instruct-q4_K_M").strip()
PROMPT_REWRITE_SHORTEN = os.getenv("PROMPT_REWRITE_SHORTEN", "0").strip().lower() in ("1", "true", "yes", "y")
PROMPT_REWRITE_WORD_MIN = int(os.getenv("PROMPT_REWRITE_WORD_MIN", "40"))
PROMPT_REWRITE_WORD_MAX = int(os.getenv("PROMPT_REWRITE_WORD_MAX", "60"))
LEONARDO_API_ENDPOINT = "https://cloud.leonardo.ai/api/rest/v1"

# ComfyUI vars
COMFYUI_BASE_URL = os.getenv("COMFYUI_BASE_URL", "http://192.168.1.176:8188").rstrip("/")
COMFYUI_TIMEOUT_S = int(os.getenv("COMFYUI_TIMEOUT_S", "1800"))
COMFYUI_REQUEST_TIMEOUT_S = int(os.getenv("COMFYUI_REQUEST_TIMEOUT_S", os.getenv("IMAGE_REQUEST_TIMEOUT_S", "900")))
COMFYUI_POLL_S = float(os.getenv("COMFYUI_POLL_S", "2.0"))
COMFYUI_WIDTH = os.getenv("COMFYUI_WIDTH", "").strip()
COMFYUI_HEIGHT = os.getenv("COMFYUI_HEIGHT", "").strip()
COMFYUI_STEPS = os.getenv("COMFYUI_STEPS", "").strip()
COMFYUI_LORA_NAME = os.getenv("COMFYUI_LORA_NAME", "Flux_2-Turbo-LoRA_comfyui.safetensors").strip()
COMFYUI_WORKFLOW_JSON = os.getenv("COMFYUI_WORKFLOW_JSON", "").strip()
COMFYUI_REMAP_IDS = os.getenv("COMFYUI_REMAP_IDS", "1").strip().lower() in ("1", "true", "yes", "y")
COMFYUI_NORMALIZE_PROMPT = os.getenv("COMFYUI_NORMALIZE_PROMPT", "0").strip().lower() in ("1", "true", "yes", "y")
DEBUG_COMFY_API = os.getenv("DEBUG_COMFY_API", "0").strip().lower() in ("1", "true", "yes", "y")
ENFORCE_WORKFLOW_JSON = os.getenv("ENFORCE_WORKFLOW_JSON", "0").strip().lower() in ("1", "true", "yes", "y")
ENFORCED_WORKFLOW_PATH = os.getenv("ENFORCED_WORKFLOW_PATH", "./dailybible/workflow.json").strip()
ENFORCED_PROMPT_NODE_ID = os.getenv("ENFORCED_PROMPT_NODE_ID", "68:6").strip()
ENFORCED_PROMPT_FIELD = os.getenv("ENFORCED_PROMPT_FIELD", "text").strip()

# Optional sizing + upscale toggles
COMFYUI_ALLOW_SMALL = os.getenv("COMFYUI_ALLOW_SMALL", "0").strip().lower() in ("1", "true", "yes", "y")
UPSCALE_ENABLED = os.getenv("UPSCALE_ENABLED", "0").strip().lower() in ("1", "true", "yes", "y")
UPSCALE_WORKFLOW_JSON = os.getenv("UPSCALE_WORKFLOW_JSON", "/home/trilobyte/ai/mod/upscale.json").strip()
UPSCALE_COMFYUI_BASE_URL = os.getenv("UPSCALE_COMFYUI_BASE_URL", "http://192.168.1.94:8188").rstrip("/")
UPSCALE_TIMEOUT_S = int(os.getenv("UPSCALE_TIMEOUT_S", "1800"))
IMAGE_REQUEST_TIMEOUT_S = int(os.getenv("IMAGE_REQUEST_TIMEOUT_S", "900"))
IMAGE_DOWNLOAD_TIMEOUT_S = int(os.getenv("IMAGE_DOWNLOAD_TIMEOUT_S", "900"))
LEONARDO_REQUEST_TIMEOUT_S = int(os.getenv("LEONARDO_REQUEST_TIMEOUT_S", str(IMAGE_REQUEST_TIMEOUT_S)))
LEONARDO_POLL_ATTEMPTS = int(os.getenv("LEONARDO_POLL_ATTEMPTS", "90"))
LEONARDO_POLL_INTERVAL_S = float(os.getenv("LEONARDO_POLL_INTERVAL_S", "10"))

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

# ---------------------------------------------------------------------------- #
# ComfyUI API debug helpers                                                     #
# ---------------------------------------------------------------------------- #
_COMFY_DEBUG_LOGGER = None

def _ensure_comfy_debug_logger() -> logging.Logger | None:
    global _COMFY_DEBUG_LOGGER
    if not DEBUG_COMFY_API:
        return None
    if _COMFY_DEBUG_LOGGER:
        return _COMFY_DEBUG_LOGGER
    logger = logging.getLogger("comfy_api_debug")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            logs_dir / "comfy_api_debug.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        )
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    _COMFY_DEBUG_LOGGER = logger
    return logger

def _comfy_debug(msg: str, payload: dict | list | None = None) -> None:
    if not DEBUG_COMFY_API:
        return
    logger = _ensure_comfy_debug_logger()
    prefix = "[COMFY_API_DEBUG]"
    if payload is None:
        print(f"{prefix} {msg}")
        if logger:
            logger.info(msg)
        return
    pretty = json.dumps(payload, indent=2, sort_keys=True)
    print(f"{prefix} {msg}:\n{pretty}")
    if logger:
        logger.info("%s:\n%s", msg, pretty)

def _comfy_dump_workflow(workflow: dict) -> None:
    if not DEBUG_COMFY_API:
        return
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / "last_api_workflow.json"
    path.write_text(json.dumps(workflow, indent=2, sort_keys=True), encoding="utf-8")

def _comfy_new_debug_ctx(section_idx: int) -> dict:
    return {
        "section_idx": section_idx,
        "request_received_ts": datetime.now().isoformat(timespec="seconds"),
        "submit_start": None,
        "submit_end": None,
        "poll_start": None,
        "first_exec_ts": None,
        "poll_end": None,
        "download_start": None,
        "download_end": None,
        "prompt_id": None,
        "submit_response": None,
        "last_poll_response": None,
    }

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
    """
    Prefer local Ollama-style prompt rewrite endpoint. Falls back to legacy OpenAI rewrite if configured.
    """
    if not original_prompt:
        return original_prompt

    # Local rewrite (preferred)
    try:
        shorten_note = ""
        if PROMPT_REWRITE_SHORTEN:
            shorten_note = (
                f" Also target about {PROMPT_REWRITE_WORD_MIN}-{PROMPT_REWRITE_WORD_MAX} words total, "
                "including any negative clauses like 'no text, no writing, no watermark, no logo'."
            )
        payload = {
            "model": LOCAL_PROMPT_REWRITE_MODEL,
            "prompt": (
                "Rewrite this image prompt to be safe for moderation while preserving the scene and details."
                f"{shorten_note} Return only the rewritten prompt text.\n\n"
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

    # Legacy OpenAI fallback (if configured)
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
                    {
                        "role": "user",
                        "content": (
                            "Rewrite this image prompt to be safe"
                            + (
                                f" and about {PROMPT_REWRITE_WORD_MIN}-{PROMPT_REWRITE_WORD_MAX} words total, "
                                "including negative clauses like 'no text, no writing, no watermark, no logo'"
                                if PROMPT_REWRITE_SHORTEN
                                else ""
                            )
                            + f":\n{original_prompt}"
                        ),
                    },
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
    """
    Requirements:
      - Remove leading 'Vivid scene for:' if present.
      - Add a 'No Text' part to the prompt.
    """
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

    # Append no-text guidance (kept as a separate paragraph)
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
        resp = requests.get(url_or_path, stream=True, timeout=IMAGE_DOWNLOAD_TIMEOUT_S)
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
def generate_image_once(prompt: str, model: dict) -> str:
    """Submit one generation, poll until complete, return image URL."""
    if not LEONARDO_API_KEY:
        raise RuntimeError("LEONARDO_API_KEY missing but VISUAL_BACKEND=leonardo")
    payload = {
        "prompt": prompt,
        "modelId": model["id"],
        "width": model["width"],
        "height": model["height"],
        "num_images": model.get("num_images", 1),
        "negative_prompt": model.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
    }
    url = f"{LEONARDO_API_ENDPOINT}/generations"
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=LEONARDO_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    gen_id = data.get("generations_by_pk", {}).get("id") or data.get("sdGenerationJob", {}).get("generationId")
    if not gen_id:
        raise RuntimeError("No generation ID in response")
    logging.info(f"Generation started: {gen_id}")

    for _ in range(LEONARDO_POLL_ATTEMPTS):
        time.sleep(LEONARDO_POLL_INTERVAL_S)
        poll = requests.get(f"{LEONARDO_API_ENDPOINT}/generations/{gen_id}", headers=HEADERS, timeout=LEONARDO_REQUEST_TIMEOUT_S)
        poll.raise_for_status()
        st = poll.json()
        status = (
            st.get("status")
            or st.get("generations_by_pk", {}).get("status")
            or st.get("sdGenerationJob", {}).get("status")
        )
        if status and status.lower() == "complete":
            imgs = st.get("generations_by_pk", {}).get("generated_images") or []
            if imgs:
                return imgs[0].get("url")
            url_fb = st.get("sdGenerationJob", {}).get("imageUrl")
            if url_fb:
                return url_fb
            break
    raise RuntimeError("Image generation timeout or no URL returned")
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
        timeout=LEONARDO_REQUEST_TIMEOUT_S
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
    wait_time = LEONARDO_POLL_INTERVAL_S if wait_time is None else wait_time
    for attempt in range(1, LEONARDO_POLL_ATTEMPTS + 1):
        try:
            resp = requests.get(
                f"{LEONARDO_API_ENDPOINT}/generations/{generation_id}",
                headers=HEADERS,
                timeout=LEONARDO_REQUEST_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json() if resp.content else {}

            status = (
                data.get("status")
                or data.get("generations_by_pk", {}).get("status")
                or data.get("sdGenerationJob", {}).get("status", "")
            ).lower()

            logging.info(f"Leonardo poll {attempt}/{LEONARDO_POLL_ATTEMPTS}: {status}")

            if status == "complete":
                return data
            if status == "failed":
                raise RuntimeError("Leonardo generation failed.")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Leonardo poll error attempt {attempt}: {e}")

        delay = wait_time + (attempt * 0.5)
        time.sleep(delay)

    raise RuntimeError(f"Leonardo generation timed out after {LEONARDO_POLL_ATTEMPTS} polling attempts.")

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

_WORKFLOW_CACHE = None

def _load_workflow_template() -> dict:
    global _WORKFLOW_CACHE
    if _WORKFLOW_CACHE is not None:
        return json.loads(json.dumps(_WORKFLOW_CACHE))
    if COMFYUI_WORKFLOW_JSON and os.path.exists(COMFYUI_WORKFLOW_JSON):
        with open(COMFYUI_WORKFLOW_JSON, "r", encoding="utf-8") as f:
            _WORKFLOW_CACHE = json.load(f)
    else:
        _WORKFLOW_CACHE = COMFYUI_WORKFLOW_TEMPLATE
    return json.loads(json.dumps(_WORKFLOW_CACHE))

def _find_first_node(workflow: dict, class_type: str) -> str | None:
    for k, node in workflow.items():
        if node.get("class_type") == class_type:
            return k
    return None

def build_workflow(prompt_text: str, section_idx: int) -> dict:
    """
    Build a workflow for ComfyUI submission.
    - ENFORCE_WORKFLOW_JSON=1: load exact JSON and ONLY patch prompt text.
    - ENFORCE_WORKFLOW_JSON=0: use dynamic workflow builder (backwards compatible).
    """
    if ENFORCE_WORKFLOW_JSON:
        path = Path(ENFORCED_WORKFLOW_PATH)
        if not path.exists():
            raise RuntimeError(f"ENFORCED_WORKFLOW_PATH not found: {path}")
        workflow = json.loads(path.read_text(encoding="utf-8"))
        node = workflow.get(ENFORCED_PROMPT_NODE_ID)
        if not node:
            raise RuntimeError(
                f"ENFORCED_PROMPT_NODE_ID not found: {ENFORCED_PROMPT_NODE_ID}"
            )
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            raise RuntimeError(
                f"ENFORCED_PROMPT_NODE_ID inputs missing for node {ENFORCED_PROMPT_NODE_ID}"
            )
        if ENFORCED_PROMPT_FIELD not in inputs:
            raise RuntimeError(
                f"ENFORCED_PROMPT_FIELD not found: {ENFORCED_PROMPT_NODE_ID}.inputs.{ENFORCED_PROMPT_FIELD}"
            )
        inputs[ENFORCED_PROMPT_FIELD] = prompt_text
        if DEBUG_COMFY_API:
            _comfy_debug("workflow_mode ENFORCED_WORKFLOW_JSON")
            _comfy_debug(f"comfy_base_url {COMFYUI_BASE_URL}")
            _comfy_debug(
                f"patched_node {ENFORCED_PROMPT_NODE_ID}.inputs.{ENFORCED_PROMPT_FIELD}"
            )
            _comfy_debug(f"final_prompt {prompt_text}")
            _comfy_dump_workflow(workflow)
        return workflow

    wf = comfyui_build_workflow(prompt_text, section_idx)
    if DEBUG_COMFY_API:
        _comfy_debug("workflow_mode DYNAMIC_WORKFLOW")
        _comfy_debug(f"comfy_base_url {COMFYUI_BASE_URL}")
        _comfy_debug(f"final_prompt {prompt_text}")
        _comfy_dump_workflow(wf)
    return wf

def _remap_node_ids_for_comfyui_api(workflow: dict) -> dict:
    """
    ComfyUI /prompt typically expects node ids to be simple numeric strings.
    Your workflow uses ids like "68:10". This function remaps:
      - keys "68:10" -> "10"
      - any references ["68:10", 0] -> ["10", 0]
    If a key has no ":" it is kept as-is.
    """
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
    wf = _load_workflow_template()

    final_prompt = normalize_flux2_prompt(prompt) if COMFYUI_NORMALIZE_PROMPT else prompt
    logging.info(f"[COMFYUI] Section {section_idx} prompt: {final_prompt}")
    text_node = _find_first_node(wf, "CLIPTextEncode")
    if text_node:
        wf[text_node].setdefault("inputs", {})
        wf[text_node]["inputs"]["text"] = final_prompt

    # Force the lora name to one ComfyUI actually has
    lora_node = _find_first_node(wf, "LoraLoader")
    if lora_node:
        wf[lora_node].setdefault("inputs", {})
        wf[lora_node]["inputs"]["lora_name"] = COMFYUI_LORA_NAME

    save_node = _find_first_node(wf, "SaveImage")
    if save_node:
        wf[save_node].setdefault("inputs", {})
        wf[save_node]["inputs"]["filename_prefix"] = f"Flux2_Turbo_section_{section_idx}"

    noise_node = _find_first_node(wf, "RandomNoise")
    if noise_node:
        wf[noise_node].setdefault("inputs", {})
        wf[noise_node]["inputs"]["noise_seed"] = random.randint(1, 2_000_000_000)

    # Optional size overrides from .env (enforce min height unless COMFYUI_ALLOW_SMALL=1)
    def _parse_int(val: str) -> int | None:
        if not val:
            return None
        if val.isdigit():
            return int(val)
        digits = "".join(ch for ch in val if ch.isdigit())
        return int(digits) if digits else None

    w = _parse_int(COMFYUI_WIDTH)
    h = _parse_int(COMFYUI_HEIGHT)
    steps = _parse_int(COMFYUI_STEPS)
    if w:
        latent_node = _find_first_node(wf, "EmptyFlux2LatentImage")
        sched_node = _find_first_node(wf, "Flux2Scheduler")
        if latent_node:
            wf[latent_node].setdefault("inputs", {})
            wf[latent_node]["inputs"]["width"] = w
        if sched_node:
            wf[sched_node].setdefault("inputs", {})
            wf[sched_node]["inputs"]["width"] = w
    if h:
        if not COMFYUI_ALLOW_SMALL and h < 1024:
            h = 1024
        latent_node = _find_first_node(wf, "EmptyFlux2LatentImage")
        sched_node = _find_first_node(wf, "Flux2Scheduler")
        if latent_node:
            wf[latent_node].setdefault("inputs", {})
            wf[latent_node]["inputs"]["height"] = h
        if sched_node:
            wf[sched_node].setdefault("inputs", {})
            wf[sched_node]["inputs"]["height"] = h

    if steps:
        sched_node = _find_first_node(wf, "Flux2Scheduler")
        if sched_node:
            wf[sched_node].setdefault("inputs", {})
            wf[sched_node]["inputs"]["steps"] = steps

    # Ensure scheduler width/height match latent if both exist
    latent_node = _find_first_node(wf, "EmptyFlux2LatentImage")
    sched_node = _find_first_node(wf, "Flux2Scheduler")
    if latent_node and sched_node:
        lw = wf[latent_node].get("inputs", {}).get("width")
        lh = wf[latent_node].get("inputs", {}).get("height")
        if lw:
            wf[sched_node].setdefault("inputs", {})
            wf[sched_node]["inputs"]["width"] = lw
        if lh:
            wf[sched_node].setdefault("inputs", {})
            wf[sched_node]["inputs"]["height"] = lh

    log_w = wf.get(latent_node, {}).get("inputs", {}).get("width") if latent_node else None
    log_h = wf.get(latent_node, {}).get("inputs", {}).get("height") if latent_node else None
    log_steps = wf.get(sched_node, {}).get("inputs", {}).get("steps") if sched_node else None
    logging.info(
        f"[COMFYUI] Section {section_idx} settings: "
        f"width={log_w} height={log_h} steps={log_steps} "
        f"lora={COMFYUI_LORA_NAME} base_url={COMFYUI_BASE_URL}"
    )

    # Remap ids like "68:10" -> "10" so /prompt accepts it
    if COMFYUI_REMAP_IDS:
        wf = _remap_node_ids_for_comfyui_api(wf)
    return wf

def comfyui_queue_prompt(workflow: dict, base_url: str | None = None, debug_ctx: dict | None = None) -> str:
    url = f"{(base_url or COMFYUI_BASE_URL)}/prompt"
    payload = {"prompt": workflow}
    if debug_ctx is not None and DEBUG_COMFY_API:
        req_bytes = len(json.dumps(payload).encode("utf-8"))
        _comfy_debug(
            f"request_size_bytes section={debug_ctx.get('section_idx')} size={req_bytes}"
        )
    if debug_ctx is not None:
        debug_ctx["submit_start"] = time.time()
    resp = requests.post(url, json=payload, timeout=COMFYUI_REQUEST_TIMEOUT_S)

    if resp.status_code >= 400:
        logging.error(f"ComfyUI /prompt error {resp.status_code}: {resp.text}")
        resp.raise_for_status()

    data = resp.json()
    if debug_ctx is not None:
        debug_ctx["submit_end"] = time.time()
        debug_ctx["submit_response"] = data
        _comfy_debug(f"prompt_submit_response section={debug_ctx.get('section_idx')}", data)
    prompt_id = data.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"No prompt_id returned from ComfyUI: {data}")
    if debug_ctx is not None:
        debug_ctx["prompt_id"] = prompt_id
        _comfy_debug(f"prompt_id_received section={debug_ctx.get('section_idx')} prompt_id={prompt_id}")
    return prompt_id

def comfyui_poll_history(prompt_id: str, base_url: str | None = None, timeout_s: int | None = None, debug_ctx: dict | None = None) -> dict:
    url = f"{(base_url or COMFYUI_BASE_URL)}/history/{prompt_id}"
    deadline = time.time() + (timeout_s or COMFYUI_TIMEOUT_S)
    if debug_ctx is not None and debug_ctx.get("poll_start") is None:
        debug_ctx["poll_start"] = time.time()
        _comfy_debug(
            f"poll_start section={debug_ctx.get('section_idx')} prompt_id={prompt_id} ts={datetime.now().isoformat(timespec='seconds')}"
        )

    while time.time() < deadline:
        resp = requests.get(url, timeout=COMFYUI_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json() or {}
        if debug_ctx is not None:
            debug_ctx["last_poll_response"] = data
            _comfy_debug(
                f"poll_response section={debug_ctx.get('section_idx')} prompt_id={prompt_id}",
                data,
            )
        job = data.get(prompt_id) or {}
        outputs = job.get("outputs") or {}
        if debug_ctx is not None and debug_ctx.get("first_exec_ts") is None:
            status = (job.get("status") or {}).get("status_str")
            if status and status not in ("queued", "pending"):
                debug_ctx["first_exec_ts"] = time.time()
        if outputs:
            if debug_ctx is not None:
                debug_ctx["poll_end"] = time.time()
                _comfy_debug(
                    f"poll_complete section={debug_ctx.get('section_idx')} prompt_id={prompt_id} ts={datetime.now().isoformat(timespec='seconds')}"
                )
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

def comfyui_download_view(file_info: dict, out_path: str, debug_ctx: dict | None = None):
    filename = file_info.get("filename")
    subfolder = file_info.get("subfolder", "")
    ftype = file_info.get("type", "output")

    if not filename:
        raise RuntimeError(f"Missing filename in file_info: {file_info}")

    url = f"{COMFYUI_BASE_URL}/view"
    params = {"filename": filename, "subfolder": subfolder, "type": ftype}
    if debug_ctx is not None:
        debug_ctx["download_start"] = time.time()
    resp = requests.get(url, params=params, stream=True, timeout=IMAGE_DOWNLOAD_TIMEOUT_S)
    resp.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(1024 * 64):
            if chunk:
                f.write(chunk)

    if debug_ctx is not None:
        debug_ctx["download_end"] = time.time()
    logging.info(f"Downloaded ComfyUI image -> {out_path}")

def _comfyui_upload_image(image_path: str) -> str:
    url = f"{UPSCALE_COMFYUI_BASE_URL}/upload/image"
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "application/octet-stream")}
        resp = requests.post(url, files=files, timeout=COMFYUI_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json() if resp.content else {}
    filename = data.get("name") or data.get("filename")
    if not filename:
        raise RuntimeError(f"ComfyUI upload returned no filename: {data}")
    return filename

def _load_upscale_workflow() -> dict:
    if not UPSCALE_WORKFLOW_JSON or not os.path.exists(UPSCALE_WORKFLOW_JSON):
        raise FileNotFoundError(f"Upscale workflow JSON not found: {UPSCALE_WORKFLOW_JSON}")
    with open(UPSCALE_WORKFLOW_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def upscale_image_comfyui(input_path: str, out_dir: str, section_idx: int) -> str:
    workflow = _load_upscale_workflow()

    # Find LoadImage node and set uploaded filename
    load_node_key = None
    for k, node in workflow.items():
        if node.get("class_type") == "LoadImage":
            load_node_key = k
            break
    if not load_node_key:
        raise RuntimeError("Upscale workflow missing LoadImage node.")

    uploaded_name = _comfyui_upload_image(input_path)
    workflow[load_node_key].setdefault("inputs", {})
    workflow[load_node_key]["inputs"]["image"] = uploaded_name

    prompt_id = comfyui_queue_prompt(workflow, base_url=UPSCALE_COMFYUI_BASE_URL)
    logging.info(f"ComfyUI upscale queued prompt_id={prompt_id} section={section_idx}")
    job = comfyui_poll_history(prompt_id, base_url=UPSCALE_COMFYUI_BASE_URL, timeout_s=UPSCALE_TIMEOUT_S)
    file_info = comfyui_extract_first_image_file(job)
    if not file_info:
        raise RuntimeError(f"No images found in ComfyUI upscale outputs. job={job}")

    ext = os.path.splitext(file_info.get("filename", ""))[1] or ".png"
    out_path = os.path.join(out_dir, f"section_{section_idx}_upscaled{ext}")
    url = f"{UPSCALE_COMFYUI_BASE_URL}/view"
    params = {"filename": file_info.get("filename"), "subfolder": file_info.get("subfolder", ""), "type": file_info.get("type", "output")}
    resp = requests.get(url, params=params, stream=True, timeout=IMAGE_DOWNLOAD_TIMEOUT_S)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(1024 * 64):
            if chunk:
                f.write(chunk)
    return out_path


def generate_image_with_retry_comfyui(prompt: str, section_idx: int = 1) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            debug_ctx = _comfy_new_debug_ctx(section_idx) if DEBUG_COMFY_API else None
            if debug_ctx is not None:
                _comfy_debug(
                    f"request_received section={section_idx} ts={debug_ctx.get('request_received_ts')}"
                )
            wf = build_workflow(prompt, section_idx)
            if debug_ctx is not None:
                _comfy_debug(f"workflow_json section={section_idx}", wf)
            prompt_id = comfyui_queue_prompt(wf, debug_ctx=debug_ctx)
            logging.info(f"ComfyUI queued prompt_id={prompt_id} section={section_idx}")
            job = comfyui_poll_history(prompt_id, debug_ctx=debug_ctx)
            file_info = comfyui_extract_first_image_file(job)
            if not file_info:
                raise RuntimeError(f"No images found in ComfyUI history outputs. job={job}")

            ext = os.path.splitext(file_info.get("filename", ""))[1] or ".png"
            out_path = os.path.join(OUTPUT_DIR, f"section_{section_idx}{ext}")
            comfyui_download_view(file_info, out_path, debug_ctx=debug_ctx)
            if debug_ctx is not None:
                submit_start = debug_ctx.get("submit_start")
                submit_end = debug_ctx.get("submit_end")
                poll_start = debug_ctx.get("poll_start")
                poll_end = debug_ctx.get("poll_end")
                first_exec = debug_ctx.get("first_exec_ts") or poll_start or submit_end
                download_start = debug_ctx.get("download_start")
                download_end = debug_ctx.get("download_end")

                submit_s = (submit_end - submit_start) if submit_start and submit_end else 0.0
                queue_s = (first_exec - submit_end) if first_exec and submit_end else 0.0
                exec_s = (poll_end - first_exec) if poll_end and first_exec else 0.0
                download_s = (download_end - download_start) if download_end and download_start else 0.0
                total_s = (download_end - submit_start) if download_end and submit_start else 0.0

                _comfy_debug(
                    "job_timing_summary "
                    f"job_id={debug_ctx.get('prompt_id') or 'unknown'} "
                    f"submit={submit_s:.2f}s "
                    f"queue={queue_s:.2f}s "
                    f"exec={exec_s:.2f}s "
                    f"download={download_s:.2f}s "
                    f"total={total_s:.2f}s"
                )
                _comfy_debug(
                    f"job_timing_metrics job_id={debug_ctx.get('prompt_id') or 'unknown'}",
                    {
                        "request_submit_time": round(submit_s, 3),
                        "comfy_queue_wait_time": round(queue_s, 3),
                        "comfy_execution_time": round(exec_s, 3),
                        "result_download_time": round(download_s, 3),
                        "total_time": round(total_s, 3),
                    },
                )
            if UPSCALE_ENABLED:
                try:
                    return upscale_image_comfyui(out_path, OUTPUT_DIR, section_idx)
                except Exception as err:
                    logging.warning(f"Upscale failed, using base image. Error: {err}")
            return out_path

        except Exception as err:
            logging.error(f"ComfyUI attempt {attempt} error: {err}")
            time.sleep(RETRY_DELAY * attempt)

    logging.error(f"ComfyUI retries exhausted for section {section_idx}")
    return None

# ---------------------------------------------------------------------------- #
# BACKWARDS COMPATIBILITY SHIMS                                                 #
# ---------------------------------------------------------------------------- #
def generate_image_with_retry(prompt: str, model_config: dict = None):
    """
    Old code expects:
      gen_id, used_prompt = generate_image_with_retry(prompt, config)

    Leonardo mode:
      returns (generation_id, used_prompt)

    ComfyUI mode:
      returns (local_path, prompt) so old code won't crash.
    """
    backend = os.getenv("VISUAL_BACKEND", VISUAL_BACKEND).strip().lower()

    if backend == "comfyui":
        img_path = generate_image_with_retry_comfyui(prompt, section_idx=1)
        return img_path, prompt

    return generate_image_with_retry_leonardo(prompt, model_config)

def poll_generation_status(generation_id: str, wait_time: float = 10) -> dict:
    """
    Old code calls poll_generation_status(gen_id).
    If gen_id is a local file path, return a compatible dict.
    """
    if generation_id and isinstance(generation_id, str) and os.path.exists(generation_id):
        return {"status": "complete", "local_image_path": generation_id}

    return poll_generation_status_leonardo(generation_id, wait_time=wait_time)

def extract_image_url(generation_data: dict) -> str:
    """
    Old code calls extract_image_url(result) and expects a URL.
    For ComfyUI, we return the local file path.
    """
    if isinstance(generation_data, dict) and generation_data.get("local_image_path"):
        return generation_data["local_image_path"]
    return extract_image_url_leonardo(generation_data)

# ---------------------------------------------------------------------------- #
# MODERN helper (recommended for new code)                                      #
# ---------------------------------------------------------------------------- #
def generate_visual(prompt: str, section_idx: int = 1, style_name: str = None) -> str:
    """
    Returns a local file path in both modes.
    """
    backend = os.getenv("VISUAL_BACKEND", VISUAL_BACKEND).strip().lower()

    if backend == "comfyui":
        return generate_image_with_retry_comfyui(prompt, section_idx=section_idx)

    config = get_model_config_by_style(style_name)
    img_url = generate_image_once(prompt, config)
    ext = os.path.splitext(urlparse(img_url).path)[1] or ".jpg"
    out_path = os.path.join(OUTPUT_DIR, f"section_{section_idx}{ext}")
    download_content(img_url, out_path)
    if UPSCALE_ENABLED:
        try:
            return upscale_image_comfyui(out_path, OUTPUT_DIR, section_idx)
        except Exception as err:
            logging.warning(f"Upscale failed, using base image. Error: {err}")
    return out_path
