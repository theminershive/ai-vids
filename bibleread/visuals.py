#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visuals.py

Backend-agnostic image generation helpers. Supports:
- Leonardo (legacy)
- ComfyUI (local)
"""
import os
import sys
import json
import time
import logging
import requests
import shutil
import random
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse

# Setup logging
default_log = logging.StreamHandler()
logging.basicConfig(level=logging.INFO,
                    handlers=[default_log],
                    format='%(asctime)s [%(levelname)s] %(message)s')

# Load env
dotenv_path = Path(__file__).parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path)

LEO_ENDPOINT = 'https://cloud.leonardo.ai/api/rest/v1'
_image_backend_env = os.getenv('IMAGE_BACKEND', '').strip().lower()
_visual_backend_env = os.getenv('VISUAL_BACKEND', 'leonardo').strip().lower()
if _image_backend_env in ('flux', 'comfyui'):
    VISUAL_BACKEND = 'comfyui'
elif _image_backend_env == 'leonardo':
    VISUAL_BACKEND = 'leonardo'
else:
    VISUAL_BACKEND = _visual_backend_env

# ComfyUI settings
COMFYUI_BASE_URL = os.getenv('COMFYUI_BASE_URL', 'http://192.168.1.176:8188').rstrip('/')
COMFYUI_WIDTH = int(os.getenv('COMFYUI_WIDTH', '1080'))
COMFYUI_HEIGHT = int(os.getenv('COMFYUI_HEIGHT', '1920'))
COMFYUI_STEPS = int(os.getenv('COMFYUI_STEPS', '20'))
COMFYUI_CFG = float(os.getenv('COMFYUI_CFG', '7'))
COMFYUI_SAMPLER = os.getenv('COMFYUI_SAMPLER', 'dpmpp_2m')
COMFYUI_SCHEDULER = os.getenv('COMFYUI_SCHEDULER', 'karras')
COMFYUI_SEED = int(os.getenv('COMFYUI_SEED', '-1'))
COMFYUI_NEGATIVE_PROMPT = os.getenv(
    'COMFYUI_NEGATIVE_PROMPT',
    'text, watermark, signature, logo, blurry'
)
COMFYUI_FILENAME_PREFIX = os.getenv('COMFYUI_FILENAME_PREFIX', 'bibleread')
COMFYUI_LORA_NAME = os.getenv('COMFYUI_LORA_NAME', 'Flux_2-Turbo-LoRA_comfyui.safetensors').strip()
COMFYUI_TIMEOUT_S = int(os.getenv('COMFYUI_TIMEOUT_S', '1800'))
COMFYUI_POLL_S = float(os.getenv('COMFYUI_POLL_S', '2.0'))
COMFYUI_REQUEST_TIMEOUT_S = int(os.getenv('COMFYUI_REQUEST_TIMEOUT_S', os.getenv('IMAGE_REQUEST_TIMEOUT_S', '900')))
IMAGE_REQUEST_TIMEOUT_S = int(os.getenv('IMAGE_REQUEST_TIMEOUT_S', '900'))
IMAGE_DOWNLOAD_TIMEOUT_S = int(os.getenv('IMAGE_DOWNLOAD_TIMEOUT_S', '900'))
LEONARDO_REQUEST_TIMEOUT_S = int(os.getenv('LEONARDO_REQUEST_TIMEOUT_S', str(IMAGE_REQUEST_TIMEOUT_S)))
LEONARDO_POLL_ATTEMPTS = int(os.getenv('LEONARDO_POLL_ATTEMPTS', '90'))
LEONARDO_POLL_INTERVAL_S = float(os.getenv('LEONARDO_POLL_INTERVAL_S', '10'))

# Strong "no text" guidance appended to the POSITIVE prompt (for workflows without an explicit negative node)
NO_TEXT_POSITIVE_GUIDANCE = os.getenv(
    'NO_TEXT_POSITIVE_GUIDANCE',
    (
        'NO TEXT: no letters, no words, no typography, no logos, no watermarks, no signage, '
        'no captions, no UI overlays, no labels, no QR codes, no barcodes, no license plates, '
        'no posters, no newspaper, no book text.'
    )
).strip()


def _get_headers() -> dict:
    api_key = os.getenv('LEONARDO_API_KEY')
    if not api_key:
        raise RuntimeError("LEONARDO_API_KEY not set in environment")
    return {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
        'accept': 'application/json'
    }

# Models (subset of your original list)
CUSTOM_MODELS = [
    { 'id': 'b2614463-296c-462a-9586-aafdb8f00e36', 'name': 'Leonardo Phoenix 1.0', 'width': 576, 'height': 1024 },
    { 'id': 'b2614463-296c-462a-9586-aafdb8f00e36', 'name': 'Flux Dev',         'width': 576, 'height': 1024 },
]
STYLE_ALIASES = { 'phoenix': 'Leonardo Phoenix 1.0', 'leonardo phoenix': 'Leonardo Phoenix 1.0' }


def get_model_by_name(name: str) -> dict:
    if VISUAL_BACKEND == 'comfyui':
        return {
            'id': 'comfyui',
            'name': 'ComfyUI',
            'width': COMFYUI_WIDTH,
            'height': COMFYUI_HEIGHT,
        }
    # resolve alias
    key = name.strip().lower()
    name = STYLE_ALIASES.get(key, name)
    for m in CUSTOM_MODELS:
        if m['name'].lower() == name.lower():
            return m
    logging.warning(f"Style '{name}' not found, falling back to first model")
    return CUSTOM_MODELS[0]


def generate_image_once(prompt: str, model: dict) -> str:
    """Submit one generation, poll until complete, return image URL."""
    if VISUAL_BACKEND == 'comfyui':
        return _generate_comfyui_image(prompt, model)
    payload = {
        'prompt': prompt,
        'modelId': model['id'],
        'width': model['width'],
        'height': model['height'],
        'num_images': 1,
        'negative_prompt': "text, watermark, signature, logo, blurry"
    }
    # POST to /generations
    url = f"{LEO_ENDPOINT}/generations"
    headers = _get_headers()
    resp = requests.post(url, headers=headers, json=payload, timeout=LEONARDO_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()
    gen_id = data.get('generations_by_pk', {}).get('id') or data.get('sdGenerationJob', {}).get('generationId')
    if not gen_id:
        raise RuntimeError("No generation ID in response")
    logging.info(f"Generation started: {gen_id}")

    # Poll
    for i in range(LEONARDO_POLL_ATTEMPTS):
        time.sleep(LEONARDO_POLL_INTERVAL_S)
        poll = requests.get(
            f"{LEO_ENDPOINT}/generations/{gen_id}",
            headers=headers,
            timeout=LEONARDO_REQUEST_TIMEOUT_S,
        )
        poll.raise_for_status()
        st = poll.json()
        status = (st.get('status') or
                  st.get('generations_by_pk', {}).get('status') or
                  st.get('sdGenerationJob', {}).get('status'))
        if status and status.lower() == 'complete':
            # extract URL
            imgs = st.get('generations_by_pk', {}).get('generated_images') or []
            if imgs:
                return imgs[0].get('url')
            # fallback
            url_fb = st.get('sdGenerationJob', {}).get('imageUrl')
            if url_fb:
                return url_fb
            break
    raise RuntimeError("Image generation timeout or no URL returned")


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
        'vivid scene for:',
        'vivid scene for :',
        'vivid scene:',
        'vivid scene :',
    ]
    for pref in prefixes:
        if lower.startswith(pref):
            p = p[len(pref):].strip()
            break

    if 'no text' not in p.lower():
        p = f"{p}\n\n{NO_TEXT_POSITIVE_GUIDANCE}"

    return p


# ComfyUI workflow (Flux2 Turbo LoRA)
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
            mapping[k] = k.split(":")[-1]
        else:
            mapping[k] = k

    new_wf = {}
    for old_k, node in workflow.items():
        new_wf[mapping[old_k]] = node

    def rewrite_value(v):
        if isinstance(v, list):
            if len(v) == 2 and isinstance(v[0], str) and v[0] in mapping and isinstance(v[1], int):
                return [mapping[v[0]], v[1]]
            return [rewrite_value(x) for x in v]
        if isinstance(v, dict):
            return {kk: rewrite_value(vv) for kk, vv in v.items()}
        return v

    return rewrite_value(new_wf)


def comfyui_build_workflow(prompt: str) -> dict:
    wf = json.loads(json.dumps(COMFYUI_WORKFLOW_TEMPLATE))
    wf["68:6"]["inputs"]["text"] = normalize_flux2_prompt(prompt)
    wf["68:70"]["inputs"]["lora_name"] = COMFYUI_LORA_NAME
    wf["9"]["inputs"]["filename_prefix"] = COMFYUI_FILENAME_PREFIX
    wf["68:25"]["inputs"]["noise_seed"] = random.randint(1, 2_000_000_000)

    if str(COMFYUI_WIDTH).isdigit():
        wf["68:47"]["inputs"]["width"] = int(COMFYUI_WIDTH)
        wf["68:48"]["inputs"]["width"] = int(COMFYUI_WIDTH)
    if str(COMFYUI_HEIGHT).isdigit():
        wf["68:47"]["inputs"]["height"] = int(COMFYUI_HEIGHT)
        wf["68:48"]["inputs"]["height"] = int(COMFYUI_HEIGHT)

    return _remap_node_ids_for_comfyui_api(wf)


def comfyui_queue_prompt(workflow: dict) -> str:
    resp = requests.post(
        f"{COMFYUI_BASE_URL}/prompt",
        json={"prompt": workflow},
        timeout=COMFYUI_REQUEST_TIMEOUT_S,
    )
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
        resp = requests.get(url, timeout=COMFYUI_REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json() or {}
        job = data.get(prompt_id) or {}
        if job.get("outputs"):
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


def _generate_comfyui_image(prompt: str, model: dict) -> str:
    workflow = comfyui_build_workflow(prompt)
    prompt_id = comfyui_queue_prompt(workflow)
    job = comfyui_poll_history(prompt_id)
    image = comfyui_extract_first_image_file(job)
    filename = image.get("filename")
    subfolder = image.get("subfolder", "")
    image_type = image.get("type", "output")
    if not filename:
        raise RuntimeError("ComfyUI generation returned no image filename.")
    return f"{COMFYUI_BASE_URL}/view?filename={filename}&subfolder={subfolder}&type={image_type}"


def download_file(url: str, dest: Path):
    if url.startswith("file://"):
        src = Path(url[7:])
        shutil.copyfile(src, dest)
        logging.info(f"Saved image to {dest}")
        return
    if Path(url).is_file():
        shutil.copyfile(Path(url), dest)
        logging.info(f"Saved image to {dest}")
        return
    r = requests.get(url, stream=True, timeout=IMAGE_DOWNLOAD_TIMEOUT_S)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    logging.info(f"Saved image to {dest}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python single_visual_update.py PATH_TO_JSON [OUTPUT_DIR]")
        sys.exit(1)
    json_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2] if len(sys.argv)>2 else Path('downloaded_content'))
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(json_path.read_text(encoding='utf-8'))
    section = data['sections'][0]
    seg = section['segments'][0]
    prompt = seg['visual']['prompt']
    style = data.get('settings', {}).get('image_generation_style', CUSTOM_MODELS[0]['name'])

    model = get_model_by_name(style)
    logging.info(f"Using model {model['name']} ({model['width']}x{model['height']})")
    img_url = generate_image_once(prompt, model)
    logging.info(f"Image URL: {img_url}")

    # Build filename
    ext = os.path.splitext(urlparse(img_url).path)[1] or '.png'
    fname = out_dir / f"section1_segment1{ext}"
    download_file(img_url, fname)

    # Update JSON
    rel = os.path.relpath(fname, json_path.parent)
    seg['visual']['image_path'] = rel
    json_path.write_text(json.dumps(data, indent=2))
    logging.info(f"Updated JSON: {json_path} -> image_path set to {rel}")

if __name__ == '__main__':
    main()
