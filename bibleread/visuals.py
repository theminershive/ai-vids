#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_visual_update.py

Perform one Leonardo.ai generation for the first section+segment prompt in an assembler JSON,
download the image, and update the JSON with the local image path.
"""
import os
import sys
import json
import time
import logging
import requests
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

# API configuration
API_KEY = os.getenv('LEONARDO_API_KEY')
if not API_KEY:
    logging.error("LEONARDO_API_KEY not set in environment")
    sys.exit(1)
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
    'accept': 'application/json'
}
LEO_ENDPOINT = 'https://cloud.leonardo.ai/api/rest/v1'

# Models (subset of your original list)
CUSTOM_MODELS = [
    { 'id': 'b2614463-296c-462a-9586-aafdb8f00e36', 'name': 'Leonardo Phoenix 1.0', 'width': 576, 'height': 1024 },
    { 'id': 'b2614463-296c-462a-9586-aafdb8f00e36', 'name': 'Flux Dev',         'width': 576, 'height': 1024 },
]
STYLE_ALIASES = { 'phoenix': 'Leonardo Phoenix 1.0', 'leonardo phoenix': 'Leonardo Phoenix 1.0' }


def get_model_by_name(name: str) -> dict:
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
    resp = requests.post(url, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()
    gen_id = data.get('generations_by_pk', {}).get('id') or data.get('sdGenerationJob', {}).get('generationId')
    if not gen_id:
        raise RuntimeError("No generation ID in response")
    logging.info(f"Generation started: {gen_id}")

    # Poll
    for i in range(30):
        time.sleep(2)
        poll = requests.get(f"{LEO_ENDPOINT}/generations/{gen_id}", headers=HEADERS)
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


def download_file(url: str, dest: Path):
    r = requests.get(url, stream=True)
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
