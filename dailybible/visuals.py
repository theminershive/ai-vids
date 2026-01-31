import os
import json
import time
import requests
import logging
import openai
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse

# ------------------- CONFIG -------------------
load_dotenv()

LEONARDO_API_KEY = os.getenv('LEONARDO_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not LEONARDO_API_KEY:
    logging.error("Leonardo API key not found. Please set LEONARDO_API_KEY in your .env file.")
    exit(1)
if not OPENAI_API_KEY:
    logging.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    exit(1)

openai.api_key = OPENAI_API_KEY

AUTHORIZATION = f"Bearer {LEONARDO_API_KEY}"
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": AUTHORIZATION
}
LEONARDO_API_ENDPOINT = "https://cloud.leonardo.ai/api/rest/v1"
OUTPUT_DIR = "downloaded_content"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, multiplied by attempt

# Default negative prompt to avoid unwanted text in generated images
NEGATIVE_PROMPT_DEFAULT = "text"

# ---------------------------------------------------------------------------- #
# Model Configuration (can be updated dynamically)                              #
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
        # Include negative_prompt for API payload
        "negative_prompt": NEGATIVE_PROMPT_DEFAULT
    }

def get_model_config_by_style(style_name=None):
    # Could adapt negative_prompt per style if needed
    return get_model_config()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("leonardo_downloader.log")
    ]
)

def rewrite_prompt(original_prompt: str) -> str:
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are an assistant that rewrites image prompts to remove or soften any content "
                    "that might be blocked by moderation, while preserving the core scene and details.")},
                {"role": "user", "content": f"Rewrite this image prompt to be safe:\n{original_prompt}"}
            ],
            max_tokens=200
        )
        new_prompt = resp.choices[0].message.content.strip()
        logging.info(f"Rewritten prompt: {new_prompt}")
        return new_prompt
    except Exception as e:
        logging.error(f"Failed to rewrite prompt: {e}")
        return original_prompt

def generate_image(prompt: str, model_config: dict = None) -> str:
    config = model_config if model_config is not None else get_model_config()
    payload = {
        "modelId": config['id'],
        "height": config['height'],
        "width": config['width'],
        "num_images": config['num_images'],
        "alchemy": config['alchemy'],
        "photoReal": config['photoReal'],
        "photoRealVersion": config['photoRealVersion'],
        "enhancePrompt": config['enhancePrompt'],
        "presetStyle": config['presetStyle'],
        "prompt": prompt,
        # Add negative_prompt field to prevent text in images
        "negative_prompt": config.get('negative_prompt', "")
    }
    # Print full payload so it's visible without timestamp prefixes
    print("Sending generation payload:")
    print(json.dumps(payload, indent=2))
    logging.info(f"Requesting generation for prompt: {prompt} with negative_prompt: {payload['negative_prompt']}")
    resp = requests.post(f"{LEONARDO_API_ENDPOINT}/generations", json=payload, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    gen = data.get('generations_by_pk') or data.get('sdGenerationJob')
    generation_id = gen.get('id') or gen.get('generationId')
    if not generation_id:
        raise RuntimeError(f"No generation ID returned: {data}")
    logging.info(f"Generation initiated: {generation_id}")
    return generation_id

# (rest of the code unchanged, polling, download, etc.)

def poll_generation_status(generation_id: str, wait_time: float = 10) -> dict:
    for attempt in range(1, 31):
        try:
            resp = requests.get(f"{LEONARDO_API_ENDPOINT}/generations/{generation_id}", headers=HEADERS)
            resp.raise_for_status()
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                logging.warning(f"JSON decode error on attempt {attempt}: {e}")
                time.sleep(wait_time)
                continue

            if not isinstance(data, dict):
                logging.warning(f"Invalid or None response on attempt {attempt}. Retrying...")
                time.sleep(wait_time)
                continue

            status = (
                data.get('status') or
                data.get('generations_by_pk', {}).get('status') or
                data.get('sdGenerationJob', {}).get('status', '')
            ).lower()

            logging.info(f"Polling {attempt}/30: {status}")

            if status == 'complete':
                return data
            if status == 'failed':
                raise RuntimeError("Generation failed on Leonardo side.")

        except requests.exceptions.RequestException as e:
            logging.warning(f"HTTP request failed on attempt {attempt}: {e}")

        delay = wait_time + (attempt * 0.5)
        logging.info(f"Waiting {delay:.1f}s before next poll...")
        time.sleep(delay)

    raise RuntimeError("Generation timed out after 30 polling attempts.")

def extract_image_url(generation_data: dict) -> str:
    img_list = generation_data.get('generations_by_pk', {}).get('generated_images', []) or \
               generation_data.get('sdGenerationJob', {}).get('generated_images', [])
    if img_list:
        return img_list[0].get('url')
    return None

def download_content(url: str, filename: str):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in resp.iter_content(1024):
            f.write(chunk)
    logging.info(f"Downloaded image to {filename}")

def generate_image_with_retry(prompt: str, model_config: dict = None) -> (str, str):
    attempt_prompt = prompt
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            gen_id = generate_image(attempt_prompt, model_config)
            return gen_id, attempt_prompt
        except requests.exceptions.HTTPError as http_err:
            code = http_err.response.status_code
            logging.warning(f"HTTP {code} on attempt {attempt} for prompt.")
            if code == 403:
                logging.info("Prompt flagged: rewriting and retrying.")
                attempt_prompt = rewrite_prompt(attempt_prompt)
        except Exception as err:
            logging.error(f"Attempt {attempt} error: {err}")
        delay = RETRY_DELAY * attempt
        logging.info(f"Retrying after {delay}s...")
        time.sleep(delay)
    logging.error(f"All retries exhausted for prompt: {prompt}")
    return None, prompt

def process_visuals(script_path: str, output_script_path: str = None, style_name: str = None) -> dict:
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        config = get_model_config_by_style(style_name)
        sections = data.get('sections', [])
        for idx, section in enumerate(sections, 1):
            vis = section.get('visual', {})
            prompt = vis.get('prompt')
            if not prompt:
                continue
            gen_id, used_prompt = generate_image_with_retry(prompt, config)
            if not gen_id:
                logging.error(f"Skipping section {idx}: no generation ID.")
                continue
            try:
                result = poll_generation_status(gen_id)
            except RuntimeError:
                logging.warning("Polling failed — trying rewritten prompt...")
                safe_prompt = rewrite_prompt(prompt)
                gen_id, _ = generate_image_with_retry(safe_prompt, config)
                if not gen_id:
                    logging.error(f"Polling failed again for section {idx}. Skipping.")
                    continue
                result = poll_generation_status(gen_id)
            img_url = extract_image_url(result)
            if not img_url:
                logging.error(f"No image URL in response for section {idx}.")
                continue
            ext = os.path.splitext(urlparse(img_url).path)[1] or '.jpg'
            out_path = os.path.join(OUTPUT_DIR, f'section_{idx}{ext}')
            download_content(img_url, out_path)
            section['visual']['image_path'] = out_path
        if output_script_path:
            with open(output_script_path, 'w', encoding='utf-8') as outf:
                json.dump(data, outf, indent=4)
        return data
    except Exception as e:
        logging.error(f"Error in process_visuals: {e}")
        return {}

if __name__ == '__main__':
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else 'video_script.json'
    outp = sys.argv[2] if len(sys.argv) > 2 else 'video_script_with_images.json'
    style = sys.argv[3] if len(sys.argv) > 3 else None
    process_visuals(inp, outp, style)
