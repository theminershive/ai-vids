#!/usr/bin/env python3

"""
Full video-production pipeline (adapted to load content filters from an external file)
"""

import json
import logging
import os
import subprocess
import re
import time
from pathlib import Path
from typing import Optional

# ——— Logging configuration —————————————————————————————————————
LOG_LEVEL = os.getenv("BIBLEREAD_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Suppress PIL debug logs
logging.getLogger("PIL").setLevel(logging.WARNING)

# ——— Load moderation rules from external file —————————————————————————————
try:
    with open("content_filters.txt", "r", encoding="utf-8") as f:
        FILTER_KEYWORDS = json.load(f)
    logging.info("Loaded content filters from content_filters.txt")
except Exception as e:
    logging.error(f"Failed to load content filters: {e}")
    FILTER_KEYWORDS = {}

# Import adapted visuals functions
from visuals import get_model_by_name, generate_image_once, download_file
from ytuploader import upload as upload_youtube
from fbupload import upload as upload_facebook
from igupload import upload as upload_instagram

try:
    import captions
except Exception:
    captions = None

# ——— Directories —————————————————————————————————————————————
READY_DIR       = Path("ready")
VISUALS_DIR     = Path("visuals")
FINAL_VIDEO_DIR = Path("final")
for d in (READY_DIR, VISUALS_DIR, FINAL_VIDEO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ——— Helper: sanitize visual prompts ——————————————————————————————————
def sanitize_visual_prompt(prompt: str) -> str:
    sanitized = prompt
    for pattern, replacement in FILTER_KEYWORDS.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized

# ——— Generate and download images using visuals.py —————————————————————————
def generate_and_download_images(script: dict) -> dict:
    model = get_model_by_name(
        script.get("settings", {}).get("image_generation_style", "")
    )
    logging.info(
        "STATUS visuals: model=%s size=%sx%s backend=%s",
        model.get("name"),
        model.get("width"),
        model.get("height"),
        os.getenv("VISUAL_BACKEND", "leonardo").strip().lower(),
    )
    total_segments = sum(len(s.get("segments", [])) for s in script.get("sections", []))
    logging.info("STATUS visuals: total_segments=%s", total_segments)
    for section in script.get("sections", []):
        for seg in section.get("segments", []):
            seg_id = f"sec{section.get('section_number', 0)}_seg{seg.get('segment_number', 0)}"
            raw_prompt = seg.get("visual", {}).get("prompt", "")
            if not raw_prompt:
                logging.info("STATUS visuals: skip segment %s (no prompt)", seg_id)
                continue
            prompt = sanitize_visual_prompt(raw_prompt)
            logging.debug("PROMPT raw %s: %s", seg_id, raw_prompt.replace("\n", " \\n "))
            logging.info("PROMPT %s: %s", seg_id, prompt.replace("\n", " \\n "))
            start = time.perf_counter()
            url = generate_image_once(prompt, model)
            elapsed = time.perf_counter() - start
            logging.info("STATUS visuals: generated %s in %.1fs url=%s", seg_id, elapsed, url)
            img_file = VISUALS_DIR / f"sec{section.get('section_number',0)}_seg{seg['segment_number']}.png"
            download_file(url, img_file)
            logging.info("STATUS visuals: saved %s -> %s", seg_id, img_file)
            seg["visual"]["image_path"] = str(img_file)
    return script

# ——— Create captions via Whisper —————————————————————————————————————
def create_captions(video_path: str) -> Optional[list]:
    if captions is None:
        logging.info("Captions module not available; skipping captioning.")
        return None
    try:
        logging.info("STATUS captions: extracting audio from %s", video_path)
        audio_temp = captions.extract_audio(video_path)
        logging.info("STATUS captions: transcribing %s", audio_temp)
        transcription = captions.transcribe_audio_whisper(audio_temp)
        caps = captions.generate_captions_from_whisper(transcription)
        Path(audio_temp).unlink(missing_ok=True)
        logging.info("STATUS captions: generated %s caption entries", len(caps))
        return caps
    except Exception as exc:
        logging.warning(f"Captioning failed: {exc}")
        return None

# ——— Main pipeline —————————————————————————————————————————————
def main():
    logging.info("STATUS pipeline=start")
    # 1. Generate assembler JSON
    logging.info("=== 1. Generating assembler JSON (readasmb) ===")
    logging.info("STATUS step=readasmb start")
    start = time.perf_counter()
    subprocess.run(["python3", "readasmb.py"], check=True)
    logging.info("STATUS step=readasmb done elapsed=%.1fs", time.perf_counter() - start)

    # 2. Pick latest ready JSON
    ready_files = sorted(READY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not ready_files:
        logging.error("No assembler JSON found in ./ready")
        return
    script_path = ready_files[-1]
    logging.info(f"Using assembler JSON ➜ {script_path.name}")
    logging.info("STATUS assembler_json=%s", script_path)
    script = json.loads(script_path.read_text(encoding="utf-8"))

    # 3. Visual generation
    logging.info("=== 2. Visual generation ===")
    logging.info("STATUS step=visuals start")
    start = time.perf_counter()
    script = generate_and_download_images(script)
    script_path.write_text(json.dumps(script, indent=4), encoding="utf-8")
    logging.info("STATUS step=visuals done elapsed=%.1fs", time.perf_counter() - start)

    # 4. Video assembly
    logging.info("=== 3. Video assembly ===")
    logging.info("STATUS step=assemble start")
    start = time.perf_counter()
    subprocess.run(["python3", "video_assemble.py", str(script_path)], check=True)
    logging.info("STATUS step=assemble done elapsed=%.1fs", time.perf_counter() - start)
    assembled = json.loads(script_path.read_text(encoding="utf-8"))
    final_vid = Path(assembled.get("final_video", ""))
    if not final_vid.exists():
        logging.error(f"video_assemble did not produce expected file: {final_vid}")
        return
    logging.info("STATUS assembled_video=%s", final_vid)

    # 5. Whisper captioning
    logging.info("=== 4. Whisper captioning ===")
    logging.info("STATUS step=captions start")
    start = time.perf_counter()
    cap_vid = final_vid.with_name(final_vid.stem + "_cap.mp4")
    caps = create_captions(str(final_vid))
    if caps:
        try:
            captions.add_captions_to_video(
                input_video_path=str(final_vid),
                transcription=caps,
                output_video_path=str(cap_vid),
            )
        except Exception as exc:
            logging.warning(f"Caption overlay failed: {exc}")
    if not cap_vid.exists():
        cap_vid = final_vid
    logging.info("STATUS step=captions done elapsed=%.1fs output=%s", time.perf_counter() - start, cap_vid)

    # 6. Overlay text
    logging.info("=== 5. Overlay text ===")
    logging.info("STATUS step=overlay start")
    start = time.perf_counter()
    out_path = FINAL_VIDEO_DIR / f"{script_path.stem}_final.mp4"
    status = subprocess.run([
        "python3", "overlay.py",
        "--json", str(script_path),
        "--video", str(cap_vid),
        "--output", str(out_path),
    ])
    if status.returncode != 0:
        logging.error("overlay.py failed")
        return
    logging.info("STATUS step=overlay done elapsed=%.1fs", time.perf_counter() - start)
    logging.info(f"Overlay complete ➜ {out_path}")

    # 7. Upload sequence
    #logging.info("=== 6. Uploading ===")
    #try:
    #    from oauth_get2 import refresh_token
    #    refresh_token()
    #    logging.info("OAuth refresh: SUCCESS")
    #except Exception as e:
    #    logging.error(f"OAuth refresh failed: {e}")

    #for uploader, name in [(upload_youtube, "YouTube"), (upload_facebook, "Facebook"), (upload_instagram, "Instagram")]:
    #    try:
    #        result = uploader(str(script_path))
    #        logging.info(f"{name} upload ✓ {result}")
    #    except Exception as exc:
    #        logging.error(f"{name} upload failed: {exc}")

    logging.info("=== Pipeline finished successfully ===")
    logging.info("STATUS pipeline=success")

if __name__ == "__main__":
    main()
