#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from config import AppConfig, ensure_dirs, load_config
from getfact import main as generate_fact
from image_backend import generate_image, save_image
from overlay import overlay_verses
from video_assemble import assemble_from_json

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def sanitize_visual_prompt(prompt: str) -> str:
    filter_keywords = {
        # Add regex replacements as needed
    }
    sanitized = prompt
    for pattern, replacement in filter_keywords.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def generate_and_download_images(config: AppConfig, script: dict, script_path: Path) -> dict:
    style = script.get("settings", {}).get("image_generation_style", "")
    for section in script.get("sections", []):
        for seg in section.get("segments", []):
            raw_prompt = seg.get("visual", {}).get("prompt", "")
            if not raw_prompt:
                continue
            prompt = sanitize_visual_prompt(raw_prompt)
            result = generate_image(prompt, config)
            img_file = config.paths.visuals_dir / f"sec{section.get('section_number',0)}_seg{seg['segment_number']}.png"
            save_image(result, img_file)
            seg["visual"]["image_path"] = str(img_file)
    return script


def pick_latest_ready(config: AppConfig) -> Optional[Path]:
    ready_files = sorted(config.paths.ready_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not ready_files:
        return None
    return ready_files[-1]


def main() -> None:
    config = load_config()
    ensure_dirs(config)

    logging.info("=== 1. Generating assembler JSON ===")
    try:
        generate_fact()
    except SystemExit:
        logging.warning("getfact.py failed; proceeding with latest ready JSON if available.")

    script_path = pick_latest_ready(config)
    if not script_path:
        logging.error("No assembler JSON found in %s", config.paths.ready_dir)
        return

    logging.info("Using assembler JSON -> %s", script_path.name)
    script = json.loads(script_path.read_text(encoding="utf-8"))

    logging.info("=== 2. Visual generation ===")
    script = generate_and_download_images(config, script, script_path)
    script_path.write_text(json.dumps(script, indent=4), encoding="utf-8")

    logging.info("=== 3. Video assembly ===")
    assembled_video = assemble_from_json(config, script_path)
    if not assembled_video.exists():
        logging.error("video_assemble did not produce expected file: %s", assembled_video)
        return

    logging.info("=== 4. Overlay text ===")
    out_path = config.paths.final_dir / f"{script_path.stem}_final.mp4"
    overlay_verses(config, assembled_video, script_path, out_path)
    logging.info("Overlay complete -> %s", out_path)

    logging.info("=== Pipeline finished successfully ===")


if __name__ == "__main__":
    main()
