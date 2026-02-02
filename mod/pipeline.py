#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from config import AppConfig, ensure_dirs, load_config
from getfact import main as generate_fact
from image_backend import generate_image, save_image
from overlay import overlay_verses
from ytuploader import upload as upload_youtube
from fbupload import upload as upload_facebook
from igupload import upload as upload_instagram
from video_assemble import assemble_from_json

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent
MONITOR_DIR = BASE_DIR.parent / "monitor"
if MONITOR_DIR.exists():
    sys.path.insert(0, str(MONITOR_DIR))
try:
    import monitoring  # type: ignore
    monitor = monitoring.get_monitor("mod", base_dir=BASE_DIR)
except Exception:
    monitor = None


def sanitize_visual_prompt(prompt: str) -> str:
    filter_keywords = {
        # Add regex replacements as needed
    }
    sanitized = prompt
    for pattern, replacement in filter_keywords.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized

def _extract_run_number(stem: str) -> int | None:
    match = re.match(r"^(\\d+)", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


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
    if monitor:
        monitor.stage_start("assemble")
    try:
        generate_fact()
        if monitor:
            monitor.stage_end("assemble", ok=True)
    except SystemExit:
        logging.warning("getfact.py failed; proceeding with latest ready JSON if available.")
        if monitor:
            monitor.stage_end("assemble", ok=False, error="getfact_failed")

    script_path = pick_latest_ready(config)
    if not script_path:
        logging.error("No assembler JSON found in %s", config.paths.ready_dir)
        if monitor:
            monitor.run_error("no_ready_json")
        return

    logging.info("Using assembler JSON -> %s", script_path.name)
    script = json.loads(script_path.read_text(encoding="utf-8"))
    topic = script.get("topic") or script.get("title") or script.get("reference") or script.get("headline")
    run_number = _extract_run_number(script_path.stem)
    if monitor:
        monitor.run_start(
            run_id=script_path.stem,
            topic=topic,
            script_path=str(script_path.resolve()),
            meta={
                "ready_name": script_path.name,
                "section_count": len(script.get("sections", [])),
                "reference": script.get("reference"),
                "run_number": run_number,
            },
        )

    logging.info("=== 2. Visual generation ===")
    if monitor:
        monitor.stage_start("visuals")
    try:
        script = generate_and_download_images(config, script, script_path)
        script_path.write_text(json.dumps(script, indent=4), encoding="utf-8")
        if monitor:
            monitor.stage_end("visuals", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("visuals", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise

    logging.info("=== 3. Video assembly ===")
    if monitor:
        monitor.stage_start("assemble_video")
    try:
        assembled_video = assemble_from_json(config, script_path)
        if not assembled_video.exists():
            logging.error("video_assemble did not produce expected file: %s", assembled_video)
            if monitor:
                monitor.stage_end("assemble_video", ok=False, error="missing_final_video")
                monitor.run_error("missing_final_video")
            return
        if monitor:
            monitor.stage_end("assemble_video", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("assemble_video", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise

    logging.info("=== 4. Overlay text ===")
    out_path = config.paths.final_dir / f"{script_path.stem}_final.mp4"
    if monitor:
        monitor.stage_start("overlay")
    try:
        overlay_verses(config, assembled_video, script_path, out_path)
        if monitor:
            monitor.stage_end("overlay", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("overlay", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise
    logging.info("Overlay complete -> %s", out_path)

    logging.info("=== Pipeline finished successfully ===")
    if monitor:
        monitor.run_success(final_video=str(out_path.resolve()))

    logging.info("=== Uploading ===")
    if os.getenv("UPLOAD_YOUTUBE", "0").strip() == "1":
        try:
            yt_url = upload_youtube(str(script_path))
            logging.info("YouTube upload ✓ %s", yt_url)
        except Exception as exc:
            logging.error("YouTube upload failed: %s", exc)
    if os.getenv("UPLOAD_FACEBOOK", "0").strip() == "1":
        try:
            upload_facebook(str(script_path))
            logging.info("Facebook upload ✓")
        except Exception as exc:
            logging.error("Facebook upload failed: %s", exc)
    if os.getenv("UPLOAD_INSTAGRAM", "0").strip() == "1":
        try:
            upload_instagram(str(script_path))
            logging.info("Instagram upload ✓")
        except Exception as exc:
            logging.error("Instagram upload failed: %s", exc)


if __name__ == "__main__":
    main()
