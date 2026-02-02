#!/usr/bin/env python3
"""
Full video-production pipeline using topic-driven scripts and dailybible-style video assembly.
"""

import json
import logging
import os
import subprocess
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

from config import load_config, ensure_dirs
from visuals import generate_visual, rewrite_prompt
from tts import process_tts
from video_assembler import assemble_video
import captions
from ytuploader import upload as upload_youtube
from fbupload import upload as upload_facebook
from igupload import upload as upload_instagram

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
BASE_DIR = Path(__file__).parent.resolve()

MONITOR_DIR = BASE_DIR.parent / "monitor"
if MONITOR_DIR.exists():
    sys.path.insert(0, str(MONITOR_DIR))
try:
    import monitoring  # type: ignore
    monitor = monitoring.get_monitor("mod2", base_dir=BASE_DIR)
except Exception:
    monitor = None


def sanitize_visual_prompt(prompt: str) -> str:
    filter_keywords = {
        r"\bchild\b": "figure",
        r"\bchildren\b": "figures",
        r"\bflesh\b": "skin",
        r"\bkid\b": "person",
        r"\bminor\b": "individual",
        r"\bnaked\b": "modest",
        r"\bnakedness\b": "modestness",
        r"\bslain\b": "fallen",
        r"\bgore\b": "dramatic intensity",
    }
    out = prompt or ""
    for pattern, repl in filter_keywords.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _save_script(script_path: Path, script: dict) -> None:
    script_path.write_text(json.dumps(script, indent=4, ensure_ascii=False), encoding="utf-8")

def _extract_run_number(stem: str) -> int | None:
    match = re.match(r"^(\\d+)", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def generate_and_download_images(script: dict, visuals_dir: Path) -> dict:
    style = script.get("style", "") or script.get("image_style", "")
    sections = script.get("sections", [])
    for section_idx, section in enumerate(sections, start=1):
        segments = section.get("segments", []) or []
        seg0 = segments[0] if segments else {}

        prompt = ""
        if isinstance(seg0, dict):
            visual_seg = seg0.get("visual") or {}
            if isinstance(visual_seg, dict):
                prompt = (visual_seg.get("prompt") or "").strip()

        if not prompt:
            visual_old = section.get("visual") or {}
            if isinstance(visual_old, dict):
                prompt = (visual_old.get("prompt") or "").strip()

        if not prompt:
            logging.warning("Section %s: missing visuals.prompt; skipping image generation.", section_idx)
            continue

        prompt = rewrite_prompt(prompt)
        prompt = sanitize_visual_prompt(prompt)

        logging.info("[VIS] Section %s: generating image...", section_idx)
        saved = generate_visual(prompt, section_idx=section_idx, style_name=style)
        if not saved:
            logging.error("[VIS] Section %s: generate_visual returned no image", section_idx)
            continue

        saved_path = Path(saved)
        out_path = visuals_dir / f"section_{section_idx}{saved_path.suffix or '.png'}"
        if saved_path.resolve() != out_path.resolve():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved_path, out_path)
            saved_path = out_path

        saved_abs = str(saved_path.resolve())

        if segments and isinstance(seg0, dict):
            seg0.setdefault("visual", {})
            if isinstance(seg0["visual"], dict):
                seg0["visual"]["image_path"] = saved_abs
            else:
                seg0["visual"] = {"image_path": saved_abs}
            segments[0] = seg0
            section["segments"] = segments
        else:
            section.setdefault("visual", {})
            if isinstance(section["visual"], dict):
                section["visual"]["image_path"] = saved_abs
            else:
                section["visual"] = {"image_path": saved_abs}

        logging.info("[VIS] Section %s: saved -> %s", section_idx, saved_abs)

    return script


def create_captions(video_path: str) -> Optional[list]:
    try:
        audio_temp = captions.extract_audio(video_path)
        transcription = captions.transcribe_audio(audio_temp)
        caps = captions.generate_captions_from_whisper(transcription)
        Path(audio_temp).unlink(missing_ok=True)
        return caps
    except Exception as exc:
        logging.warning("Captioning failed: %s", exc)
        return None


def main():
    config = load_config()
    ensure_dirs(config)

    logging.info("=== 1. assemble.py ===")
    if monitor:
        monitor.stage_start("assemble")
    try:
        subprocess.run(["python3", str(BASE_DIR / "assemble.py")], check=True)
        if monitor:
            monitor.stage_end("assemble", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("assemble", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise

    ready_files = sorted(config.paths.ready_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not ready_files:
        logging.error("No assembler JSON found in ./ready")
        if monitor:
            monitor.run_error("no_ready_json")
        return
    script_path = ready_files[-1]
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
        script = generate_and_download_images(script, config.paths.visuals_dir)
        _save_script(script_path, script)
        if monitor:
            monitor.stage_end("visuals", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("visuals", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise

    logging.info("=== 3. TTS generation ===")
    if monitor:
        monitor.stage_start("tts")
    try:
        script = process_tts(script, audio_dir=config.paths.audio_dir)
        _save_script(script_path, script)
        if monitor:
            monitor.stage_end("tts", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("tts", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise

    logging.info("=== 4. Video assembly ===")
    if monitor:
        monitor.stage_start("assemble_video")
    try:
        assemble_video(str(script_path))
        assembled = json.loads(script_path.read_text(encoding="utf-8"))
        if monitor:
            monitor.stage_end("assemble_video", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("assemble_video", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise

    final_vid = Path(assembled.get("final_video", ""))
    if not final_vid.exists():
        logging.error("assemble_video did not produce expected file: %s", final_vid)
        if monitor:
            monitor.run_error("missing_final_video")
        return

    logging.info("=== 5. Whisper captioning ===")
    cap_vid = final_vid.with_name(final_vid.stem + "_cap.mp4")
    if monitor:
        monitor.stage_start("captions")
    try:
        caps = create_captions(str(final_vid))
        if caps:
            try:
                captions.add_captions_to_video(
                    input_video_path=str(final_vid),
                    transcription=caps,
                    output_video_path=str(cap_vid),
                )
            except Exception as exc:
                logging.warning("Caption overlay failed: %s", exc)
        if monitor:
            monitor.stage_end("captions", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("captions", ok=False, error=str(exc))
        raise

    if cap_vid.exists():
        assembled["captions_video"] = str(Path(cap_vid).resolve())
        assembled["final_video"] = str(Path(cap_vid).resolve())
        _save_script(script_path, assembled)
        logging.info("Captioned video -> %s", cap_vid)
    else:
        cap_vid = final_vid
        logging.info("No captioned output produced; continuing with assembled video.")

    logging.info("=== 6. Overlay text ===")
    out_path = config.paths.final_dir / f"{script_path.stem}_final.mp4"
    config.paths.final_dir.mkdir(parents=True, exist_ok=True)

    if monitor:
        monitor.stage_start("overlay")
    status = subprocess.run([
        "python3", str(BASE_DIR / "overlay.py"),
        "--input_video", str(cap_vid),
        "--output_video", str(out_path),
        str(script_path),
    ])

    if status.returncode != 0 or not out_path.exists():
        logging.error("overlay.py failed or did not produce output")
        if monitor:
            monitor.stage_end("overlay", ok=False, error="overlay_failed")
            monitor.run_error("overlay_failed")
        return
    if monitor:
        monitor.stage_end("overlay", ok=True)

    logging.info("Overlay complete -> %s", out_path)

    assembled = json.loads(script_path.read_text(encoding="utf-8"))
    assembled["overlay_video"] = str(Path(out_path).resolve())
    assembled["final_video"] = str(Path(out_path).resolve())
    _save_script(script_path, assembled)

    logging.info("=== Pipeline finished successfully ===")
    if monitor:
        monitor.run_success(final_video=str(Path(out_path).resolve()))

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
