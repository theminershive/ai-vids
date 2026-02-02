#!/usr/bin/env python3
"""
Full video-production pipeline (FIXED)

Fixes:
- Persist final output paths back into the assembler JSON after captions + overlay
- Keep the same pipeline order/behavior

Also fixes (2026-01-31):
- Visual prompt lookup is now BACKWARDS COMPATIBLE with BOTH JSON formats:
    OLD: section["visual"]["prompt"]
    NEW: section["segments"][0]["visual"]["prompt"]
  (and writes image_path back into the same place it read from)
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

# Suppress PIL debug logs
logging.getLogger("PIL").setLevel(logging.WARNING)

from visuals import generate_visual, rewrite_prompt
from titlegen import generate_social_media
from tts import process_tts
from video_assembler import assemble_video
import captions

# uploaders (kept, commented in main)
from ytuploader import upload as upload_youtube
from fbupload import upload as upload_facebook
from igupload import upload as upload_instagram

# ——— Logging ————————————————————————————————————————————————
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ——— Directories —————————————————————————————————————————————
READY_DIR       = Path("ready")
VISUALS_DIR     = Path("visuals")
FINAL_VIDEO_DIR = Path("final")

for d in (READY_DIR, VISUALS_DIR, FINAL_VIDEO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ——— Monitoring ——————————————————————————————————————————————
BASE_DIR = Path(__file__).resolve().parent
MONITOR_DIR = BASE_DIR.parent / "monitor"
if MONITOR_DIR.exists():
    sys.path.insert(0, str(MONITOR_DIR))
try:
    import monitoring  # type: ignore
    monitor = monitoring.get_monitor("dailybible", base_dir=BASE_DIR)
except Exception:
    monitor = None

# ——— Helper: sanitize visual prompts ——————————————————————————————————
def sanitize_visual_prompt(prompt: str) -> str:
    FILTER_KEYWORDS = {
        r"\bchild\b": "figure",
        r"\bchildren\b": "figures",
        r"\bflesh\b": "skin",
        r"\bkid\b": "person",
        r"\bminor\b": "individual",
        r"\bcircumcised\b": "",
        r"\bforeskin\b": "",
        r"\binfant\b": "figure",
        r"\btoddler\b": "person",
        r"\bbaby\b": "person",
        r"\bteen\b": "young person",
        r"\bteenager\b": "young person",
        r"\byouth\b": "individual",
        r"\bjuvenile\b": "individual",
        r"\bunderage\b": "young individual",
        r"\bchildlike\b": "figurative",
        r"\blittle\s+girl\b": "young person",
        r"\blittle\s+boy\b": "young person",
        r"\bnaked\b": "modest",
        r"\bnakedness\b": "modestness",
        r"\bslain\b": "fallen",
        r"\bgore\b": "dramatic intensity",
    }
    out = prompt or ""
    for pattern, repl in FILTER_KEYWORDS.items():
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out

def _save_script(script_path: Path, script: dict) -> None:
    """Persist JSON to disk (single place so we don't forget)."""
    script_path.write_text(json.dumps(script, indent=4, ensure_ascii=False), encoding="utf-8")

def _extract_run_number(stem: str) -> int | None:
    match = re.match(r"^(\\d+)", stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None

def generate_and_download_images(script: dict) -> dict:
    """
    Uses visuals.py to create images for each section based on a visual prompt.

    Backwards compatible with BOTH JSON formats:

    OLD (one visual per section):
        section["visual"]["prompt"]

    NEW (visual per segment):
        section["segments"][0]["visual"]["prompt"]

    Writes image_path back into the same place it was read from.
    """
    style = script.get("style", "") or script.get("image_style", "")
    sections = script.get("sections", [])
    for section_idx, section in enumerate(sections, start=1):
        # Prefer the NEW schema: section.segments[0].visual.prompt
        prompt = ""
        segments = section.get("segments", []) or []
        seg0 = segments[0] if segments else {}

        if isinstance(seg0, dict):
            visual_seg = seg0.get("visual") or {}
            if isinstance(visual_seg, dict):
                prompt = (visual_seg.get("prompt") or "").strip()

        # Fall back to OLD schema: section.visual.prompt
        if not prompt:
            visual_old = section.get("visual") or {}
            if isinstance(visual_old, dict):
                prompt = (visual_old.get("prompt") or "").strip()

        if not prompt:
            logging.warning(f"Section {section_idx}: missing visuals.prompt; skipping image generation.")
            continue

        prompt = rewrite_prompt(prompt)
        prompt = sanitize_visual_prompt(prompt)

        logging.info(f"[VIS] Section {section_idx}: generating image...")
        saved = generate_visual(prompt, section_idx=section_idx, style_name=style)
        if not saved:
            logging.error(f"[VIS] Section {section_idx}: generate_visual returned no image")
            continue

        saved_path = Path(saved)
        out_path = VISUALS_DIR / f"section_{section_idx}{saved_path.suffix or '.png'}"
        if saved_path.resolve() != out_path.resolve():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(saved_path, out_path)
            saved_path = out_path

        saved_abs = str(saved_path.resolve())

        # Write back to NEW schema if present, otherwise OLD schema.
        if segments and isinstance(seg0, dict):
            seg0.setdefault("visual", {})
            if isinstance(seg0["visual"], dict):
                seg0["visual"]["image_path"] = saved_abs
            else:
                seg0["visual"] = {"image_path": saved_abs}
            # Ensure the mutated seg0 is stored back
            segments[0] = seg0
            section["segments"] = segments
        else:
            section.setdefault("visual", {})
            if isinstance(section["visual"], dict):
                section["visual"]["image_path"] = saved_abs
            else:
                section["visual"] = {"image_path": saved_abs}

        logging.info(f"[VIS] Section {section_idx}: saved -> {saved_abs}")

    return script

def create_captions(video_path: str) -> Optional[list]:
    """
    Extracts audio, transcribes using captions.transcribe_audio() (local broker or OpenAI depending on env),
    then converts verbose_json -> list of caption segments.
    """
    try:
        audio_temp = captions.extract_audio(video_path)
        transcription = captions.transcribe_audio(audio_temp)
        caps = captions.generate_captions_from_whisper(transcription)
        Path(audio_temp).unlink(missing_ok=True)
        return caps
    except Exception as exc:
        logging.warning(f"Captioning failed: {exc}")
        return None

# ——— Main pipeline —————————————————————————————————————————————
def main():
    logging.info("=== 1. assemble.py ====================================================")
    if monitor:
        monitor.stage_start("assemble")
    try:
        subprocess.run(["python3", "assemble.py"], check=True)
        if monitor:
            monitor.stage_end("assemble", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("assemble", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise

    ready_files = sorted(READY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not ready_files:
        logging.error("No assembler JSON found in ./ready")
        if monitor:
            monitor.run_error("no_ready_json")
        return
    script_path = ready_files[-1]
    logging.info(f"Using assembler JSON ➜ {script_path.name}")
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

    logging.info("=== 2. Visual generation =============================================")
    if monitor:
        monitor.stage_start("visuals")
    try:
        script = generate_and_download_images(script)
        _save_script(script_path, script)
        if monitor:
            monitor.stage_end("visuals", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("visuals", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise


    logging.info("=== 3. TTS generation ================================================")
    if monitor:
        monitor.stage_start("tts")
    try:
        script = process_tts(script)
        _save_script(script_path, script)
        if monitor:
            monitor.stage_end("tts", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("tts", ok=False, error=str(exc))
            monitor.run_error(str(exc))
        raise

    logging.info("=== 4. Video assembly ===============================================")
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
        logging.error(f"assemble_video did not produce expected file: {final_vid}")
        if monitor:
            monitor.run_error("missing_final_video")
        return

    logging.info("=== 5. Whisper captioning ============================================")
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
                logging.warning(f"Caption overlay failed: {exc}")
        if monitor:
            monitor.stage_end("captions", ok=True)
    except Exception as exc:
        if monitor:
            monitor.stage_end("captions", ok=False, error=str(exc))
        raise

    if cap_vid.exists():
        # Persist caption result back to JSON
        assembled["captions_video"] = str(Path(cap_vid).resolve())
        assembled["final_video"] = str(Path(cap_vid).resolve())
        _save_script(script_path, assembled)
        logging.info(f"Captioned video ➜ {cap_vid}")
    else:
        cap_vid = final_vid
        logging.info("No captioned output produced; continuing with assembled video.")

    logging.info("=== 6. Overlay text ==================================================")
    out_path = FINAL_VIDEO_DIR / f"{script_path.stem}_final.mp4"
    FINAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    if monitor:
        monitor.stage_start("overlay")
    status = subprocess.run([
        "python3", "overlay.py",
        "--input_video", str(cap_vid),
        "--output_video", str(out_path),
        "--start_text", script.get("reference", ""),
        "--end_text", "Thanks for watching! Subscribe!",
        "--start_font_path", "Bangers-Regular.ttf",
        "--end_font_path", "Bangers-Regular.ttf",
        "--start_fontsize", "75",
        "--end_fontsize", "75",
        "--fade_in", "--fade_out",
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

    logging.info(f"Overlay complete ➜ {out_path}")

    # ✅ persist overlay output back into JSON
    assembled = json.loads(script_path.read_text(encoding="utf-8"))
    assembled["overlay_video"] = str(Path(out_path).resolve())
    assembled["final_video"] = str(Path(out_path).resolve())
    _save_script(script_path, assembled)

    logging.info("=== 7. Uploading =====================================================")
    if os.getenv("UPLOAD_YOUTUBE", "0").strip() == "1":
        try:
            yt_url = upload_youtube(str(script_path))
            logging.info(f"YouTube upload ✓ {yt_url}")
        except Exception as exc:
            logging.error(f"YouTube upload failed: {exc}")
    if os.getenv("UPLOAD_FACEBOOK", "0").strip() == "1":
        try:
            upload_facebook(str(script_path))
            logging.info("Facebook upload ✓")
        except Exception as exc:
            logging.error(f"Facebook upload failed: {exc}")
    if os.getenv("UPLOAD_INSTAGRAM", "0").strip() == "1":
        try:
            upload_instagram(str(script_path))
            logging.info("Instagram upload ✓")
        except Exception as exc:
            logging.error(f"Instagram upload failed: {exc}")

    logging.info("=== Pipeline finished successfully ====================================")
    logging.info(f"Final JSON: {script_path.resolve()}")
    logging.info(f"Final video: {Path(assembled['final_video']).resolve()}")
    if monitor:
        monitor.run_success(final_video=str(Path(assembled["final_video"]).resolve()))

if __name__ == "__main__":
    main()
