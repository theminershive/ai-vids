#!/usr/bin/env python3
"""
Full video-production pipeline using topic-driven scripts and dailybible-style video assembly.
"""

import json
import logging
import subprocess
import re
import shutil
from pathlib import Path
from typing import Optional

from config import load_config, ensure_dirs
from visuals import generate_visual, rewrite_prompt
from tts import process_tts
from video_assembler import assemble_video
import captions

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
BASE_DIR = Path(__file__).parent.resolve()


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
    subprocess.run(["python3", str(BASE_DIR / "assemble.py")], check=True)

    ready_files = sorted(config.paths.ready_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not ready_files:
        logging.error("No assembler JSON found in ./ready")
        return
    script_path = ready_files[-1]
    logging.info("Using assembler JSON -> %s", script_path.name)
    script = json.loads(script_path.read_text(encoding="utf-8"))

    logging.info("=== 2. Visual generation ===")
    script = generate_and_download_images(script, config.paths.visuals_dir)
    _save_script(script_path, script)

    logging.info("=== 3. TTS generation ===")
    script = process_tts(script, audio_dir=config.paths.audio_dir)
    _save_script(script_path, script)

    logging.info("=== 4. Video assembly ===")
    assemble_video(str(script_path))
    assembled = json.loads(script_path.read_text(encoding="utf-8"))

    final_vid = Path(assembled.get("final_video", ""))
    if not final_vid.exists():
        logging.error("assemble_video did not produce expected file: %s", final_vid)
        return

    logging.info("=== 5. Whisper captioning ===")
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
            logging.warning("Caption overlay failed: %s", exc)

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

    status = subprocess.run([
        "python3", str(BASE_DIR / "overlay.py"),
        "--input_video", str(cap_vid),
        "--output_video", str(out_path),
        str(script_path),
    ])

    if status.returncode != 0 or not out_path.exists():
        logging.error("overlay.py failed or did not produce output")
        return

    logging.info("Overlay complete -> %s", out_path)

    assembled = json.loads(script_path.read_text(encoding="utf-8"))
    assembled["overlay_video"] = str(Path(out_path).resolve())
    assembled["final_video"] = str(Path(out_path).resolve())
    _save_script(script_path, assembled)

    logging.info("=== Pipeline finished successfully ===")


if __name__ == "__main__":
    main()
