#!/usr/bin/env python3
"""
Render one or more ready/*.json assembler files into a final captioned + overlayed video.
Matches the mod2 workflow steps: visuals -> TTS -> assemble -> captions -> overlay.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

from config import load_config, ensure_dirs
from app import generate_and_download_images, create_captions, _save_script
from tts import process_tts
from video_assembler import assemble_video
import captions

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def _episode_sort_key(path: Path) -> tuple[int, str]:
    name = path.stem.lower()
    m = __import__("re").search(r"_ep(\\d+)_", name)
    if m:
        return (int(m.group(1)), name)
    m = __import__("re").search(r"(\\d+)", name)
    if m:
        return (int(m.group(1)), name)
    return (0, name)


def _run_pipeline(script_path: Path, config) -> None:
    script = json.loads(script_path.read_text(encoding="utf-8"))

    logging.info("=== Visual generation ===")
    # inject theme via app.generate_and_download_images (reads theme from script)
    script = generate_and_download_images(script, config.paths.visuals_dir)
    _save_script(script_path, script)

    logging.info("=== TTS generation ===")
    script = process_tts(script, audio_dir=config.paths.audio_dir)
    _save_script(script_path, script)

    logging.info("=== Video assembly ===")
    assemble_video(str(script_path))
    assembled = json.loads(script_path.read_text(encoding="utf-8"))

    final_vid = Path(assembled.get("final_video", ""))
    if not final_vid.exists():
        logging.error("assemble_video did not produce expected file: %s", final_vid)
        return

    logging.info("=== Whisper captioning ===")
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

    logging.info("=== Overlay text ===")
    out_path = config.paths.final_dir / f"{script_path.stem}_final.mp4"
    config.paths.final_dir.mkdir(parents=True, exist_ok=True)

    status = shutil.which("python3") or "python3"
    res = __import__("subprocess").run([
        status, str(Path(__file__).parent / "overlay.py"),
        "--input_video", str(cap_vid),
        "--output_video", str(out_path),
        str(script_path),
    ])

    if res.returncode != 0 or not out_path.exists():
        logging.error("overlay.py failed or did not produce output")
        return

    logging.info("Overlay complete -> %s", out_path)
    assembled = json.loads(script_path.read_text(encoding="utf-8"))
    assembled["overlay_video"] = str(Path(out_path).resolve())
    assembled["final_video"] = str(Path(out_path).resolve())
    _save_script(script_path, assembled)

    logging.info("=== Finished ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render ready/*.json into final captioned videos.")
    parser.add_argument("--json", help="Path to a specific assembler JSON.")
    parser.add_argument("--all", action="store_true", help="Render all JSONs in ready/ (oldest -> newest).")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)

    ready_dir = config.paths.ready_dir
    if args.json:
        script_path = Path(args.json).resolve()
        if not script_path.exists():
            raise SystemExit(f"JSON not found: {script_path}")
        _run_pipeline(script_path, config)
        return

    json_files = sorted(ready_dir.glob("*.json"), key=_episode_sort_key)
    if not json_files:
        raise SystemExit("No assembler JSON found in ready/")

    if not args.all:
        _run_pipeline(json_files[0], config)
        return

    for path in json_files:
        logging.info("Rendering %s", path.name)
        _run_pipeline(path, config)


if __name__ == "__main__":
    main()
