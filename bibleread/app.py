#!/usr/bin/env python3
"""
app.py (FIXED)
✅ Backwards compatible with BOTH old + new visuals.py behavior.

What this fixes:
- Old app logic assumed Leonardo: gen_id -> poll -> image_url -> requests.get(url)
- With ComfyUI, your "gen_id" is actually a LOCAL PATH (downloaded_content/section_1.png)
  and requests.get() explodes with MissingSchema.
- This version ALWAYS routes downloads through visuals.download_content(), which is
  "smart" (URL OR local path) in your visuals.py.

It also:
- Uses section_idx correctly (not always 1)
- Uses Flux prompt normalization via visuals.normalize_flux2_prompt() implicitly (ComfyUI workflow builder does it)
- Still works if VISUAL_BACKEND=leonardo (same old flow)
"""

import json
import logging
from pathlib import Path

from dotenv import load_dotenv

# Your visuals.py (the one you pasted)
import visuals


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


READY_DIR = Path("ready")
OUT_JSON_DEFAULT = Path("video_script_with_visuals.json")

# Where you want final images referenced from the script
# (keep "downloaded_content" if that's your convention)
IMAGES_DIR = Path("downloaded_content")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _latest_json(ready_dir: Path) -> Path:
    files = sorted(ready_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No JSON files found in {ready_dir.resolve()}")
    return files[-1]


def _safe_section_idx(section: dict, fallback: int) -> int:
    # supports section_number, sectionIndex, etc.
    for k in ("section_number", "sectionIndex", "section_idx", "index"):
        if k in section and str(section[k]).isdigit():
            return int(section[k])
    return fallback


def _safe_segment_idx(seg: dict, fallback: int) -> int:
    for k in ("segment_number", "segmentIndex", "segment_idx", "index"):
        if k in seg and str(seg[k]).isdigit():
            return int(seg[k])
    return fallback


def generate_and_download_images(script: dict) -> dict:
    """
    Backwards compatible visual generation:
    - Leonardo: gen_id -> poll -> url -> download_content(url, out_path)
    - ComfyUI: local_path returned -> download_content(local_path, out_path) (copies)
    """
    settings = script.get("settings") or {}
    style_name = settings.get("image_generation_style") or None

    sections = script.get("sections") or []
    if not sections:
        logging.warning("No sections found in script JSON.")
        return script

    for s_i, section in enumerate(sections, start=1):
        section_idx = _safe_section_idx(section, s_i)
        segments = section.get("segments") or []
        if not segments:
            continue

        for g_i, seg in enumerate(segments, start=1):
            seg_idx = _safe_segment_idx(seg, g_i)

            visual = seg.get("visual") or {}
            prompt = (visual.get("prompt") or "").strip()
            if not prompt:
                continue

            # ---- Preferred modern API: visuals.generate_visual() always returns a LOCAL PATH
            if hasattr(visuals, "generate_visual"):
                local_path = visuals.generate_visual(prompt, section_idx=section_idx, style_name=style_name)
                if not local_path:
                    logging.error(f"Visual generation failed (section {section_idx} seg {seg_idx}).")
                    continue

                out_path = IMAGES_DIR / f"sec{section_idx}_seg{seg_idx}.png"
                visuals.download_content(local_path, str(out_path))  # smart: url OR local path
                seg.setdefault("visual", {})
                seg["visual"]["image_path"] = str(out_path)
                continue

            # ---- Old API: generate_image_with_retry() returns (gen_id_or_local_path, used_prompt)
            gen_id_or_path, used_prompt = visuals.generate_image_with_retry(prompt, visuals.get_model_config_by_style(style_name))

            if not gen_id_or_path:
                logging.error(f"Visual generation failed (section {section_idx} seg {seg_idx}).")
                continue

            # Poll (Leonardo) OR local-path passthrough (ComfyUI shim returns complete dict)
            result = visuals.poll_generation_status(gen_id_or_path)

            # Extract URL (Leonardo) OR local path (ComfyUI shim returns local path)
            url_or_path = visuals.extract_image_url(result)
            if not url_or_path:
                logging.error(f"No image URL/path returned (section {section_idx} seg {seg_idx}).")
                continue

            out_path = IMAGES_DIR / f"sec{section_idx}_seg{seg_idx}.png"
            visuals.download_content(url_or_path, str(out_path))
            seg.setdefault("visual", {})
            seg["visual"]["image_path"] = str(out_path)

    return script


def main():
    # Load latest ready JSON
    src = _latest_json(READY_DIR)
    logging.info(f"Using: {src}")

    script = json.loads(src.read_text(encoding="utf-8"))

    # Generate visuals
    logging.info("Generating visuals...")
    script = generate_and_download_images(script)

    # Save back to the same file (or change to OUT_JSON_DEFAULT if you prefer)
    src.write_text(json.dumps(script, indent=4), encoding="utf-8")
    logging.info(f"Updated JSON written: {src}")


if __name__ == "__main__":
    main()
