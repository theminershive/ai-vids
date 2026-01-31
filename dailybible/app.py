#!/usr/bin/env python3

"""
Full video-production pipeline
"""

import json
import logging
import os
import subprocess
import re
from pathlib import Path
from typing import Optional

# Suppress PIL debug logs
logging.getLogger("PIL").setLevel(logging.WARNING)

from visuals import (
    get_model_config_by_style,
    generate_image_with_retry,
    poll_generation_status,
    extract_image_url,
    download_content,
    rewrite_prompt,
)
from titlegen import generate_social_media
from tts import process_tts
from video_assembler import assemble_video
import captions

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
        r"\bslain\b": "",
        r"\bkilled\b": "",
        r"\bblood\b": "red liquid",
        r"\bdeath\b": "the act of ending life",
        r"\binjured\b": "",
        r"\bwound(?:ed|ing)?\b": "",
        r"\bhurt(?:ed|ing)?\b": "",
        r"\bgore\b": "graphic",
        r"\bbleeding\b": "graphic",
        r"\bviolent\b": "aggressive",
        r"\bcelebrity\b": "figure",
        r"\bfamous\b": "well-known",
        r"\bpublic\s+figure\b": "figure",
        r"\bpenis\b": "anatomical part",
        r"\bcock\b": "anatomical part",
        r"\bdick\b": "anatomical part",
        r"\bshaft\b": "anatomical part",
        r"\btesticle\b": "anatomical part",
        r"\btesticles\b": "anatomical parts",
        r"\bscrotum\b": "anatomical part",
        r"\bvagina\b": "anatomical part",
        r"\bvulva\b": "anatomical part",
        r"\blabia\b": "anatomical parts",
        r"\bclitoris\b": "anatomical part",
        r"\bpussy\b": "anatomical part",
        r"\bbreast\b": "anatomical part",
        r"\bbreasts\b": "anatomical parts",
        r"\bboob\b": "anatomical part",
        r"\bboobs\b": "anatomical parts",
        r"\btit\b": "anatomical part",
        r"\btits\b": "anatomical parts",
        r"\bnipple\b": "anatomical part",
        r"\bnipples\b": "anatomical parts",
        r"\bbutt\b": "anatomical part",
        r"\bbuttocks\b": "anatomical parts",
        r"\bass\b": "anatomical part",
        r"\banus\b": "anatomical part",
        r"\basshole\b": "anatomical part",
        r"\bcum\b": "fluid",
        r"\bsemen\b": "fluid",
        r"\bsperm\b": "fluid",
        r"\bmilf\b": "individual",
        r"\borgasm\b": "reaction",
        r"\barousal\b": "reaction",
        r"\bsex\b": "intimacy",
        r"\bsexual\b": "intimate",
        r"\bfuck\b": "act",
        r"\bfucking\b": "act",
        r"\bintercourse\b": "act",
        r"\bpenetration\b": "act",
        r"\bmasturbat(e|ion|ing)\b": "act",
        r"\bjerk\s*off\b": "act",
        r"\bhandjob\b": "act",
        r"\bblowjob\b": "act",
        r"\bsuck\b": "act",
        r"\bgrope\b": "act",
        r"\bstrip(per|ping)\b": "act",
        r"\bbukkake\b": "act",
        r"\bcp\b": "content",
        r"\bchild\s*porn\b": "content",
        r"\bincest\b": "content",
        r"\brape\b": "act",
        r"\bbeastiality\b": "content",
        r"\bzoophilia\b": "content",
    }
    sanitized = prompt
    for pattern, replacement in FILTER_KEYWORDS.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized

# ——— Generate and download images ——————————————————————————————————
def generate_and_download_images(script: dict) -> dict:
    for section in script.get("sections", []):
        segs = section.get("segments") or [
            {
                "segment_number": idx,
                "narration": {"text": ns.get("narration", "")},
                "visual": {"type": "image", "prompt": ns.get("visual_prompt", "")},
            }
            for idx, ns in enumerate(section.get("narration_segments", []), start=1)
        ]
        section["segments"] = segs

        model_cfg = get_model_config_by_style(
            script.get("settings", {}).get("image_generation_style", "")
        )
        for seg in segs:
            raw_prompt = seg["visual"].get("prompt", "")
            if not raw_prompt:
                continue
            prompt = sanitize_visual_prompt(raw_prompt)
            gen_id, used_prompt = generate_image_with_retry(prompt, model_cfg)
            if not gen_id:
                logging.error(f"Skipping section {section.get('section_number',0)} segment {seg['segment_number']} due to generation failure.")
                continue
            try:
                data = poll_generation_status(gen_id)
            except RuntimeError:
                logging.warning("Polling failed — trying rewritten prompt...")
                safe_prompt = rewrite_prompt(prompt)
                gen_id, _ = generate_image_with_retry(safe_prompt, model_cfg)
                if not gen_id:
                    logging.error("Skipping due to failure even after rewrite.")
                    continue
                data = poll_generation_status(gen_id)

            if not data:
                logging.error("Image generation did not complete.")
                continue
            url = extract_image_url(data)
            if not url:
                logging.error("Could not extract image URL.")
                continue
            img_file = VISUALS_DIR / f"section_{section.get('section_number',0)}_segment_{seg['segment_number']}.png"
            download_content(url, str(img_file))
            seg["visual"]["image_path"] = str(img_file)
    return script

# ——— Create captions —————————————————————————————————————————————
def create_captions(video_path: str) -> Optional[list]:
    try:
        audio_temp = captions.extract_audio(video_path)
        transcription = captions.transcribe_audio_whisper(audio_temp)
        caps = captions.generate_captions_from_whisper(transcription)
        Path(audio_temp).unlink(missing_ok=True)
        return caps
    except Exception as exc:
        logging.warning(f"Captioning failed: {exc}")
        return None

# ——— Main pipeline —————————————————————————————————————————————
def main():
    logging.info("=== 1. assemble.py ====================================================")
    subprocess.run(["python3", "assemble.py"], check=True)

    ready_files = sorted(READY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not ready_files:
        logging.error("No assembler JSON found in ./ready")
        return
    script_path = ready_files[-1]
    logging.info(f"Using assembler JSON ➜ {script_path.name}")
    script = json.loads(script_path.read_text(encoding="utf-8"))

    logging.info("=== 2. Visual generation =============================================")
    script = generate_and_download_images(script)

    logging.info("=== 3. TTS generation ================================================")
    script = process_tts(script)
    script_path.write_text(json.dumps(script, indent=4), encoding="utf-8")

    logging.info("=== 4. Video assembly ===============================================")
    assemble_video(str(script_path))
    assembled = json.loads(script_path.read_text(encoding="utf-8"))
    final_vid = Path(assembled.get("final_video", ""))
    if not final_vid.exists():
        logging.error(f"assemble_video did not produce expected file: {final_vid}")
        return

    logging.info("=== 5. Whisper captioning ============================================")
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

    logging.info("=== 6. Overlay text ==================================================")
    out_path = FINAL_VIDEO_DIR / f"{script_path.stem}_final.mp4"
    FINAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
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
    if status.returncode != 0:
        logging.error("overlay.py failed")
        return
    logging.info(f"Overlay complete ➜ {out_path}")

    assembled["final_video"] = str(out_path)

    logging.info("=== 7. Uploading =====================================================")
    #try:
    #    from oauth_get2 import refresh_token
    #    refresh_token()
    #    logging.info("OAuth refresh: SUCCESS")
    #except Exception as e:
    #    logging.error(f"OAuth refresh failed: {e}")
    #try:
    #    yt_url = upload_youtube(str(script_path))
    #    logging.info(f"YouTube upload ✓ {yt_url}")
    #except Exception as exc:
    #    logging.error(f"YouTube upload failed: {exc}")
    #try:
    #    upload_facebook(str(script_path))
    #    logging.info("Facebook upload ✓")
    #except Exception as exc:
    #    logging.error(f"Facebook upload failed: {exc}")
    #try:
    #    upload_instagram(str(script_path))
    #    logging.info("Instagram upload ✓")
    #except Exception as exc:
    #    logging.error(f"Instagram upload failed: {exc}")

    logging.info("=== Pipeline finished successfully ====================================")

if __name__ == "__main__":
    main()
