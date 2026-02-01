#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv

import requests

# Load environment variables from .env if present
dotenv_path = Path(__file__).parent / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)

# -------------------------
# Ollama config (LOCAL LLM)
# -------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://192.168.1.176:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "90"))

# Tracking processed inputs
PROCESSED_FILE = "processed_files.json"

# Directories
INPUT_DIR = Path("bible")
OUTPUT_DIR = Path("ready")

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def load_processed():
    if Path(PROCESSED_FILE).exists():
        try:
            return set(json.loads(Path(PROCESSED_FILE).read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_processed(processed):
    Path(PROCESSED_FILE).write_text(
        json.dumps(sorted(processed), indent=2),
        encoding="utf-8"
    )


def ollama_generate(prompt: str) -> str:
    """
    Call Ollama /api/generate (non-stream) and return response text.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 220,
        },
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def generate_visual_prompt(style_reference: str, narration_text: str) -> str:
    """
    Generate a vivid vertical 9:16 image prompt for the verse using Ollama.

    Requirements (enforced via instruction):
    - This is for a Bible Passage short video for church/Bible viewers
    - Output ONLY the final prompt (no labels like "PROMPT:", no bullets, no extra commentary)
    - The prompt itself must include: "no text, no writing, no watermark, no logo"
    - Do NOT start with "Vivid scene for:"
    """
    base_style = style_reference.strip() if style_reference and style_reference.strip() else (
        "Cinematic, reverent Bible illustration, ultra-detailed, dramatic lighting, "
        "soft volumetric rays, realistic textures, vertical 9:16 composition"
    )

    llm_prompt = (
        "You write image-generation prompts.\n"
        "Output ONLY one single-paragraph prompt. No labels, no quotes, no bullets, no extra text.\n"
        "This prompt is for a short Bible Passage vertical video for church/Bible viewers "
        "(reverent, uplifting, family-friendly).\n"
        "The prompt MUST explicitly include: 'no text, no writing, no watermark, no logo'.\n"
        "The prompt MUST NOT start with 'Vivid scene for:' or any similar prefix.\n"
        "Include: subject, setting, composition, lighting, mood, style cues.\n"
        "Keep it respectful and faithful to the passage.\n"
        "Keep it optimized for a vertical 9:16 image.\n"
        "Avoid: modern signage, UI elements, overlays, watermarks, brand marks.\n\n"
        f"STYLE REFERENCE:\n{base_style}\n\n"
        f"BIBLE PASSAGE TEXT:\n{narration_text}\n"
    )

    try:
        out = ollama_generate(llm_prompt).strip().strip('"\'')
        out = re.sub(r"^\s*Vivid scene for:\s*", "", out, flags=re.IGNORECASE).strip()
        out = " ".join(out.split())
        return out if out else base_style
    except Exception as e:
        print(f"[ERROR] Ollama visual prompt generation failed: {e}")
        return base_style


def generate_assembler_json(input_path: Path, output_path: Path):
    data = json.loads(input_path.read_text(encoding="utf-8"))

    assembler = {
        "settings": {
            "video_size": "1080x1920",
            "use_transitions": True,
            "use_background_music": True,
            "background_music_type": "uplifting",
            "image_generation_style": "Leonardo Phoenix 1.0",
            "style_selection_reason": "Selected based on script content."
        },
        "sections": []
    }

    # Stable style reference (tune as you like)
    DEFAULT_STYLE_REF = (
        "Cinematic, reverent Bible illustration, ultra-detailed, dramatic lighting, "
        "soft volumetric rays, realistic textures, shallow depth of field, "
        "high contrast, natural color grading, vertical 9:16 composition"
    )

    # Build initial sections
    for idx, verse in enumerate(data.get("verses", []), start=1):
        name = (verse.get("name") or "").strip()
        text = (verse.get("text") or "").strip()

        existing_prompt = (verse.get("visual_prompt") or "").strip()
        if existing_prompt:
            base_prompt = existing_prompt
        else:
            # Generate prompt via local Ollama
            base_prompt = generate_visual_prompt(DEFAULT_STYLE_REF, text)

        words = text.split()
        short = " ".join(words[:5]).rstrip(".,")
        title = f"{name} – {short}" if name else short
        duration = min(15, max(5, len(words) // 2))

        section = {
            "original_name": name,
            "title": title,
            "section_duration": duration,
            "segments": [
                {
                    "segment_number": 1,
                    "narration": {"text": text, "start": 0, "duration": duration},
                    "visual": {"type": "image", "prompt": base_prompt, "start": 0, "duration": duration},
                    "sound": {"transition_effect": "fade_in"}
                }
            ]
        }
        assembler["sections"].append(section)

    # Ensure at least 4 sections by splitting the longest if needed
    while len(assembler["sections"]) < 4 and assembler["sections"]:
        longest_idx = max(
            range(len(assembler["sections"])),
            key=lambda i: len(assembler["sections"][i]["segments"][0]["narration"]["text"].split())
        )
        longest = assembler["sections"].pop(longest_idx)
        orig_name = longest.get("original_name", "")
        full_text = longest["segments"][0]["narration"]["text"]
        words = full_text.split()

        if len(words) < 6:
            assembler["sections"].insert(longest_idx, longest)
            break

        mid = len(words) // 2
        parts = [" ".join(words[:mid]), " ".join(words[mid:])]

        p1 = longest["segments"][0]["visual"]["prompt"]

        # Use p1 as a style reference for p2 (this encourages consistency)
        p2 = generate_visual_prompt(p1, parts[1])

        prompts = [p1, p2]

        for part_idx, part_text in enumerate(parts, start=1):
            words_part = part_text.split()
            short_part = " ".join(words_part[:5]).rstrip(".,")
            title_part = f"{orig_name} – {short_part}" if orig_name else short_part
            duration_part = min(15, max(5, len(words_part) // 2))

            new_section = {
                "original_name": orig_name,
                "title": title_part,
                "section_duration": duration_part,
                "segments": [
                    {
                        "segment_number": 1,
                        "narration": {"text": part_text, "start": 0, "duration": duration_part},
                        "visual": {"type": "image", "prompt": prompts[part_idx - 1], "start": 0, "duration": duration_part},
                        "sound": {"transition_effect": "fade_in"}
                    }
                ]
            }
            assembler["sections"].insert(longest_idx + part_idx - 1, new_section)

    # Re-number sections and assign unique audio/image paths
    for sec_idx, section in enumerate(assembler["sections"], start=1):
        section["section_number"] = sec_idx
        for seg in section["segments"]:
            seg_num = seg["segment_number"]
            seg["narration"]["audio_path"] = f"audio/section_{sec_idx}_segment_{seg_num}.mp3"
            seg["visual"]["image_path"] = f"visuals/section_{sec_idx}_segment_{seg_num}.png"

    # Build reference and description
    verse_names = [s.get("original_name", "") for s in assembler["sections"]]
    verse_names = [vn for vn in verse_names if vn]
    reference = f"{verse_names[0]} – {verse_names[-1]}" if verse_names else ""

    full_verses = "\n\n".join([
        f"{s.get('original_name','').strip()} {s['segments'][0]['narration']['text']}".strip()
        for s in assembler["sections"]
    ])

    prefix = (
        f"📖 Today's Verses: {reference}\n\n"
        "✝️ Welcome to Daily Bible Passages! We post 3 short Bible videos daily, featuring 3–5 verses each.\n\n"
    )
    suffix = (
        "\n\n🙌 Subscribe and join our journey from Genesis to Revelation — one small passage at a time!"
        "\n\n📖 Follow Us:\n➡️ Patreon: https://patreon.com/dailybiblepassages"
        "\n➡️ Instagram: instagram.com/thebibledailyyt"
        "\n➡️ Facebook: www.facebook.com/profile.php?id=61575301427014"
        "\n\n#DailyBiblePassages #DailyScripture #BibleReading #FaithJourney #GenesisToRevelation"
    )
    description = (prefix + full_verses + suffix)[:2000]

    tags = [
        "DailyBiblePassages", "DailyScripture", "BibleReading", "FaithJourney",
        "GenesisToRevelation", "BibleVerse", "ChristianContent",
        "FaithQuotes", "VerseOfTheDay", "GodIsGood"
    ]

    assembler.update({
        "social_media": {"title": f"📖 Today's Verses: {reference}", "description": description, "tags": tags},
        "background_music_type": assembler["settings"]["background_music_type"],
        "background_music_name": assembler["settings"]["background_music_type"],
        "background_music": assembler["settings"]["background_music_type"],
        "reference": reference,
        "tone": data.get("tone", "hopeful"),
        "image_style": assembler["settings"]["image_generation_style"]
    })

    output_path.write_text(
        json.dumps(assembler, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Generated script: {input_path} -> {output_path}")


def main():
    processed = load_processed()

    files = sorted(
        INPUT_DIR.glob("script_*.json"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))
    )

    for input_file in files:
        if str(input_file) in processed:
            continue

        seq = len(processed) + 1
        out_file = OUTPUT_DIR / f"{seq}_assembler.json"

        try:
            generate_assembler_json(input_file, out_file)
            processed.add(str(input_file))
        except Exception as e:
            print(f"[ERROR] {input_file}: {e}")

        # Process one file per run (as your original script did)
        break

    save_processed(processed)


if __name__ == "__main__":
    main()
