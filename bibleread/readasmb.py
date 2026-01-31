#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import openai

# Load environment variables from .env if present
dotenv_path = Path(__file__).parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path)

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Tracking processed inputs
PROCESSED_FILE = 'processed_files.json'
# Directories
INPUT_DIR = Path('bible')
OUTPUT_DIR = Path('ready')

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)


def load_processed():
    if Path(PROCESSED_FILE).exists():
        try:
            return set(json.loads(Path(PROCESSED_FILE).read_text()))
        except Exception:
            return set()
    return set()


def save_processed(processed):
    Path(PROCESSED_FILE).write_text(json.dumps(sorted(processed), indent=2))


def generate_similar_visual_prompt(base_prompt, narration_text):
    """
    Generate a new visual prompt similar in style to `base_prompt` but tailored to `narration_text`.
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a visual creative assistant."},
                {"role": "user", "content": (
                    f"Create a vivid and descriptive image prompt for the following Bible passage, "
                    f"matching the style of this existing prompt: '{base_prompt}'.\n\n"
                    f"Text: '{narration_text}'\n\n"
                    "Be creative but stay thematically consistent."
                )}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Failed visual prompt generation: {e}")
        return base_prompt + " (continued)"


def generate_assembler_json(input_path: Path, output_path: Path):
    data = json.loads(input_path.read_text())
    verses = data.get('verses', [])
    if not verses:
        print(f"[WARNING] No verses found in {input_path}")
        return

    # Build initial sections per verse
    intermediate_sections = []
    for idx, verse in enumerate(verses, start=1):
        name = verse.get('name', '').strip()
        text = verse.get('text', '').strip()
        base_prompt = verse.get('visual_prompt', f"Vivid scene for: {text}")
        words = text.split()
        short = ' '.join(words[:5]).rstrip('.,')
        title = f"{name} – {short}"
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
        intermediate_sections.append(section)

    # Combine all sections into one
    # Aggregate text and compute total duration
    aggregated_text = '\n\n'.join([sec['segments'][0]['narration']['text'] for sec in intermediate_sections])
    total_duration = sum(sec['section_duration'] for sec in intermediate_sections)
    total_duration = max(5, min(60, total_duration))  # clamp between 5 and 60s if desired

    # Use first visual prompt as base
    first_prompt = intermediate_sections[0]['segments'][0]['visual']['prompt']
    visual_prompt = generate_similar_visual_prompt(first_prompt, aggregated_text)

    # Create single combined section
    combined_section = {
        "section_number": 1,
        "original_name": f"{intermediate_sections[0]['original_name']} – {intermediate_sections[-1]['original_name']}",
        "title": f"{intermediate_sections[0]['original_name']} – {intermediate_sections[-1]['original_name']}",
        "section_duration": total_duration,
        "segments": [
            {
                "segment_number": 1,
                "narration": {
                    "text": aggregated_text,
                    "start": 0,
                    "duration": total_duration,
                    "audio_path": "audio/section_1_segment_1.mp3"
                },
                "visual": {
                    "type": "image",
                    "prompt": visual_prompt,
                    "start": 0,
                    "duration": total_duration,
                    "image_path": "visuals/section_1_segment_1.png"
                },
                "sound": {"transition_effect": "fade_in"}
            }
        ]
    }

    assembler = {
        "settings": {
            "video_size": "1080x1920",
            "use_transitions": True,
            "use_background_music": True,
            "background_music_type": "uplifting",
            "image_generation_style": "Leonardo Phoenix 1.0",
            "style_selection_reason": "Selected based on script content."
        },
        "sections": [combined_section]
    }

    # Build reference and description
    reference = combined_section['original_name']
    full_verses = aggregated_text
    prefix = f"📖 Today's Verses: {reference}\n\n✅ Welcome to Spoken in Light! We post a short video each day.\n\n"
    suffix = (
        "\n\n🙏 Subscribe and join our journey from Genesis to Revelation — one small passage at a time!"
        "\n\n📲 Follow Us:"
        "\n📺 YouTube: www.youtube.com/@SpokeninLight"
        "\n📷 Instagram: www.instagram.com/spokeninlight/"
        "\n👍 Facebook: www.facebook.com/profile.php?id=61576181588827"
        "\n\n#DailyBiblePassages #DailyScripture #BibleReading #FaithJourney #GenesisToRevelation"
    )
    description = (prefix + full_verses + suffix)[:2000]

    tags = [
        "DailyBiblePassages", "DailyScripture", "BibleReading", "FaithJourney",
        "GenesisToRevelation", "BibleVerse", "ChristianContent"
    ]

    assembler.update({
        "social_media": {"title": f"📖 Today's Verses: {reference}", "description": description, "tags": tags},
        "background_music_type": assembler['settings']['background_music_type'],
        "background_music_name": assembler['settings']['background_music_type'],
        "background_music": assembler['settings']['background_music_type'],
        "reference": reference,
        "tone": data.get('tone', 'hopeful'),
        "image_style": assembler['settings']['image_generation_style']
    })

    output_path.write_text(json.dumps(assembler, indent=2))
    print(f"Generated script: {input_path} -> {output_path}")


def main():
    processed = load_processed()
    files = sorted(
        INPUT_DIR.glob('script_*.json'),
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
        break
    save_processed(processed)


if __name__ == '__main__':
    main()
