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
    base = output_path.stem

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

    # Build initial sections
    for idx, verse in enumerate(data.get('verses', []), start=1):
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
        assembler['sections'].append(section)

    # Ensure at least 4 sections by splitting the longest if needed
    while len(assembler['sections']) < 4:
        # Identify the longest section by word count
        longest_idx = max(
            range(len(assembler['sections'])),
            key=lambda i: len(assembler['sections'][i]['segments'][0]['narration']['text'].split())
        )
        longest = assembler['sections'].pop(longest_idx)
        orig_name = longest['original_name']
        full_text = longest['segments'][0]['narration']['text']
        words = full_text.split()
        if len(words) < 6:
            assembler['sections'].insert(longest_idx, longest)
            break

        mid = len(words) // 2
        parts = [
            " ".join(words[:mid]),
            " ".join(words[mid:])
        ]
        prompts = [
            longest['segments'][0]['visual']['prompt'],
            generate_similar_visual_prompt(longest['segments'][0]['visual']['prompt'], parts[1])
        ]

        # Insert split sections
        for part_idx, part_text in enumerate(parts, start=1):
            words_part = part_text.split()
            short_part = ' '.join(words_part[:5]).rstrip('.,')
            title_part = f"{orig_name} – {short_part}"
            duration_part = min(15, max(5, len(words_part) // 2))
            new_section = {
                "original_name": orig_name,
                "title": title_part,
                "section_duration": duration_part,
                "segments": [
                    {
                        "segment_number": 1,
                        "narration": {"text": part_text, "start": 0, "duration": duration_part},
                        "visual": {"type": "image", "prompt": prompts[part_idx-1], "start": 0, "duration": duration_part},
                        "sound": {"transition_effect": "fade_in"}
                    }
                ]
            }
            assembler['sections'].insert(longest_idx + part_idx - 1, new_section)

    # Re-number sections and assign unique audio/image paths
    for sec_idx, section in enumerate(assembler['sections'], start=1):
        section['section_number'] = sec_idx
        for seg in section['segments']:
            seg_num = seg['segment_number']
            seg['narration']['audio_path'] = f"audio/section_{sec_idx}_segment_{seg_num}.mp3"
            seg['visual']['image_path'] = f"visuals/section_{sec_idx}_segment_{seg_num}.png"

    # Build reference and description
    verse_names = [s['original_name'] for s in assembler['sections']]
    reference = f"{verse_names[0]} – {verse_names[-1]}" if verse_names else ""
    full_verses = "\n\n".join([
        f"{s['original_name']} {s['segments'][0]['narration']['text']}" for s in assembler['sections']
    ])

    prefix = f"📖 Today's Verses: {reference}\n\n✝️ Welcome to Daily Bible Passages! We post 3 short Bible videos daily, featuring 3–5 verses each.\n\n"
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
        "social_media": { "title": f"📖 Today's Verses: {reference}", "description": description, "tags": tags },
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
