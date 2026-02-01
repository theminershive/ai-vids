#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from pathlib import Path

from config import AppConfig, ensure_dirs, load_config
from memory_manager import MemoryManager
from prompt_engine import FactPayload, generate_fact_payload, generate_seo_payload, load_prompt_assets
from topic_generator import load_topic_assets, select_topic


def load_history(path: Path) -> set[str]:
    if path.exists():
        try:
            return set(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return set()
    return set()


def save_history(path: Path, history: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(history), indent=2), encoding="utf-8")


def _duration_for_text(text: str) -> int:
    words = text.split()
    return min(15, max(5, len(words) // 2))


def assemble_script(config: AppConfig, payload: FactPayload, seo: dict) -> dict:
    segments = list(payload.segments)
    target_count = max(1, config.segments.count)
    while len(segments) < target_count and segments:
        longest_idx = max(range(len(segments)), key=lambda i: len(segments[i].get("narration", "").split()))
        longest = segments.pop(longest_idx)
        words = longest.get("narration", "").split()
        if len(words) < config.segments.min_words * 2:
            segments.insert(longest_idx, longest)
            break
        mid = len(words) // 2
        first_text = " ".join(words[:mid]).strip()
        second_text = " ".join(words[mid:]).strip()
        prompt = longest.get("visual_prompt", "")
        segments.insert(longest_idx, {"narration": first_text, "visual_prompt": prompt})
        segments.insert(longest_idx + 1, {"narration": second_text, "visual_prompt": prompt})

    sections = []
    for idx, seg in enumerate(segments, start=1):
        narration = seg.get("narration", "").strip()
        if not narration:
            continue
        visual_prompt = (seg.get("visual_prompt") or "").strip()
        duration = _duration_for_text(narration)
        section = {
            "section_number": idx,
            "original_name": payload.title,
            "title": payload.title,
            "section_duration": duration,
            "segments": [
                {
                    "segment_number": 1,
                    "narration": {
                        "text": narration,
                        "start": 0,
                        "duration": duration,
                        "audio_path": f"audio/section_{idx}_segment_1.mp3",
                    },
                    "visual": {
                        "type": "image",
                        "prompt": visual_prompt,
                        "start": 0,
                        "duration": duration,
                        "image_path": f"visuals/section_{idx}_segment_1.png",
                    },
                    "sound": {
                        "transition_effect": "fade_in",
                    },
                }
            ],
        }
        sections.append(section)

    narration_full = "\n\n".join(s["segments"][0]["narration"]["text"] for s in sections)

    return {
        "settings": {
            "video_size": config.video.size,
            "use_transitions": config.video.use_transitions,
            "use_background_music": config.video.use_background_music,
            "background_music_type": config.video.bg_music_tag,
            "image_generation_style": config.channel.image_generation_style,
            "style_selection_reason": f"Selected based on structure {payload.structure}.",
            "bg_music_volume": config.video.bg_music_volume,
            "transition_volume": config.video.transition_volume,
        },
        "sections": sections,
        "social_media": {
            "title": payload.title,
            "description": seo.get("description", ""),
            "tags": seo.get("tags") if seo.get("tags") else ["#weirdhistory", "#didyouknow", "#funfact"],
        },
        "background_music_type": config.video.bg_music_tag,
        "background_music": config.video.bg_music_tag,
        "tone": "Valentino",
        "image_style": config.channel.image_generation_style,
        "reference": payload.title,
        "narration_full": narration_full,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Generate assembler JSON from a topic.")
    parser.add_argument("--topic", help="Optional fixed topic override")
    parser.add_argument("--structure", help="Optional fixed structure name")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)

    rng = random.Random()
    history = load_history(config.memory.history_file)
    memory = MemoryManager.build(config.memory)

    prompt_assets = load_prompt_assets(config.prompts)
    topic_assets = load_topic_assets(config.topics)

    for attempt in range(config.openai.max_retries):
        if args.topic:
            plan = select_topic(
                rng=rng,
                assets=topic_assets,
                recent_topics=memory.topics.items,
                recent_hooks=memory.hooks.items,
                recent_structures=memory.styles.items,
            )
            plan = plan.__class__(topic=args.topic, category=plan.category, hook=plan.hook, structure=args.structure or plan.structure)
        else:
            plan = select_topic(
                rng=rng,
                assets=topic_assets,
                recent_topics=memory.topics.items,
                recent_hooks=memory.hooks.items,
                recent_structures=memory.styles.items,
            )
        if args.structure:
            plan = plan.__class__(topic=plan.topic, category=plan.category, hook=plan.hook, structure=args.structure)

        logging.info("Topic candidate: %s (%s)", plan.topic, plan.category)
        try:
            payload = generate_fact_payload(
                config=config,
                assets=prompt_assets,
                rng=rng,
                topic=plan.topic,
                hook=plan.hook,
                structure=plan.structure,
            )
        except Exception as exc:
            logging.warning("Fact generation failed: %s", exc)
            time.sleep(1)
            continue

        if payload.title in history:
            logging.info("Title already used, retrying: %s", payload.title)
            time.sleep(1)
            continue

        seo = generate_seo_payload(config=config, assets=prompt_assets, rng=rng, title=payload.title, fact_text=payload.segments[0].get("narration", ""))
        assembler = assemble_script(config, payload, seo)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", payload.title).strip("_").lower()
        out_path = config.paths.ready_dir / f"{slug}_assembler.json"
        out_path.write_text(json.dumps(assembler, indent=2), encoding="utf-8")
        logging.info("Assembler JSON saved to %s", out_path)

        history.add(payload.title)
        save_history(config.memory.history_file, history)
        memory.remember(payload.topic, payload.structure, payload.hook)
        break
    else:
        raise SystemExit("Failed to generate a unique fact after retries.")


if __name__ == "__main__":
    main()
