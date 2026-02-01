#!/usr/bin/env python3
from __future__ import annotations

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


def assemble_script(config: AppConfig, payload: FactPayload, seo: dict) -> dict:
    text = payload.fact.strip()
    duration = min(45, max(25, len(text.split()) // 3))
    return {
        "settings": {
            "video_size": config.video.size,
            "use_transitions": True,
            "use_background_music": True,
            "background_music_type": config.video.bg_music_tag,
            "image_generation_style": config.channel.image_generation_style,
            "style_selection_reason": f"Selected based on structure {payload.structure}.",
            "bg_music_volume": config.video.bg_music_volume,
        },
        "sections": [
            {
                "section_number": 1,
                "original_name": payload.title,
                "title": payload.title,
                "section_duration": duration,
                "segments": [
                    {
                        "segment_number": 1,
                        "narration": {
                            "text": text,
                            "start": 0,
                            "duration": duration,
                            "audio_path": "audio/section_1_segment_1.mp3",
                        },
                        "visual": {
                            "type": "image",
                            "prompt": payload.visual_prompt,
                            "start": 0,
                            "duration": duration,
                            "image_path": "visuals/section_1_segment_1.png",
                        },
                        "sound": {
                            "transition_effect": "fade_in",
                        },
                    }
                ],
            }
        ],
        "social_media": {
            "title": payload.title,
            "description": seo.get("description", ""),
            "tags": seo.get("tags")
            if seo.get("tags")
            else ["#weirdhistory", "#didyouknow", "#funfact", "#trivia", "#viralshorts"],
        },
        "background_music_type": config.video.bg_music_tag,
        "background_music": config.video.bg_music_tag,
        "tone": "Valentino",
        "image_style": config.channel.image_generation_style,
        "reference": payload.title,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    config = load_config()
    ensure_dirs(config)

    rng = random.Random()
    history = load_history(config.memory.history_file)
    memory = MemoryManager.build(config.memory)

    prompt_assets = load_prompt_assets(config.prompts)
    topic_assets = load_topic_assets(config.topics)

    for attempt in range(config.openai.max_retries):
        plan = select_topic(
            rng=rng,
            assets=topic_assets,
            recent_topics=memory.topics.items,
            recent_hooks=memory.hooks.items,
            recent_structures=memory.styles.items,
        )
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

        seo = generate_seo_payload(config=config, assets=prompt_assets, rng=rng, title=payload.title, fact_text=payload.fact)
        assembler = assemble_script(config, payload, seo)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", payload.title).strip("_").lower()
        out_path = config.paths.ready_dir / f"{slug}_assembler.json"
        out_path.write_text(json.dumps(assembler, indent=2), encoding="utf-8")
        logging.info("Fact script saved to %s", out_path)

        history.add(payload.title)
        save_history(config.memory.history_file, history)
        memory.remember(payload.topic, payload.structure, payload.hook)
        break
    else:
        raise SystemExit("Failed to generate a unique fact after retries.")


if __name__ == "__main__":
    main()
