#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from config import AppConfig, PromptPaths
from openai_client import build_openai_client


@dataclass(frozen=True)
class PromptAssets:
    script_system: str
    script_templates: List[Dict[str, str]]
    seo_system: str
    seo_user_template: str


@dataclass(frozen=True)
class FactPayload:
    topic: str
    title: str
    segments: List[Dict[str, str]]
    hook: str
    structure: str


def load_prompt_assets(paths: PromptPaths) -> PromptAssets:
    script_system = Path(paths.script_system).read_text(encoding="utf-8").strip()
    script_templates = json.loads(Path(paths.script_templates).read_text(encoding="utf-8"))
    seo_system = Path(paths.seo_system).read_text(encoding="utf-8").strip()
    seo_user_template = Path(paths.seo_user).read_text(encoding="utf-8").strip()
    return PromptAssets(
        script_system=script_system,
        script_templates=script_templates if isinstance(script_templates, list) else [],
        seo_system=seo_system,
        seo_user_template=seo_user_template,
    )


def _extract_json(content: str) -> Dict[str, object]:
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        raise ValueError("No JSON object found in response.")
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # handle common LLM issue: double-encoded JSON
        try:
            decoded = json.loads(raw.strip("\"'"))
            if isinstance(decoded, str):
                return json.loads(decoded)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
    return json.loads(raw)


def _random_temperature(rng: random.Random, range_bounds: Tuple[float, float]) -> float:
    low, high = range_bounds
    return max(0.2, min(1.2, rng.uniform(low, high)))


def call_chat_with_retries(
    client,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_completion_tokens: int,
    timeout: float,
    max_retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            wait = min(6, 1.5 * attempt)
            logging.warning("OpenAI call failed (attempt %s/%s): %s", attempt, max_retries, exc)
            time.sleep(wait)
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_error}")


def _normalize_segments(raw_segments: object) -> List[Dict[str, str]]:
    if isinstance(raw_segments, list):
        segments = raw_segments
    elif isinstance(raw_segments, dict):
        segments = [raw_segments]
    else:
        segments = []
    cleaned = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        narration = str(seg.get("narration", "")).strip()
        visual_prompt = str(seg.get("visual_prompt", "")).strip()
        if narration:
            cleaned.append({"narration": narration, "visual_prompt": visual_prompt})
    return cleaned


def generate_fact_payload(
    config: AppConfig,
    assets: PromptAssets,
    rng: random.Random,
    topic: str,
    hook: str,
    structure: str,
) -> FactPayload:
    template = next((t for t in assets.script_templates if t.get("name") == structure), None)
    if not template:
        template = assets.script_templates[0] if assets.script_templates else {"template": "Topic: {topic}\nHook: {hook}"}
    prompt = template["template"].format(
        topic=topic,
        hook=hook,
        segments_count=config.segments.count,
    )
    messages = [
        {"role": "system", "content": assets.script_system},
        {"role": "user", "content": prompt},
        {
            "role": "user",
            "content": (
                "Return ONLY valid JSON with keys: title, segments. "
                "Do NOT wrap the JSON in quotes."
            ),
        },
    ]
    client = build_openai_client()
    content = call_chat_with_retries(
        client=client,
        model=config.openai.model,
        messages=messages,
        temperature=_random_temperature(rng, config.openai.temperature_range),
        max_completion_tokens=900,
        timeout=config.openai.request_timeout,
        max_retries=config.openai.max_retries,
    )
    payload = _extract_json(content)
    title = str(payload.get("title", "")).strip()
    segments = _normalize_segments(payload.get("segments"))
    if not title or not segments:
        raise ValueError("Payload missing title or segments.")
    return FactPayload(
        topic=topic,
        title=title,
        segments=segments,
        hook=hook,
        structure=structure,
    )


def generate_seo_payload(
    config: AppConfig,
    assets: PromptAssets,
    rng: random.Random,
    title: str,
    fact_text: str,
) -> Dict[str, List[str]]:
    seo_prompt = (
        assets.seo_user_template
        .replace("{title}", title)
        .replace("{fact_text}", fact_text)
    )
    client = build_openai_client()
    content = call_chat_with_retries(
        client=client,
        model=config.openai.model,
        messages=[
            {"role": "system", "content": assets.seo_system},
            {"role": "user", "content": seo_prompt},
        ],
        temperature=_random_temperature(rng, config.openai.temperature_range),
        max_completion_tokens=350,
        timeout=config.openai.request_timeout,
        max_retries=config.openai.max_retries,
    )
    data = _extract_json(content)
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [f"#{t.strip()}" for t in tags.split("#") if t.strip()]
    return {
        "description": str(data.get("description", "")).strip(),
        "tags": tags if isinstance(tags, list) else [],
    }
