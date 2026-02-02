#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List

from config import TopicPaths


@dataclass(frozen=True)
class TopicAssets:
    seeds: Dict[str, List[str]]
    hooks: List[str]
    structures: List[str]
    twists: List[str]


@dataclass(frozen=True)
class TopicPlan:
    topic: str
    category: str
    hook: str
    structure: str


def load_topic_assets(paths: TopicPaths) -> TopicAssets:
    seeds = json.loads(Path(paths.seeds).read_text(encoding="utf-8"))
    hooks = json.loads(Path(paths.hooks).read_text(encoding="utf-8"))
    structures = json.loads(Path(paths.structures).read_text(encoding="utf-8"))
    twists = json.loads(Path(paths.twists).read_text(encoding="utf-8"))
    return TopicAssets(
        seeds=seeds if isinstance(seeds, dict) else {},
        hooks=hooks if isinstance(hooks, list) else [],
        structures=structures if isinstance(structures, list) else [],
        twists=twists if isinstance(twists, list) else [],
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    seq = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return max(jaccard, seq)


def _build_topic(seed: str, twist: str) -> str:
    return f"{seed}, tied to {twist}"


def generate_candidates(rng: random.Random, assets: TopicAssets, count: int) -> List[TopicPlan]:
    plans: List[TopicPlan] = []
    categories = list(assets.seeds.keys())
    for _ in range(count):
        category = rng.choice(categories)
        seed = rng.choice(assets.seeds[category])
        twist = rng.choice(assets.twists)
        topic = _build_topic(seed, twist)
        hook_template = rng.choice(assets.hooks)
        hook = hook_template.format(category=category.replace("_", " "), topic=topic)
        structure = rng.choice(assets.structures)
        plans.append(TopicPlan(topic=topic, category=category, hook=hook, structure=structure))
    return plans


def select_topic(
    rng: random.Random,
    assets: TopicAssets,
    recent_topics: List[str],
    recent_hooks: List[str],
    recent_structures: List[str],
    candidate_count: int = 18,
    similarity_threshold: float = 0.68,
) -> TopicPlan:
    candidates = generate_candidates(rng, assets, candidate_count)
    best_score = -1.0
    best_plan = candidates[0]
    for plan in candidates:
        topic_similarity = max((similarity_score(plan.topic, t) for t in recent_topics), default=0.0)
        hook_similarity = max((similarity_score(plan.hook, h) for h in recent_hooks), default=0.0)
        structure_penalty = 0.2 if plan.structure in recent_structures else 0.0
        novelty_score = (1.0 - topic_similarity) + (0.5 - hook_similarity) - structure_penalty
        if topic_similarity >= similarity_threshold:
            continue
        if novelty_score > best_score:
            best_score = novelty_score
            best_plan = plan
    return best_plan
