#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONFIG_PATH = Path(
    os.getenv("MOD_CONFIG", str(Path(__file__).with_name("config.json")))
).resolve()


@dataclass(frozen=True)
class ChannelConfig:
    name: str
    cta_text: str
    image_generation_style: str


@dataclass(frozen=True)
class PathsConfig:
    ready_dir: Path
    visuals_dir: Path
    final_dir: Path
    sounds_dir: Path
    fallback_bg_music: Path
    fonts_dir: Path
    memory_dir: Path


@dataclass(frozen=True)
class PromptPaths:
    script_system: Path
    script_templates: Path
    seo_system: Path
    seo_user: Path


@dataclass(frozen=True)
class TopicPaths:
    seeds: Path
    hooks: Path
    structures: Path
    twists: Path


@dataclass(frozen=True)
class OpenAIConfig:
    model: str
    image_model: str
    temperature_range: Tuple[float, float]
    request_timeout: float
    max_retries: int


@dataclass(frozen=True)
class ImageBackendConfig:
    type: str
    flux_api_url: str
    flux_workflow_name: str
    flux_timeout: float


@dataclass(frozen=True)
class VideoConfig:
    duration_s: float
    fps: int
    fade_in_s: float
    fade_out_s: float
    size: str
    bg_music_volume: float
    bg_music_tag: str
    freesound_user: str


@dataclass(frozen=True)
class OverlayConfig:
    name_font: Path
    verse_font: Path
    name_fontsize: int
    verse_fontsize: int
    cta_fontsize: int
    text_color: str
    bg_color: Tuple[int, int, int]
    opacity: float
    padding: int
    name_pos: Tuple[str, int]
    use_stroke: bool
    show_cta: bool


@dataclass(frozen=True)
class MemoryConfig:
    topics_file: Path
    styles_file: Path
    hooks_file: Path
    history_file: Path
    history_depth: int


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    channel: ChannelConfig
    paths: PathsConfig
    prompts: PromptPaths
    topics: TopicPaths
    openai: OpenAIConfig
    image_backend: ImageBackendConfig
    video: VideoConfig
    overlay: OverlayConfig
    memory: MemoryConfig


def _resolve_path(base: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(config_path: Path | None = None) -> AppConfig:
    cfg_path = (config_path or DEFAULT_CONFIG_PATH).resolve()
    data = _read_config(cfg_path)
    base = cfg_path.parent

    channel = data.get("channel", {})
    paths = data.get("paths", {})
    prompts = data.get("prompts", {})
    topics = data.get("topics", {})
    openai = data.get("openai", {})
    image_backend = data.get("image_backend", {})
    video = data.get("video", {})
    overlay = data.get("overlay", {})
    memory = data.get("memory", {})

    temperature_range = tuple(openai.get("temperature_range", [0.72, 0.95]))
    if len(temperature_range) != 2:
        temperature_range = (0.72, 0.95)

    overlay_name_pos = overlay.get("name_pos", ["center", 50])
    if len(overlay_name_pos) != 2:
        overlay_name_pos = ["center", 50]

    return AppConfig(
        base_dir=base,
        channel=ChannelConfig(
            name=str(channel.get("name", "Channel")),
            cta_text=str(channel.get("cta_text", "Subscribe")),
            image_generation_style=str(channel.get("image_generation_style", "")),
        ),
        paths=PathsConfig(
            ready_dir=_resolve_path(base, paths.get("ready_dir", "ready")),
            visuals_dir=_resolve_path(base, paths.get("visuals_dir", "visuals")),
            final_dir=_resolve_path(base, paths.get("final_dir", "final")),
            sounds_dir=_resolve_path(base, paths.get("sounds_dir", "sounds")),
            fallback_bg_music=_resolve_path(base, paths.get("fallback_bg_music", "fallbacks/default_bg_music.mp3")),
            fonts_dir=_resolve_path(base, paths.get("fonts_dir", "fonts")),
            memory_dir=_resolve_path(base, paths.get("memory_dir", "memory")),
        ),
        prompts=PromptPaths(
            script_system=_resolve_path(base, prompts.get("script_system", "prompts/script_system.txt")),
            script_templates=_resolve_path(base, prompts.get("script_templates", "prompts/script_templates.json")),
            seo_system=_resolve_path(base, prompts.get("seo_system", "prompts/seo_system.txt")),
            seo_user=_resolve_path(base, prompts.get("seo_user", "prompts/seo_user.txt")),
        ),
        topics=TopicPaths(
            seeds=_resolve_path(base, topics.get("seeds", "prompts/topic_seeds.json")),
            hooks=_resolve_path(base, topics.get("hooks", "prompts/hooks.json")),
            structures=_resolve_path(base, topics.get("structures", "prompts/structures.json")),
            twists=_resolve_path(base, topics.get("twists", "prompts/twists.json")),
        ),
        openai=OpenAIConfig(
            model=str(openai.get("model", "gpt-5.2")),
            image_model=str(openai.get("image_model", "gpt-image-1")),
            temperature_range=(float(temperature_range[0]), float(temperature_range[1])),
            request_timeout=float(openai.get("request_timeout", 45)),
            max_retries=int(openai.get("max_retries", 4)),
        ),
        image_backend=ImageBackendConfig(
            type=str(
                os.getenv("IMAGE_BACKEND", image_backend.get("type", "flux"))
            ).strip().lower(),
            flux_api_url=str(
                os.getenv("FLUX_API_URL", image_backend.get("flux_api_url", ""))
            ).strip(),
            flux_workflow_name=str(
                os.getenv("FLUX_WORKFLOW_NAME", image_backend.get("flux_workflow_name", ""))
            ).strip(),
            flux_timeout=float(
                os.getenv("FLUX_TIMEOUT", image_backend.get("flux_timeout", 90))
            ),
        ),
        video=VideoConfig(
            duration_s=float(video.get("duration_s", 15.0)),
            fps=int(video.get("fps", 30)),
            fade_in_s=float(video.get("fade_in_s", 1.0)),
            fade_out_s=float(video.get("fade_out_s", 1.0)),
            size=str(video.get("size", "1080x1920")),
            bg_music_volume=float(video.get("bg_music_volume", 0.08)),
            bg_music_tag=str(video.get("bg_music_tag", "cinematic")),
            freesound_user=str(video.get("freesound_user", "Nancy_Sinclair")),
        ),
        overlay=OverlayConfig(
            name_font=_resolve_path(base, overlay.get("name_font", "fonts/Anton-Regular.ttf")),
            verse_font=_resolve_path(base, overlay.get("verse_font", "fonts/Montserrat-Bold.ttf")),
            name_fontsize=int(overlay.get("name_fontsize", 70)),
            verse_fontsize=int(overlay.get("verse_fontsize", 45)),
            cta_fontsize=int(overlay.get("cta_fontsize", 35)),
            text_color=str(overlay.get("text_color", "white")),
            bg_color=tuple(overlay.get("bg_color", [0, 0, 0])),
            opacity=float(overlay.get("opacity", 0.5)),
            padding=int(overlay.get("padding", 30)),
            name_pos=(str(overlay_name_pos[0]), int(overlay_name_pos[1])),
            use_stroke=bool(overlay.get("use_stroke", False)),
            show_cta=bool(overlay.get("show_cta", True)),
        ),
        memory=MemoryConfig(
            topics_file=_resolve_path(base, memory.get("topics_file", "memory/recent_topics.json")),
            styles_file=_resolve_path(base, memory.get("styles_file", "memory/recent_styles.json")),
            hooks_file=_resolve_path(base, memory.get("hooks_file", "memory/recent_hooks.json")),
            history_file=_resolve_path(base, memory.get("history_file", "memory/facts_history.json")),
            history_depth=int(memory.get("history_depth", 12)),
        ),
    )


def ensure_dirs(config: AppConfig) -> None:
    for path in (
        config.paths.ready_dir,
        config.paths.visuals_dir,
        config.paths.final_dir,
        config.paths.sounds_dir,
        config.paths.memory_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
