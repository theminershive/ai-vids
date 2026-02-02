#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = Path(os.getenv("MOD2_CONFIG", str(BASE_DIR / "config.json"))).resolve()


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
    audio_dir: Path
    fallback_bg_music: Path
    fallback_transition: Path
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
    temperature_range: Tuple[float, float]
    request_timeout: float
    max_retries: int


@dataclass(frozen=True)
class ImageBackendConfig:
    visual_backend: str
    comfyui_base_url: str
    comfyui_timeout_s: int
    comfyui_poll_s: float
    comfyui_width: str
    comfyui_height: str
    comfyui_lora_name: str
    leonardo_api_endpoint: str
    local_prompt_rewrite_url: str
    local_prompt_rewrite_model: str


@dataclass(frozen=True)
class VideoConfig:
    size: str
    fps: int
    use_transitions: bool
    use_background_music: bool
    bg_music_volume: float
    transition_volume: float
    bg_music_tag: str
    freesound_user: str
    narration_initial_delay: float
    end_extension: float


@dataclass(frozen=True)
class CaptionsConfig:
    text_size: int
    font: str
    color: str
    stroke_color: str
    stroke_width: int
    max_words_per_caption: int


@dataclass(frozen=True)
class OverlayConfig:
    start_text: str
    end_text: str
    start_font: Path
    end_font: Path
    start_fontsize: int
    end_fontsize: int
    text_color: str
    bg_color: Tuple[int, int, int]
    bg_opacity: float
    padding: int
    fade_duration: float
    end_duration: float


@dataclass(frozen=True)
class MemoryConfig:
    topics_file: Path
    styles_file: Path
    hooks_file: Path
    history_file: Path
    history_depth: int


@dataclass(frozen=True)
class SegmentsConfig:
    count: int
    min_words: int
    max_words: int


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
    captions: CaptionsConfig
    overlay: OverlayConfig
    memory: MemoryConfig
    segments: SegmentsConfig


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
    captions = data.get("captions", {})
    overlay = data.get("overlay", {})
    memory = data.get("memory", {})
    segments = data.get("segments", {})

    temp_range = tuple(openai.get("temperature_range", [0.72, 0.95]))
    if len(temp_range) != 2:
        temp_range = (0.72, 0.95)

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
            audio_dir=_resolve_path(base, paths.get("audio_dir", "audio")),
            fallback_bg_music=_resolve_path(base, paths.get("fallback_bg_music", "fallbacks/default_bg_music.mp3")),
            fallback_transition=_resolve_path(base, paths.get("fallback_transition", "fallbacks/default_transition.mp3")),
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
            temperature_range=(float(temp_range[0]), float(temp_range[1])),
            request_timeout=float(openai.get("request_timeout", 45)),
            max_retries=int(openai.get("max_retries", 4)),
        ),
        image_backend=ImageBackendConfig(
            visual_backend=str(
                (
                    "comfyui"
                    if os.getenv("IMAGE_BACKEND", "").strip().lower() in ("flux", "comfyui")
                    else (
                        "leonardo"
                        if os.getenv("IMAGE_BACKEND", "").strip().lower() == "leonardo"
                        else image_backend.get("visual_backend", "comfyui")
                    )
                )
            ).strip().lower(),
            comfyui_base_url=str(image_backend.get("comfyui_base_url", "http://127.0.0.1:8188")).strip(),
            comfyui_timeout_s=int(image_backend.get("comfyui_timeout_s", 1800)),
            comfyui_poll_s=float(image_backend.get("comfyui_poll_s", 2.0)),
            comfyui_width=str(image_backend.get("comfyui_width", "")),
            comfyui_height=str(image_backend.get("comfyui_height", "")),
            comfyui_lora_name=str(image_backend.get("comfyui_lora_name", "Flux_2-Turbo-LoRA_comfyui.safetensors")),
            leonardo_api_endpoint=str(image_backend.get("leonardo_api_endpoint", "https://cloud.leonardo.ai/api/rest/v1")),
            local_prompt_rewrite_url=str(image_backend.get("local_prompt_rewrite_url", "")),
            local_prompt_rewrite_model=str(image_backend.get("local_prompt_rewrite_model", "")),
        ),
        video=VideoConfig(
            size=str(video.get("size", "1080x1920")),
            fps=int(video.get("fps", 24)),
            use_transitions=bool(video.get("use_transitions", True)),
            use_background_music=bool(video.get("use_background_music", True)),
            bg_music_volume=float(video.get("bg_music_volume", 0.15)),
            transition_volume=float(video.get("transition_volume", 0.05)),
            bg_music_tag=str(video.get("bg_music_tag", "cinematic")),
            freesound_user=str(video.get("freesound_user", "Nancy_Sinclair")),
            narration_initial_delay=float(video.get("narration_initial_delay", 0.25)),
            end_extension=float(video.get("end_extension", 0.5)),
        ),
        captions=CaptionsConfig(
            text_size=int(captions.get("text_size", 85)),
            font=str(captions.get("font", "Bangers-Regular.ttf")),
            color=str(captions.get("color", "white")),
            stroke_color=str(captions.get("stroke_color", "black")),
            stroke_width=int(captions.get("stroke_width", 1)),
            max_words_per_caption=int(captions.get("max_words_per_caption", 8)),
        ),
        overlay=OverlayConfig(
            start_text=str(overlay.get("start_text", "{reference}")),
            end_text=str(overlay.get("end_text", "Thanks for watching! Subscribe!")),
            start_font=_resolve_path(base, overlay.get("start_font", "fonts/Bangers-Regular.ttf")),
            end_font=_resolve_path(base, overlay.get("end_font", "fonts/Bangers-Regular.ttf")),
            start_fontsize=int(overlay.get("start_fontsize", 75)),
            end_fontsize=int(overlay.get("end_fontsize", 75)),
            text_color=str(overlay.get("text_color", "white")),
            bg_color=tuple(overlay.get("bg_color", [0, 0, 0])),
            bg_opacity=float(overlay.get("bg_opacity", 0.3)),
            padding=int(overlay.get("padding", 5)),
            fade_duration=float(overlay.get("fade_duration", 1.0)),
            end_duration=float(overlay.get("end_duration", 7.0)),
        ),
        memory=MemoryConfig(
            topics_file=_resolve_path(base, memory.get("topics_file", "memory/recent_topics.json")),
            styles_file=_resolve_path(base, memory.get("styles_file", "memory/recent_styles.json")),
            hooks_file=_resolve_path(base, memory.get("hooks_file", "memory/recent_hooks.json")),
            history_file=_resolve_path(base, memory.get("history_file", "memory/facts_history.json")),
            history_depth=int(memory.get("history_depth", 12)),
        ),
        segments=SegmentsConfig(
            count=int(segments.get("count", 4)),
            min_words=int(segments.get("min_words", 18)),
            max_words=int(segments.get("max_words", 55)),
        ),
    )


def ensure_dirs(config: AppConfig) -> None:
    for path in (
        config.paths.ready_dir,
        config.paths.visuals_dir,
        config.paths.final_dir,
        config.paths.sounds_dir,
        config.paths.audio_dir,
        config.paths.memory_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _video_size_tuple(config: AppConfig) -> Tuple[int, int]:
    w, h = config.video.size.split("x")
    return int(w), int(h)


VIDEO_SIZE = _video_size_tuple(load_config())
FPS = load_config().video.fps

CAPTION_SETTINGS = {
    "TEXT_SIZE": load_config().captions.text_size,
    "FONT": load_config().captions.font,
    "COLOR": load_config().captions.color,
    "STROKE_COLOR": load_config().captions.stroke_color,
    "STROKE_WIDTH": load_config().captions.stroke_width,
    "MAX_WORDS_PER_CAPTION": load_config().captions.max_words_per_caption,
}
