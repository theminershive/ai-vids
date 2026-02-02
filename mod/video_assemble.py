#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_assemble.py

- Reads assembler JSON
- Builds a short video from the generated image at the configured size
- Fetches background music from Freesound (optional)
- Writes MP4 and updates JSON
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image
from dotenv import load_dotenv

from moviepy.editor import ImageClip, AudioFileClip
from moviepy.audio.fx.all import audio_loop, audio_fadein, audio_fadeout

from config import AppConfig, load_config

load_dotenv()

# Ensure PIL ANTIALIAS compatibility (Pillow>=10 removed Image.ANTIALIAS)
try:
    Image.ANTIALIAS
except AttributeError:
    Image.ANTIALIAS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

FREESOUND_BASE = "https://freesound.org/apiv2"


def search_sounds(api_key: str, query: str, user: str, num_results: int = 20):
    params = {
        "query": query,
        "filter": f"username:\"{user}\" AND license:\"Creative Commons 0\"",
        "sort": "rating_desc",
        "fields": "id,name,previews,license,duration,username",
        "token": api_key,
        "page_size": num_results,
    }
    try:
        r = requests.get(f"{FREESOUND_BASE}/search/text/", params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


def download_sound(sound: dict, out_path: Path) -> str | None:
    url = sound.get("previews", {}).get("preview-hq-mp3")
    if not url:
        return None
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return str(out_path)
    except Exception:
        return None


def fetch_background_music(config: AppConfig, total_duration: float) -> str | None:
    api_key = os.getenv("FREESOUND_API_KEY", "").strip()
    if api_key:
        results = search_sounds(api_key, config.video.bg_music_tag, config.video.freesound_user)
        for sound in results:
            out_dir = config.paths.sounds_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"bg_{sound['id']}.mp3"
            got = download_sound(sound, out_file)
            if got:
                return got
    fallback = config.paths.fallback_bg_music
    if fallback.exists():
        return str(fallback)
    return None


def assemble_from_json(config: AppConfig, json_path: Path) -> Path:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    settings = data.get("settings", {})

    vs = settings.get("video_size", config.video.size)
    w, h = map(int, vs.split("x"))
    video_size = (w, h)

    section = data["sections"][0]
    seg = section["segments"][0]
    raw_ip = Path(seg["visual"]["image_path"])
    img_file = raw_ip if raw_ip.is_absolute() else (json_path.parent / raw_ip).resolve()

    if not img_file.is_file():
        raise FileNotFoundError(f"Image not found at {img_file}")

    clip = (
        ImageClip(str(img_file))
        .set_duration(config.video.duration_s)
        .resize(video_size)
    )

    bg_path = fetch_background_music(config, config.video.duration_s)
    bg_volume = settings.get("bg_music_volume", config.video.bg_music_volume)

    if bg_path:
        bg_audio = AudioFileClip(bg_path)
        bg_audio = audio_loop(bg_audio, duration=config.video.duration_s)
        bg_audio = bg_audio.fx(audio_fadein, config.video.fade_in_s).fx(audio_fadeout, config.video.fade_out_s)
        bg_audio = bg_audio.volumex(bg_volume)
        final_clip = clip.set_audio(bg_audio)
    else:
        final_clip = clip

    out_dir = config.paths.final_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{json_path.stem}.mp4"
    final_clip.write_videofile(str(out_file), fps=config.video.fps, codec="libx264", audio_codec="aac")

    data["final_video"] = str(out_file)
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_file


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python video_assemble.py <assembler.json>")
        raise SystemExit(1)
    json_path = Path(sys.argv[1]).resolve()
    if not json_path.is_file():
        raise SystemExit(f"JSON not found: {json_path}")
    config = load_config()
    assemble_from_json(config, json_path)


if __name__ == "__main__":
    main()
