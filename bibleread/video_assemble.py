#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_assemble.py

– Reads assembler JSON (e.g. ready/2_assembler.json)
– Resolves the single image path relative to that JSON
– Builds a 10-second video from that image at the JSON’s video_size
– Searches Freesound (user Nancy_Sinclair) for background music tag
– Loops and fades in/out background music with configurable volume
– Writes the MP4 into a configurable final video directory and updates JSON
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Ensure PIL ANTIALIAS compatibility
from PIL import Image
try:
    _ = Image.ANTIALIAS
except AttributeError:
    Image.ANTIALIAS = getattr(Image, 'LANCZOS', Image.BICUBIC)

from moviepy.editor import ImageClip, AudioFileClip
from moviepy.audio.fx.all import audio_loop, audio_fadein, audio_fadeout

# ---------------- Configuration ----------------
DURATION        = 15.0        # seconds
FPS             = 30          # frames per second
FADEIN_DURATION = 1.0         # fade-in for bg music
FADEOUT_DURATION= 1.0         # fade-out for bg music
FREESOUND_USER  = "Nancy_Sinclair"
FREESOUND_BASE  = "https://freesound.org/apiv2"
DEFAULT_BG      = "./fallbacks/default_bg_music.mp3"
FINAL_VIDEO_DIR = "./final/"

# Load environment
load_dotenv()
FS_API_KEY      = os.getenv('FREESOUND_API_KEY')
FINAL_DIR_ENV   = os.getenv('FINAL_VIDEO_DIR')

BANNED_SONGS = []  # add banned titles if needed

# ---------------- Helpers ----------------
def search_sounds(query, user, num_results=20):
    params = {
        'query': query,
        'filter': f"username:\"{user}\" AND license:\"Creative Commons 0\"",
        'sort': 'rating_desc',
        'fields': 'id,name,previews,license,duration,username',
        'token': FS_API_KEY,
        'page_size': num_results
    }
    try:
        r = requests.get(f"{FREESOUND_BASE}/search/text/", params=params)
        r.raise_for_status()
        return r.json().get('results', [])
    except Exception:
        return []


def download_sound(sound, out_path):
    url = sound.get('previews', {}).get('preview-hq-mp3')
    if not url:
        return None
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
        return str(out_path)
    except Exception:
        return None


def fetch_background_music(tag, total_duration):
    """
    Search Freesound for 'tag' by Nancy_Sinclair. Fallback to default.
    """
    if FS_API_KEY:
        results = search_sounds(tag, FREESOUND_USER)
        for s in results:
            name = s.get('name', '')
            if name in BANNED_SONGS:
                continue
            out_dir = Path("./sounds")
            out_dir.mkdir(exist_ok=True)
            out_file = out_dir / f"bg_{s['id']}.mp3"
            got = download_sound(s, out_file)
            if got:
                print(f"[BG SELECTED] {name}")
                return got
    # fallback
    if Path(DEFAULT_BG).exists():
        print("[FALLBACK] Using default background music.")
        return DEFAULT_BG
    raise FileNotFoundError("No background music available.")

# ---------------- Main ----------------
def main():
    if len(sys.argv) != 2:
        print("Usage: python video_assemble.py <assembler.json>")
        sys.exit(1)

    json_path = Path(sys.argv[1]).resolve()
    if not json_path.is_file():
        print(f"JSON not found: {json_path}")
        sys.exit(1)

    # Load script JSON
    data     = json.loads(json_path.read_text(encoding="utf-8"))
    settings = data.get("settings", {})
    vs       = settings.get("video_size", "1080x1920")
    w, h     = map(int, vs.split("x"))
    VIDEO_SIZE = (w, h)

    # Determine final video directory
    if FINAL_DIR_ENV:
        final_dir = Path(FINAL_DIR_ENV)
    else:
        final_dir = json_path.parent
    final_dir.mkdir(parents=True, exist_ok=True)

    # Resolve image path relative to JSON
    section = data["sections"][0]
    seg     = section["segments"][0]
    raw_ip  = Path(seg["visual"]["image_path"])
    img_file= raw_ip if raw_ip.is_absolute() else (raw_ip).resolve()

    if not img_file.is_file():
        print(f"[ERROR] Image not found at {img_file}")
        sys.exit(1)

    # Create image clip
    clip = (
        ImageClip(str(img_file))
        .set_duration(DURATION)
        .resize(VIDEO_SIZE)
    )

    # Fetch background music
    tag = data.get("background_music_type", "")
    try:
        bg_path = fetch_background_music(tag, DURATION)
    except Exception as e:
        print(f"[WARN] {e} – skipping music.")
        bg_path = None

    # Volume settings
    bg_volume = settings.get('bg_music_volume', 0.08)

    # Attach looped and faded bg audio with volume
    if bg_path:
        bg_audio = AudioFileClip(bg_path)
        bg_audio = audio_loop(bg_audio, duration=DURATION)
        bg_audio = bg_audio.fx(audio_fadein, FADEIN_DURATION).fx(audio_fadeout, FADEOUT_DURATION)
        bg_audio = bg_audio.volumex(bg_volume)
        final_clip = clip.set_audio(bg_audio)
    else:
        final_clip = clip

    # Write output
    out_file = final_dir / f"{json_path.stem}.mp4"
    final_clip.write_videofile(
        str(out_file), fps=FPS, codec="libx264", audio_codec="aac"
    )

    # Update JSON
    data['final_video'] = str(out_file)
    json_path.write_text(json.dumps(data, indent=2))
    print(f"✔ Video created: {out_file}\n✔ JSON updated.")

if __name__ == "__main__":
    main()
