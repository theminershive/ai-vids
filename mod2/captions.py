#!/usr/bin/env python3
import sys
import os
import json
import argparse
import tempfile
from typing import List, Dict, Optional, Tuple

from moviepy.editor import TextClip, ImageClip, VideoFileClip, CompositeVideoClip
from PIL import Image, ImageFilter
import matplotlib.font_manager as fm
import numpy as np
from dotenv import load_dotenv

from config import CAPTION_SETTINGS, BASE_DIR

# Load environment variables
load_dotenv()

# -----------------------------
# ENV (STT)
# -----------------------------
STT_BACKEND = os.getenv("STT_BACKEND", "").strip().lower()  # e.g. faster_whisper_api
STT_API_BASE = os.getenv("STT_API_BASE", "").strip()
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "auto").strip()

# Choose transcription provider: local or openai
TRANSCRIBE_PROVIDER = os.getenv("TRANSCRIBE_PROVIDER", "local").strip().lower()

# Local GPU Audio Broker STT endpoint
LOCAL_STT_URL = os.getenv("LOCAL_STT_URL", "").strip()  # e.g. http://192.168.1.94:9910/v1/audio/transcriptions
LOCAL_STT_TIMEOUT = int(os.getenv("LOCAL_STT_TIMEOUT", "1800"))

# Optional fallback if local broker fails
TRANSCRIBE_FALLBACK = os.getenv("TRANSCRIBE_FALLBACK", "").strip().lower()  # "openai" or ""

DEBUG_TTS = os.getenv("DEBUG_TTS", "0").strip().lower() in ("1", "true", "yes", "y")  # you use DEBUG_TTS flag globally

# -----------------------------
# Audio extraction
# -----------------------------
def extract_audio(input_video_path: str) -> str:
    try:
        video = VideoFileClip(input_video_path)
        audio = video.audio
        # Export audio as MP3 to reduce file size on large videos
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            audio.write_audiofile(temp_file.name, codec='libmp3lame')
            return temp_file.name
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return ""

# -----------------------------
# Transcription (NEW: local broker + fallback), but KEEP old name
# -----------------------------
def _debug(msg: str):
    if DEBUG_TTS:
        print(f"[captions][debug] {msg}")

def _transcribe_local_broker(audio_file_path: str, language: str = "auto", prompt: Optional[str] = None) -> Dict:
    if not LOCAL_STT_URL:
        raise RuntimeError("LOCAL_STT_URL is not set")

    try:
        import requests
    except Exception as e:
        raise RuntimeError("requests is required for LOCAL STT broker calls (pip install requests)") from e

    _debug(f"Local STT -> POST {LOCAL_STT_URL} file={audio_file_path} lang={language}")

    with open(audio_file_path, "rb") as f:
        files = {"file": (os.path.basename(audio_file_path), f, "application/octet-stream")}
        data = {
            "model_name": "faster-whisper",
            "language": language or "auto",
            "task": "transcribe",
            "response_format": "verbose_json",
        }
        if prompt:
            data["prompt"] = prompt

        r = requests.post(LOCAL_STT_URL, files=files, data=data, timeout=LOCAL_STT_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"Local STT broker failed: {r.status_code} {r.text[:500]}")
        return r.json()

def _transcribe_openai(audio_file_path: str) -> Dict:
    """
    Supports BOTH:
    - legacy openai<1.0 style: openai.Audio.transcribe(...)
    - newer openai>=1.0 style: OpenAI().audio.transcriptions.create(...)
    Returns verbose_json-like dict (or {}).
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (needed for openai fallback)")

    # Try new SDK first
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        with open(audio_file_path, "rb") as audio_file:
            resp = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
            )
        # resp may be pydantic/object; normalize to dict
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if isinstance(resp, dict):
            return resp
        # last resort
        return json.loads(str(resp))
    except Exception as new_sdk_err:
        _debug(f"OpenAI new SDK path failed, trying legacy. err={type(new_sdk_err).__name__}: {new_sdk_err}")

    # Legacy SDK fallback
    try:
        import openai
        openai.api_key = api_key
        with open(audio_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
            )
        # legacy returns dict-like
        return response if isinstance(response, dict) else dict(response)
    except Exception as legacy_err:
        raise RuntimeError(f"OpenAI legacy SDK failed: {type(legacy_err).__name__}: {legacy_err}")

def transcribe_audio_whisper(audio_file_path: str) -> Dict:
    """
    BACKWARDS COMPAT ENTRYPOINT.
    appnew.py calls this name for caption generation. :contentReference[oaicite:2]{index=2}

    Behavior:
    - If TRANSCRIBE_PROVIDER=local (default) OR STT_BACKEND indicates broker usage,
      call LOCAL_STT_URL.
    - If that fails and TRANSCRIBE_FALLBACK=openai, call OpenAI Whisper.
    """
    # Decide provider
    provider = TRANSCRIBE_PROVIDER
    if STT_BACKEND in ("faster_whisper_api", "faster-whisper-api", "broker", "local"):
        provider = "local"

    language = os.getenv("STT_LANGUAGE", STT_LANGUAGE) or "auto"
    try:
        if provider == "local":
            return _transcribe_local_broker(audio_file_path, language=language, prompt=None)

        # explicit openai
        return _transcribe_openai(audio_file_path)

    except Exception as e:
        _debug(f"Primary transcription failed ({provider}): {type(e).__name__}: {e}")

        if TRANSCRIBE_FALLBACK == "openai":
            try:
                return _transcribe_openai(audio_file_path)
            except Exception as e2:
                _debug(f"Fallback transcription failed (openai): {type(e2).__name__}: {e2}")
                return {}

        return {}

# (Optional) alias if any other code expects these names
def transcribe_audio(audio_file_path: str) -> Dict:
    return transcribe_audio_whisper(audio_file_path)

def transcribe_audio_openai(audio_file_path: str) -> Dict:
    return _transcribe_openai(audio_file_path)

# -----------------------------
# Captions generation
# -----------------------------
def generate_captions_from_whisper(transcription: Dict) -> List[Dict]:
    captions = []
    for segment in transcription.get('segments', []):
        captions.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'].strip()
        })
    return captions

# -----------------------------
# Caption overlay helpers
# -----------------------------
def get_default_font() -> str:
    font_path = CAPTION_SETTINGS.get("FONT", "Bangers-Regular.ttf")
    font_path = str((BASE_DIR / font_path).resolve()) if not os.path.isabs(font_path) else font_path
    if os.path.isfile(font_path):
        return font_path
    return fm.findfont("DejaVu Sans")

def does_text_fit(text: str, fontsize: int, font: str, max_width: int) -> bool:
    try:
        txt_clip = TextClip(text, fontsize=fontsize, font=font, method='caption', size=(max_width, None), align='center')
        return txt_clip.w <= max_width
    except Exception:
        return False

def split_long_word(word: str, max_length: int = 15) -> List[str]:
    if len(word) <= max_length:
        return [word]
    parts = []
    while len(word) > max_length:
        parts.append(word[:max_length] + '-')
        word = word[max_length:]
    parts.append(word)
    return parts

def moviepy_to_pillow(clip) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp:
        clip.save_frame(temp.name)
        return Image.open(temp.name).convert("RGBA")

def _compute_caption_box(video_w: int, video_h: int) -> Tuple[Tuple[int, int], Tuple]:
    """Return (box_size, position)."""
    width = int(video_w * 0.8)
    height = int(video_h * 0.4)
    bottom_margin = int(video_h * 0.3)
    y_center = video_h - bottom_margin - height // 2
    return (width, height), ('center', y_center)

def add_captions_to_video(
    input_video_path: str,
    transcription: List[Dict],
    output_video_path: str,
    font_path: Optional[str] = None,
    fontsize: int = CAPTION_SETTINGS.get('TEXT_SIZE', 24),
    color: str = CAPTION_SETTINGS.get('COLOR', 'white'),
    stroke_color: str = CAPTION_SETTINGS.get('STROKE_COLOR', 'black'),
    stroke_width: int = CAPTION_SETTINGS.get('STROKE_WIDTH', 2),
    position: Optional[tuple] = None,
    blur_radius: int = 0,
    opacity: float = 1.0,
    max_words_per_caption: int = CAPTION_SETTINGS.get('MAX_WORDS_PER_CAPTION', 8),
    time_scale: float = 1.0,
    start_delay: float = 0.0,
    duration_adjust: float = 0.0,
    per_caption_offset: Optional[Dict[int, float]] = None
):
    try:
        video = VideoFileClip(input_video_path)
    except Exception as e:
        print(f"Error loading video: {e}")
        return

    if font_path is None:
        try:
            font_path = get_default_font()
        except Exception as e:
            print(e)
            return

    if not os.path.isfile(font_path):
        print(f"Font file not found at {font_path}")
        return

    box_size, dyn_pos = _compute_caption_box(video.w, video.h)
    max_caption_width = box_size[0]
    offsets = per_caption_offset or {}

    words_list = []
    for idx, seg in enumerate(transcription):
        text = seg['text']
        start = seg['start'] * time_scale + start_delay + offsets.get(idx, 0)
        end = seg['end'] * time_scale + start_delay + duration_adjust + offsets.get(idx, 0)
        words = text.split()
        if not words:
            continue
        dur = (end - start) / len(words)
        for i, w in enumerate(words):
            words_list.append({
                "word": w,
                "start": start + i * dur,
                "end": start + (i + 1) * dur
            })

    captions = []
    current, s, e = [], None, None
    for w in words_list:
        if not current:
            s = w['start']
        current.append(w['word'])
        e = w['end']
        if len(current) >= max_words_per_caption:
            captions.append({"start": s, "end": e, "text": " ".join(current)})
            current, s, e = [], None, None
    if current:
        captions.append({"start": s, "end": e, "text": " ".join(current)})

    processed = []
    for cap in captions:
        lines, line = [], ""
        for word in cap['text'].split():
            test = f"{line} {word}".strip()
            if does_text_fit(test, fontsize, font_path, max_caption_width):
                line = test
            else:
                if line:
                    lines.append(line)
                if does_text_fit(word, fontsize, font_path, max_caption_width):
                    line = word
                else:
                    for part in split_long_word(word):
                        if does_text_fit(part, fontsize, font_path, max_caption_width):
                            lines.append(part)
                    line = ""
        if line:
            lines.append(line)
        processed.append({"start": cap['start'], "end": cap['end'], "text": "\n".join(lines)})

    clips = []
    for cap in processed:
        try:
            txt = TextClip(
                txt=cap['text'],
                fontsize=fontsize,
                color=color,
                font=font_path,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                method='caption',
                size=box_size,
                align='center'
            )
        except Exception as e:
            print(f"Error creating TextClip: {e}")
            continue

        if opacity < 1.0:
            txt = txt.set_opacity(opacity)

        if blur_radius > 0:
            img = moviepy_to_pillow(txt)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            txt = ImageClip(np.array(img)).set_duration(cap['end'] - cap['start'])

        pos = position if position is not None else dyn_pos
        clips.append(txt.set_start(cap['start']).set_duration(cap['end'] - cap['start']).set_position(pos))

    final_video = CompositeVideoClip([video] + clips)
    try:
        final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
        print(f"Video with captions saved to {output_video_path}")
    except Exception as e:
        print(f"Error writing output video: {e}")

def main():
    parser = argparse.ArgumentParser(description='Add captions to video and optionally update workflow JSON.')
    parser.add_argument('json_file', nargs='?', help='Optional path to workflow JSON containing video path.')
    parser.add_argument('--input_video', help='Path to input video if not using JSON.')
    parser.add_argument('--output_video', help='Path to output video if not using JSON.')
    parser.add_argument('--font_path', default=None)
    parser.add_argument('--fontsize', type=int, default=CAPTION_SETTINGS.get('TEXT_SIZE', 24))
    parser.add_argument('--color', default=CAPTION_SETTINGS.get('COLOR', 'white'))
    parser.add_argument('--stroke_color', default=CAPTION_SETTINGS.get('STROKE_COLOR', 'black'))
    parser.add_argument('--stroke_width', type=int, default=CAPTION_SETTINGS.get('STROKE_WIDTH', 2))
    parser.add_argument('--position', nargs=2, type=int, metavar=('X', 'Y'), help='Caption position (x y).')
    parser.add_argument('--blur_radius', type=int, default=0)
    parser.add_argument('--opacity', type=float, default=1.0)
    parser.add_argument('--max_words_per_caption', type=int, default=CAPTION_SETTINGS.get('MAX_WORDS_PER_CAPTION', 8))
    parser.add_argument('--time_scale', type=float, default=1.0)
    parser.add_argument('--start_delay', type=float, default=0.0)
    parser.add_argument('--duration_adjust', type=float, default=0.0)
    parser.add_argument('--per_caption_offset', type=json.loads, default={})
    args = parser.parse_args()

    json_file_path = args.json_file
    if json_file_path:
        with open(json_file_path, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
        input_video = data.get('video') or data.get('video_path') or data.get('input_video') or data.get('final_video')
        if not input_video:
            print("Error: No input video path found in JSON.")
            sys.exit(1)
        base, ext = os.path.splitext(input_video)
        output_video = data.get('captions_video') or f"{base}_captions{ext}"
    else:
        if not args.input_video or not args.output_video:
            parser.print_help()
            sys.exit(1)
        input_video = args.input_video
        output_video = args.output_video

    audio_path = extract_audio(input_video)
    if not audio_path:
        sys.exit(1)

    transcription = transcribe_audio_whisper(audio_path)

    try:
        os.remove(audio_path)
    except Exception:
        pass

    captions_list = generate_captions_from_whisper(transcription)
    if not captions_list:
        print("No captions generated. Exiting.")
        sys.exit(1)

    add_captions_to_video(
        input_video_path=input_video,
        transcription=captions_list,
        output_video_path=output_video,
        font_path=args.font_path,
        fontsize=args.fontsize,
        color=args.color,
        stroke_color=args.stroke_color,
        stroke_width=args.stroke_width,
        position=tuple(args.position) if args.position else None,
        blur_radius=args.blur_radius,
        opacity=args.opacity,
        max_words_per_caption=args.max_words_per_caption,
        time_scale=args.time_scale,
        start_delay=args.start_delay,
        duration_adjust=args.duration_adjust,
        per_caption_offset=args.per_caption_offset
    )

if __name__ == "__main__":
    main()
