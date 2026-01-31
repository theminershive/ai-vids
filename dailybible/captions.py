import sys
import os
import json
import argparse
import tempfile
from typing import List, Dict, Optional
from moviepy.editor import TextClip, ImageClip, VideoFileClip, CompositeVideoClip
from PIL import Image, ImageFilter
import matplotlib.font_manager as fm
import numpy as np
import openai
from dotenv import load_dotenv
from config import CAPTION_SETTINGS, BASE_DIR

# Load environment variables
load_dotenv()

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

def transcribe_audio_whisper(audio_file_path: str) -> Dict:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY.")
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json"
            )
        return response
    except Exception as e:
        print(f"Error transcribing audio with Whisper: {e}")
        return {}

def generate_captions_from_whisper(transcription: Dict) -> List[Dict]:
    captions = []
    for segment in transcription.get('segments', []):
        captions.append({
            "start": segment['start'],
            "end": segment['end'],
            "text": segment['text'].strip()
        })
    return captions

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

def _compute_caption_box(video_w: int, video_h: int):
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
    except:
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

    if json_file_path:
        try:
            data['captions_video'] = os.path.abspath(output_video)
            data['final_video'] = os.path.abspath(output_video)
            with open(json_file_path, 'w', encoding='utf-8') as jf:
                json.dump(data, jf, indent=4)
            print(f"Updated JSON with final video path: {json_file_path}")
        except Exception as e:
            print(f"Error updating JSON file: {e}")

if __name__ == "__main__":
    main()
