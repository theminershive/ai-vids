#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from moviepy.editor import CompositeVideoClip, ImageClip, VideoFileClip

from config import AppConfig, load_config

DEFAULT_BLUR_RADIUS = 10
ENABLE_BLUR = True
MAX_TEXT_WIDTH_RATIO = 0.80
TEXT_STROKE_COLOR = "black"
TEXT_STROKE_WIDTH = 2
MAX_VERSE_FONT_SIZE = 80
MIN_VERSE_FONT_SIZE = 24
INTER_CLIP_GAP = 40
MIN_FILL_RATIO = 0.60


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        line = []
        while words:
            line.append(words.pop(0))
            test = " ".join(line + words[:1])
            if draw.textbbox((0, 0), test, font=font)[2] > max_width:
                words.insert(0, line.pop())
                break
        lines.append(" ".join(line))
        while words:
            line = []
            while words:
                line.append(words.pop(0))
                test = " ".join(line + words[:1])
                if draw.textbbox((0, 0), test, font=font)[2] > max_width:
                    break
            lines.append(" ".join(line))
    return "\n".join(lines)


def apply_blur_behind_text(img: Image.Image, box: tuple[int, int, int, int], radius: int = DEFAULT_BLUR_RADIUS) -> Image.Image:
    region = img.crop(box)
    blurred = region.filter(ImageFilter.GaussianBlur(radius))
    img.paste(blurred, box)
    return img


def make_text_clip(
    text: str,
    font_path: Path,
    fontsize: int,
    color: str,
    max_width: int,
    padding: int,
    bg_color: tuple[int, int, int],
    opacity: float,
    duration: float,
    position,
    use_stroke: bool = False,
) -> ImageClip:
    font = ImageFont.truetype(str(font_path), fontsize)
    wrapped = wrap_text(text, font, max_width)
    tmp = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align="center", spacing=10)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    box_w, box_h = int(text_w + 2 * padding), int(text_h + 2 * padding)
    img_box = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img_box)
    rgba = tuple(bg_color) + (int(255 * opacity),)
    draw.rounded_rectangle([(0, 0), (box_w, box_h)], fill=rgba, radius=20)
    draw.multiline_text(
        ((box_w - text_w) // 2, (box_h - text_h) // 2),
        wrapped,
        font=font,
        fill=color,
        align="center",
        spacing=10,
        **({"stroke_width": TEXT_STROKE_WIDTH, "stroke_fill": TEXT_STROKE_COLOR} if use_stroke else {}),
    )
    clip = ImageClip(np.array(img_box)).set_duration(duration)
    clip = clip.set_position(position)
    clip._position = position
    return clip


def parse_position(args, default):
    if args:
        x, y = args
        try:
            x = int(x)
        except Exception:
            pass
        try:
            y = int(y)
        except Exception:
            pass
        return (x, y)
    return default


def overlay_verses(
    config: AppConfig,
    video_path: Path,
    json_path: Path,
    output_path: Path,
) -> None:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    section = data.get("sections", [{}])[0]
    original_name = section.get("original_name", "")
    raw = section.get("segments", [{}])[0].get("narration", {}).get("text", "")
    raw = re.sub(r"#(?:HOOK|BODY|OUTRO)\s*", "", raw).strip()

    video = VideoFileClip(str(video_path))
    vw, vh = video.size
    dur = video.duration
    max_w = int(vw * MAX_TEXT_WIDTH_RATIO)

    overlay_cfg = config.overlay

    title_y = overlay_cfg.name_pos[1]
    title_pos = ("center", title_y)
    title_clip = make_text_clip(
        original_name,
        overlay_cfg.name_font,
        overlay_cfg.name_fontsize,
        overlay_cfg.text_color,
        max_w,
        overlay_cfg.padding,
        overlay_cfg.bg_color,
        overlay_cfg.opacity,
        dur,
        title_pos,
        overlay_cfg.use_stroke,
    )

    verse_y = title_y + title_clip.size[1] + INTER_CLIP_GAP
    verse_pos = ("center", verse_y)
    min_bottom = int(vh * MIN_FILL_RATIO)

    fs = overlay_cfg.verse_fontsize
    verse_clip = None
    while fs <= MAX_VERSE_FONT_SIZE:
        temp = make_text_clip(
            raw,
            overlay_cfg.verse_font,
            fs,
            overlay_cfg.text_color,
            max_w,
            overlay_cfg.padding,
            overlay_cfg.bg_color,
            overlay_cfg.opacity,
            dur,
            verse_pos,
            overlay_cfg.use_stroke,
        )
        if verse_y + temp.size[1] >= min_bottom:
            verse_clip = temp
            break
        fs += 2
    if not verse_clip:
        verse_clip = make_text_clip(
            raw,
            overlay_cfg.verse_font,
            fs,
            overlay_cfg.text_color,
            max_w,
            overlay_cfg.padding,
            overlay_cfg.bg_color,
            overlay_cfg.opacity,
            dur,
            verse_pos,
            overlay_cfg.use_stroke,
        )
    while verse_y + verse_clip.size[1] > vh and fs > MIN_VERSE_FONT_SIZE:
        fs = max(fs - 2, MIN_VERSE_FONT_SIZE)
        verse_clip = make_text_clip(
            raw,
            overlay_cfg.verse_font,
            fs,
            overlay_cfg.text_color,
            max_w,
            overlay_cfg.padding,
            overlay_cfg.bg_color,
            overlay_cfg.opacity,
            dur,
            verse_pos,
            overlay_cfg.use_stroke,
        )

    cta_clip = None
    if overlay_cfg.show_cta:
        cta_pos = ("center", vh - 120)
        cta_clip = make_text_clip(
            config.channel.cta_text,
            overlay_cfg.verse_font,
            overlay_cfg.cta_fontsize,
            overlay_cfg.text_color,
            max_w,
            overlay_cfg.padding,
            overlay_cfg.bg_color,
            overlay_cfg.opacity,
            dur,
            cta_pos,
            overlay_cfg.use_stroke,
        )

    blur_items = [(title_clip, title_pos), (verse_clip, verse_pos)]
    if cta_clip:
        blur_items.append((cta_clip, cta_clip._position))

    if ENABLE_BLUR:
        frame = video.get_frame(0)
        bg_img = Image.fromarray(frame).convert("RGB")
        for clip, pos in blur_items:
            w, h = clip.size
            x, y = pos
            left = int((vw - w) / 2) if x == "center" else x
            box = (left, y, left + w, y + h)
            bg_img = apply_blur_behind_text(bg_img, box)
        bg_clip = ImageClip(np.array(bg_img)).set_duration(dur).set_fps(video.fps)
        final = [bg_clip, title_clip, verse_clip] + ([cta_clip] if cta_clip else [])
    else:
        final = [video, title_clip, verse_clip] + ([cta_clip] if cta_clip else [])

    comp = CompositeVideoClip(final).set_audio(video.audio)
    comp.write_videofile(str(output_path), codec="libx264", audio_codec="aac", fps=video.fps)

    data["final_video"] = str(output_path)
    Path(json_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-j", "--json", required=True)
    p.add_argument("-v", "--video", required=True)
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    config = load_config()
    overlay_verses(
        config,
        Path(args.video),
        Path(args.json),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
