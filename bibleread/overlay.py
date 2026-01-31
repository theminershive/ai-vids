import sys
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
import re

# === CONFIGURABLE SETTINGS ===
DEFAULT_BLUR_RADIUS = 10           # Radius of Gaussian blur under text boxes
ENABLE_BLUR = True                 # Toggle to enable/disable blur
DEFAULT_OPACITY = 0.5              # Background box opacity (0 to 1)
MAX_TEXT_WIDTH_RATIO = 0.80        # Text width as % of video width
TEXT_STROKE_COLOR = 'black'        # Text outline color
TEXT_STROKE_WIDTH = 2              # Text outline thickness


def wrap_text(text, font, max_width):
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    tag_pattern = r"(?:#HOOK|#BODY|#OUTRO)\s*"
    text = re.sub(tag_pattern, "\n\n", text).strip()
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        words, line = paragraph.split(), []
        while words:
            line.append(words.pop(0))
            test_line = " ".join(line + words[:1])
            if draw.textbbox((0, 0), test_line, font=font)[2] > max_width:
                words.insert(0, line.pop())
                break
        lines.append(" ".join(line))
        while words:
            line = []
            while words:
                line.append(words.pop(0))
                test_line = " ".join(line + words[:1])
                if draw.textbbox((0, 0), test_line, font=font)[2] > max_width:
                    break
            lines.append(" ".join(line))
    return "\n".join(lines)


def apply_blur_behind_text(img, box, radius=DEFAULT_BLUR_RADIUS):
    region = img.crop(box)
    blurred = region.filter(ImageFilter.GaussianBlur(radius))
    img.paste(blurred, box)
    return img


def make_text_clip(text, font_path, fontsize, color, max_width,
                   padding, bg_color, opacity, duration, position,
                   use_stroke=False):
    wrapped = wrap_text(text, ImageFont.truetype(font_path, fontsize), max_width)
    tmp = Image.new("RGBA", (10,10))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.multiline_textbbox((0,0), wrapped,
                                   font=ImageFont.truetype(font_path, fontsize),
                                   align="center", spacing=10)
    text_w, text_h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    box_w, box_h = int(text_w + 2*padding), int(text_h + 2*padding)
    img_box = Image.new("RGBA", (box_w, box_h), (0,0,0,0))
    draw = ImageDraw.Draw(img_box)
    rgba = tuple(bg_color) + (int(255*opacity),)
    draw.rounded_rectangle([0,0,box_w,box_h], fill=rgba, radius=20)
    draw.multiline_text(((box_w-text_w)//2, (box_h-text_h)//2), wrapped,
                        font=ImageFont.truetype(font_path, fontsize),
                        fill=color, align="center", spacing=10,
                        **({'stroke_width':TEXT_STROKE_WIDTH, 'stroke_fill':TEXT_STROKE_COLOR} if use_stroke else {}))
    arr = np.array(img_box)
    clip = ImageClip(arr).set_duration(duration).set_position(position)
    clip.size = (box_w, box_h)
    return clip


def parse_position(pos_args, default):
    if pos_args and len(pos_args)==2:
        x, y = pos_args
        try: y = int(y)
        except: pass
        return (x, y)
    return default


def overlay_verses(video_path, json_path, output_path,
                   name_font, verse_font,
                   name_fontsize, verse_fontsize,
                   text_color, bg_color, opacity,
                   name_pos, verse_pos, padding,
                   show_cta, cta_text, cta_fontsize,
                   use_stroke):
    config = json.loads(Path(json_path).read_text(encoding='utf-8'))
    section = config.get('sections', [{}])[0]
    title_text = section.get('original_name', '')
    verse_text = section.get('segments', [{}])[0].get('narration', {}).get('text', '')

    video = VideoFileClip(str(video_path))
    vw, vh = video.size
    duration = video.duration
    max_w = int(vw * MAX_TEXT_WIDTH_RATIO)

    # Title clip
    title_clip = make_text_clip(title_text, name_font, name_fontsize,
                                text_color, max_w, padding, bg_color,
                                opacity, duration, name_pos, use_stroke)
    # Verse clip (initial)
    verse_clip = make_text_clip(verse_text, verse_font, verse_fontsize,
                                text_color, max_w, padding, bg_color,
                                opacity, duration, verse_pos, use_stroke)
    # CTA clip
    cta_clip, cta_pos = None, None
    if show_cta:
        cta_pos = ('center', vh - 120)
        cta_clip = make_text_clip(cta_text, verse_font, cta_fontsize,
                                   text_color, max_w, padding,
                                   bg_color, opacity, duration,
                                   cta_pos, use_stroke)

    # Dynamic verse fit
    name_x = (vw - title_clip.size[0])//2 if name_pos[0]=='center' else name_pos[0]
    name_y = name_pos[1]
    name_bottom = name_y + title_clip.size[1]
    top_limit = vh - padding - (cta_pos[1] if cta_pos else vh)
    available = (cta_pos[1] if cta_pos else vh) - name_bottom - 2*padding
    vw2, vh2 = verse_clip.size
    if available>0 and vh2>available:
        scale = available / vh2
        new_fs = max(10, int(verse_fontsize * scale))
        verse_clip = make_text_clip(verse_text, verse_font, new_fs,
                                    text_color, max_w, padding,
                                    bg_color, opacity,
                                    duration, verse_pos, use_stroke)
        vw2, vh2 = verse_clip.size
    verse_x = (vw - vw2)//2
    verse_y = name_bottom + padding + max(0, (available-vh2)//2)
    verse_pos_actual = (verse_x, int(verse_y))
    verse_clip = verse_clip.set_position(verse_pos_actual)

    # Prepare clips
    clips = [video, title_clip.set_position((name_x, name_y)), verse_clip]
    if cta_clip:
        cw, ch = cta_clip.size
        cx = (vw - cw)//2
        cy = cta_pos[1]
        clips.append(cta_clip.set_position((cx, cy)))

    # Blur underlying boxes
    if ENABLE_BLUR:
        frame = video.get_frame(0)
        bg_img = Image.fromarray(frame).convert('RGB')
        boxes = []
        boxes.append((name_x, name_y, name_x+title_clip.size[0], name_y+title_clip.size[1]))
        boxes.append((verse_x, verse_y, verse_x+vw2, verse_y+vh2))
        if cta_clip:
            boxes.append((cx, cy, cx+cw, cy+ch))
        for box in boxes:
            bg_img = apply_blur_behind_text(bg_img, box)
        blurred_bg = ImageClip(np.array(bg_img)).set_duration(duration).set_fps(video.fps)
        # Overlay text on blurred frame
        comp = CompositeVideoClip([blurred_bg, *clips[1:]]).set_audio(video.audio)
    else:
        comp = CompositeVideoClip(clips).set_audio(video.audio)

    comp.write_videofile(str(output_path), codec='libx264', audio_codec='aac', fps=video.fps)

    config['final_video'] = str(output_path)
    Path(json_path).write_text(json.dumps(config, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description='Overlay text with blur')
    p.add_argument('--json','-j',required=True)
    p.add_argument('--video','-v',required=True)
    p.add_argument('--output','-o',required=True)
    p.add_argument('--name_font', default='./fonts/Anton-Regular.ttf')
    p.add_argument('--verse_font', default='./fonts/Montserrat-Bold.ttf')
    p.add_argument('--name_fontsize', type=int, default=70)
    p.add_argument('--verse_fontsize', type=int, default=45)
    p.add_argument('--cta_fontsize', type=int, default=35)
    p.add_argument('--text_color', default='white')
    p.add_argument('--bg_color', nargs=3, type=int, default=[0,0,0])
    p.add_argument('--opacity', type=float, default=DEFAULT_OPACITY)
    p.add_argument('--padding', type=int, default=30)
    p.add_argument('--name_pos', nargs=2, metavar=('X','Y'), default=['center','100'])
    p.add_argument('--verse_pos', nargs=2, metavar=('X','Y'), default=['center','480'])
    p.add_argument('--cta_text', default='Subscribe for Daily Verses')
    p.add_argument('--no_cta', action='store_true')
    p.add_argument('--use_stroke', action='store_true')
    args = p.parse_args()
    name_pos = parse_position(args.name_pos, ('center', 100))
    verse_pos = parse_position(args.verse_pos, ('center', 480))
    overlay_verses(
        Path(args.video), Path(args.json), Path(args.output),
        args.name_font, args.verse_font,
        args.name_fontsize, args.verse_fontsize,
        args.text_color, tuple(args.bg_color), args.opacity,
        name_pos, verse_pos, args.padding,
        not args.no_cta, args.cta_text, args.cta_fontsize,
        args.use_stroke
    )

if __name__ == '__main__':
    main()
