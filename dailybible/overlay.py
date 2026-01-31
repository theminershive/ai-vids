#!/usr/bin/env python3
import os
import sys
import json
import argparse
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip

def add_text_overlay(input_video_path, output_video_path,
                     start_text, end_text,
                     end_duration,
                     start_font_path, end_font_path,
                     start_fontsize, end_fontsize,
                     text_color, bg_color, col_opacity, padding,
                     fade_duration=1,
                     position=None):
    """Adds start and end text overlays for the video with automatic fades."""
    try:
        video = VideoFileClip(input_video_path)
    except Exception as e:
        print(f"Error loading video: {e}")
        sys.exit(1)

    video_width, video_height = video.size
    full_duration = video.duration
    # compute start clip duration so its fade-out ends exactly when end-duration starts
    start_clip_duration = max(full_duration - end_duration + fade_duration, 0)
    pos = position if position is not None else ('center', int(video_height * 0.2))

    # Start TextClip
    try:
        txt_start = TextClip(
            txt=start_text,
            fontsize=start_fontsize,
            font=start_font_path,
            color=text_color,
            method='caption',
            size=(video_width - 2*padding, None),
            align='center'
        )
    except Exception as e:
        print(f"Error creating TextClip with font '{start_font_path}': {e}")
        sys.exit(1)
    bg_start = txt_start.on_color(
        size=(txt_start.w + 2*padding, txt_start.h + 2*padding),
        color=bg_color,
        pos=('center', 'center'),
        col_opacity=col_opacity
    )
    # set duration and crossfade out
    start_clip = bg_start.set_start(0).set_duration(start_clip_duration)
    start_clip = start_clip.crossfadeout(fade_duration).set_position(pos)

    # End TextClip
    try:
        txt_end = TextClip(
            txt=end_text,
            fontsize=end_fontsize,
            font=end_font_path or start_font_path,
            color=text_color,
            method='caption',
            size=(video_width - 2*padding, None),
            align='center'
        )
    except Exception as e:
        print(f"Error creating end TextClip with font '{end_font_path or start_font_path}': {e}")
        sys.exit(1)
    bg_end = txt_end.on_color(
        size=(txt_end.w + 2*padding, txt_end.h + 2*padding),
        color=bg_color,
        pos=('center', 'center'),
        col_opacity=col_opacity
    )
    end_start_time = max(full_duration - end_duration, 0)
    end_clip = bg_end.set_start(end_start_time).set_duration(end_duration)
    end_clip = end_clip.crossfadein(fade_duration).set_position(pos)

    # Composite
    out = CompositeVideoClip([video, start_clip, end_clip])
    try:
        out.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
    except Exception as e:
        print(f"Error writing video file: {e}")
        sys.exit(1)


def find_newest_json():
    cwd = os.getcwd()
    ready_dir = os.path.join(cwd, 'ready')
    if not os.path.isdir(ready_dir):
        return None
    jsons = [os.path.join(ready_dir, f) for f in os.listdir(ready_dir) if f.endswith('.json')]
    return max(jsons, key=os.path.getmtime) if jsons else None


def main():
    parser = argparse.ArgumentParser(description='Overlay text and update JSON final_video.')
    parser.add_argument('--input_video', help='Path to input video.')
    parser.add_argument('--output_video', help='Path to output video.')
    parser.add_argument('--start_text', default='', help='Text for start overlay.')
    parser.add_argument('--end_text', default='Thanks for watching! Subscribe!', help='Text for end overlay.')
    parser.add_argument('--end_duration', type=float, default=7.0, help='Duration for end overlay.')
    parser.add_argument('--start_font_path', default='', help='Font path for start overlay.')
    parser.add_argument('--end_font_path', default='', help='Font path for end overlay.')
    parser.add_argument('--start_fontsize', type=int, default=75, help='Font size for start overlay.')
    parser.add_argument('--end_fontsize', type=int, default=75, help='Font size for end overlay.')
    parser.add_argument('--text_color', default='white', help='Text color.')
    parser.add_argument('--bg_color', nargs=3, type=int, default=[0,0,0], help='Background color RGB.')
    parser.add_argument('--col_opacity', type=float, default=0.3, help='Background opacity.')
    parser.add_argument('--padding', type=int, default=5, help='Padding around text.')
    parser.add_argument('--fade_duration', type=float, default=1.0, help='Fade duration for both overlays.')
    parser.add_argument('--fade_in', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--fade_out', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--position', nargs=2, type=int, metavar=('X','Y'), help='Overlay position.')
    parser.add_argument('json_file', nargs='?', help='Optional workflow JSON to update.')
    args = parser.parse_args()

    json_path = args.json_file or find_newest_json()
    config = {}
    if json_path and os.path.isfile(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

    if args.input_video and args.output_video:
        inp, out = args.input_video, args.output_video
    else:
        vid_cfg = config.get('video', {})
        inp = vid_cfg.get('input') or vid_cfg.get('video') or vid_cfg.get('video_path') or config.get('final_video')
        out = vid_cfg.get('output') or vid_cfg.get('overlay_output')
        if not inp or not out:
            print("Error: video input/output not specified.")
            sys.exit(1)

    start_text = args.start_text or config.get('reference', '')
    end_text = args.end_text
    start_font = args.start_font_path or config.get('overlays', {}).get('start', {}).get('font_file', '')
    end_font = args.end_font_path or config.get('overlays', {}).get('end', {}).get('font_file', start_font)
    start_fs = args.start_fontsize
    end_fs = args.end_fontsize
    text_color = args.text_color
    bg_color = tuple(args.bg_color)
    col_opacity = args.col_opacity
    padding = args.padding
    fade_duration = args.fade_duration
    position = tuple(args.position) if args.position else None

    add_text_overlay(
        inp, out,
        start_text, end_text,
        args.end_duration,
        start_font, end_font,
        start_fs, end_fs,
        text_color, bg_color, col_opacity, padding,
        fade_duration,
        position
    )

    if json_path:
        config['overlay_video'] = os.path.abspath(out)
        config['final_video'] = os.path.abspath(out)
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            print(f"Updated JSON with final_video: {config['final_video']}")
        except Exception as e:
            print(f"Error updating JSON: {e}")

if __name__ == "__main__":
    main()
