#!/usr/bin/env python3

import sys
import os
import json
import time
import logging
import requests
import threading
import subprocess
import re
import socket
from urllib.parse import quote
from dotenv import load_dotenv
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from social_tokens import get_instagram_token
load_dotenv()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

INSTAGRAM_ACCOUNT_ID = os.getenv('INSTAGRAM_ACCOUNT_ID')
PUBLIC_IP = os.getenv('PUBLIC_IP', '127.0.0.1')
BASE_HTTP_PORT = int(os.getenv('HTTP_PORT', '8301'))

def get_access_token():
    token = get_instagram_token()
    if token:
        return token
    logging.error("Failed to fetch Instagram access token. Check APP_ID/APP_SECRET/SHORT_LIVED_TOKEN or USER_ACCESS_TOKEN.")
    sys.exit(1)

def transcode_for_instagram(input_path):
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_ig.mp4"
    if os.path.exists(output_path):
        return output_path
    cmd = [
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264", "-profile:v", "baseline", "-level", "3.0",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

class RangeHTTPRequestHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        ctype = self.guess_type(path)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, "File not found")
            return None
        fs = os.fstat(f.fileno())
        size = fs.st_size
        start, end = 0, size - 1
        if "Range" in self.headers:
            m = re.match(r"bytes=(\d+)-(\d*)", self.headers["Range"])
            if m:
                start = int(m.group(1))
                if m.group(2):
                    end = int(m.group(2))
        self.send_response(206 if "Range" in self.headers else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.range = (start, end)
        return f

    def copyfile(self, source, outputfile):
        start, end = getattr(self, 'range', (0, None))
        remaining = None if end is None else (end - start + 1)
        bufsize = 64 * 1024
        while True:
            chunk = source.read(bufsize if remaining is None else min(bufsize, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break

def find_free_port(start_port):
    port = start_port
    while port < start_port + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                port += 1
    sys.exit(1)

def upload(json_path):
    if not os.path.exists(json_path):
        logging.error(f"JSON not found: {json_path}")
        sys.exit(1)

    token = get_access_token()
    with open(json_path, encoding='utf-8') as f:
        metadata = json.load(f)

    video_file = transcode_for_instagram(metadata.get("final_video", ""))
    social = metadata.get("social_media", {})
    title = social.get("title", "")
    description = social.get("description", "")
    tags = social.get("tags", [])

    # Get YouTube URL from top-level or social_media
    yt_url = metadata.get("youtube_url") or social.get("youtube_url")
    if yt_url and f"Watch the full video on YouTube: {yt_url}" not in description:
        description += f"\n\nWatch the full video on YouTube: {yt_url}"

    caption_parts = [title, description]
    if tags:
        caption_parts.append(", ".join(tags[:30]))
    caption = "\n\n".join([p for p in caption_parts if p])
    caption = caption[:2200]

    dirname, basename = os.path.split(video_file)
    os.chdir(dirname)
    port = find_free_port(BASE_HTTP_PORT)
    httpd = ThreadingHTTPServer(('', port), RangeHTTPRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    time.sleep(1)

    public_url = f"http://{PUBLIC_IP}:{port}/{quote(basename)}"
    media_url = f"https://graph.facebook.com/v17.0/{INSTAGRAM_ACCOUNT_ID}/media"
    payload = {
        "media_type": "REELS",
        "video_url": public_url,
        "caption": caption,
        "access_token": token
    }
    resp = requests.post(media_url, json=payload)
    creation_id = resp.json().get("id")

    publish_url = f"https://graph.facebook.com/v17.0/{INSTAGRAM_ACCOUNT_ID}/media_publish"
    for _ in range(15):
        pr = requests.post(publish_url, data={"creation_id": creation_id, "access_token": token})
        if pr.status_code == 200:
            print(pr.json())
            httpd.shutdown()
            thread.join()
            return
        time.sleep(10)
    httpd.shutdown()
    thread.join()
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python igupload.py <path_to_json>")
        sys.exit(1)
    upload(sys.argv[1])
