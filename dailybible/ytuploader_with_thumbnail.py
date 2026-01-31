#!/usr/bin/env python3

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl"
]

def extract_thumbnail(video_path, output_dir="thumbnails", base_name="thumbnail", attempts=[2, 4, 6]):
    os.makedirs(output_dir, exist_ok=True)
    for sec in attempts:
        try:
            clip = VideoFileClip(video_path)
            frame = clip.get_frame(sec)
            clip.reader.close()
            if clip.audio:
                clip.audio.reader.close_proc()

            im = Image.fromarray(frame)
            grayscale = im.convert("L")
            avg_pixel = np.mean(np.asarray(grayscale))
            if avg_pixel > 15:
                output_path = os.path.join(output_dir, f"{base_name}_t{sec}.jpg")
                im.save(output_path)
                logging.info(f"Thumbnail extracted at {sec}s: {output_path} (avg pixel: {avg_pixel:.2f})")
                return output_path
            else:
                logging.warning(f"Skipped dark frame at {sec}s (avg pixel: {avg_pixel:.2f})")
        except Exception as e:
            logging.error(f"Error extracting frame at {sec}s: {e}")
    logging.error("All attempts failed to extract a usable thumbnail.")
    return None

def load_credentials():
    creds = None
    if os.path.exists("token2.json"):
        creds = Credentials.from_authorized_user_file("token2.json", SCOPES)
    if creds and creds.expired and creds.refresh_token:
        logging.info("Refreshing expired access token...")
        creds.refresh(Request())
    if not creds or not creds.valid:
        logging.error("token2.json missing or invalid.")
        raise RuntimeError("Invalid YouTube credentials.")
    return creds

def post_publish_thumbnail(youtube, video_id, thumb_path, publish_time):
    delay = (publish_time - datetime.now(timezone.utc)).total_seconds() + 120
    logging.info(f"Waiting {int(delay)}s until after publish to re-upload thumbnail...")
    if delay > 0:
        time.sleep(delay)
    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumb_path)
        ).execute()
        logging.info("Thumbnail re-uploaded after publish.")
    except Exception as e:
        logging.error(f"Post-publish thumbnail update failed: {e}")

def upload(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        metadata = json.load(f)

    video_file = metadata.get("final_video")
    if not video_file or not os.path.exists(video_file):
        final_dir = Path("final")
        vids = list(final_dir.glob("*.*"))
        if not vids:
            raise FileNotFoundError("No video files found in final directory.")
        video_file = str(max(vids, key=lambda p: p.stat().st_mtime))

    social = metadata.get("social_media", {})
    title = social.get("title", "Default Title")
    description = social.get("description", "")
    tags = social.get("tags", [])

    creds = load_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=15)
    publish_at = scheduled_time.isoformat().replace("+00:00", "Z")

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags[:500],
            "categoryId": "22"
        },
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
            "publishAt": publish_at
        }
    }

    media = MediaFileUpload(video_file, mimetype="video/*", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            logging.info(f"Upload progress: {int(status.progress() * 100)}%")

    video_id = response.get("id")
    if not video_id:
        logging.error("Failed to retrieve video ID after upload.")
        raise RuntimeError("YouTube upload succeeded but no video ID returned.")

    yt_url = f"https://youtu.be/{video_id}"

    thumb_path = extract_thumbnail(video_file, base_name=Path(video_file).stem)
    if thumb_path and os.path.exists(thumb_path):
        success = False
        for attempt in range(5):
            try:
                time.sleep(30)
                youtube.thumbnails().set(
                    videoId=video_id,
                    media_body=MediaFileUpload(thumb_path)
                ).execute()
                logging.info(f"Thumbnail uploaded successfully on attempt {attempt + 1}")

                time.sleep(15)
                meta = youtube.videos().list(part="snippet", id=video_id).execute()
                thumb_info = meta["items"][0]["snippet"].get("thumbnails", {})
                if thumb_info:
                    logging.info(f"Confirmed thumbnail is visible: {list(thumb_info.keys())}")
                    success = True
                    break
                else:
                    logging.warning("Thumbnail not yet visible, will retry...")
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
        if not success:
            logging.error("All thumbnail upload attempts failed or not confirmed.")
        else:
            post_publish_thumbnail(youtube, video_id, thumb_path, scheduled_time)

    social['youtube_url'] = yt_url
    metadata['social_media'] = social
    metadata['youtube_url'] = yt_url

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Updated JSON metadata with YouTube URL at {json_path}")

    return yt_url

if __name__ == "__main__":
    ready_dir = Path("ready")
    json_files = list(ready_dir.glob("*.json"))
    if not json_files:
        logging.error("No JSON files found in ready directory.")
        sys.exit(1)

    latest_json = str(max(json_files, key=lambda p: p.stat().st_mtime))
    try:
        upload(latest_json)
    except Exception as e:
        logging.error(e)
        sys.exit(1)
