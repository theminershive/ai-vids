#!/usr/bin/env python3

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

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

def load_credentials():
    creds = None
    if os.path.exists("token2.json"):
        creds = Credentials.from_authorized_user_file("token2.json", SCOPES)
    if creds and creds.expired and creds.refresh_token:
        logging.info("Refreshing expired access token...")
        creds.refresh(Request())
        with open("token2.json", "w") as token:
            token.write(creds.to_json())
    if not creds or not creds.valid:
        logging.error("token2.json missing or invalid.")
        raise RuntimeError("Invalid YouTube credentials.")
    return creds

def upload(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r") as f:
        metadata = json.load(f)

    # Determine which video file to upload
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

    # Build YouTube client
    creds = load_credentials()
    youtube = build("youtube", "v3", credentials=creds)

    # Schedule publish 10 minutes from now
    scheduled_time = datetime.now(timezone.utc) + timedelta(minutes=10)
    publish_at = scheduled_time.isoformat().replace("+00:00", "Z")

    # Prepare upload body
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags[:500],
            "categoryId": "22"
        },
        "status": {
            # must be private when scheduling
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
            "publishAt": publish_at
        }
    }

    media = MediaFileUpload(video_file, mimetype="video/*", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    # Upload with progress logging
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
    logging.info(f"YouTube upload scheduled: {yt_url} (will go live at {publish_at})")

    # Update JSON metadata
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
