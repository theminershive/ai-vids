#!/usr/bin/env python3
"""
YouTube Channel Backup Script

Configure your API key, Channel ID, and output locations below or via a .env file.
Each video will be saved along with its own metadata JSON file.
"""
import os
import sys
import json
import logging
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from yt_dlp import YoutubeDL
from dotenv import load_dotenv

# Load .env file if available
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# === Configuration ===
API_KEY = os.getenv('YOUTUBE_API_KEY')
CHANNEL_ID = os.getenv('YOUTUBE_CHANNEL_ID')
OUTPUT_DIR = os.path.abspath(os.getenv('YOUTUBE_BACKUP_DIR', 'youtube_backup'))
# ======================

if not API_KEY or not CHANNEL_ID:
    print("Error: YOUTUBE_API_KEY and YOUTUBE_CHANNEL_ID must be set in .env or environment.")
    sys.exit(1)


def setup_logging():
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
    )


def get_channel_uploads_playlist_id(youtube):
    try:
        response = youtube.channels().list(
            part='contentDetails,snippet',
            id=CHANNEL_ID
        ).execute()
    except HttpError as e:
        logging.error(f"YouTube API error: {e}")
        sys.exit(1)

    items = response.get('items', [])
    if not items:
        logging.error(f"No channel found for ID: {CHANNEL_ID}")
        sys.exit(1)

    channel = items[0]
    uploads_id = channel['contentDetails']['relatedPlaylists']['uploads']
    channel_title = channel['snippet']['title']
    return uploads_id, channel_title


def fetch_video_ids(youtube, uploads_playlist_id):
    video_ids = []
    next_token = None
    while True:
        res = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_token
        ).execute()
        video_ids.extend(i['contentDetails']['videoId'] for i in res.get('items', []))
        next_token = res.get('nextPageToken')
        if not next_token:
            break
    return video_ids


def fetch_video_metadata(youtube, video_ids):
    metadata = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        logging.info(f"Fetching metadata for videos {i+1}-{i+len(batch)}")
        response = youtube.videos().list(
            part='snippet,contentDetails,statistics,status',
            id=','.join(batch)
        ).execute()
        for item in response.get('items', []):
            metadata[item['id']] = item
    return metadata


def save_individual_metadata(metadata, output_dir):
    for video_id, data in metadata.items():
        title = data['snippet'].get('title', f"video_{video_id}").replace('/', '_')
        safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_title} [{video_id}].json"
        out_path = os.path.join(output_dir, filename)

        if os.path.exists(out_path):
            logging.info(f"Skipping metadata for {video_id}, already exists: {out_path}")
            continue

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved metadata: {out_path}")


def download_videos(video_ids, output_dir, metadata):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'outtmpl': os.path.join(output_dir, '%(title)s [%(id)s].%(ext)s'),
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4'
        }],
        'ignoreerrors': True,
        'noplaylist': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        for vid in video_ids:
            title = metadata[vid]['snippet'].get('title', f"video_{vid}").replace('/', '_')
            safe_title = ''.join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            expected_filename = os.path.join(output_dir, f"{safe_title} [{vid}].mp4")

            if os.path.exists(expected_filename):
                logging.info(f"Skipping {vid}, already exists: {expected_filename}")
                continue

            url = f"https://www.youtube.com/watch?v={vid}"
            logging.info(f"Downloading {vid}...")
            try:
                ydl.download([url])
            except Exception as e:
                logging.warning(f"Failed to download {vid}: {e}")


def main():
    setup_logging()
    logging.info(f"Starting backup for channel {CHANNEL_ID} to '{OUTPUT_DIR}'")

    try:
        youtube = build('youtube', 'v3', developerKey=API_KEY)
    except Exception as e:
        logging.error(f"Failed to initialize YouTube API client: {e}")
        sys.exit(1)

    uploads_id, channel_title = get_channel_uploads_playlist_id(youtube)
    safe_channel_title = ''.join(c for c in channel_title if c.isalnum() or c in (' ', '-', '_')).strip()
    base_dir = os.path.join(OUTPUT_DIR, safe_channel_title)
    videos_dir = os.path.join(base_dir, 'videos')
    meta_dir = os.path.join(base_dir, 'metadata')

    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    video_ids = fetch_video_ids(youtube, uploads_id)
    logging.info(f"Found {len(video_ids)} videos to back up.")

    metadata = fetch_video_metadata(youtube, video_ids)
    save_individual_metadata(metadata, meta_dir)
    download_videos(video_ids, videos_dir, metadata)

    logging.info("Backup complete.")


if __name__ == '__main__':
    main()
