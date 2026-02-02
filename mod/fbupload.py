#!/usr/bin/env python3

import sys
import os
import requests
import json
import logging
from dotenv import load_dotenv

from social_tokens import get_page_access_token

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FACEBOOK_PAGE_ID = os.getenv('FACEBOOK_PAGE_ID')
GRAPH_API_BASE = 'https://graph.facebook.com'

def _resolve_page_token():
    token = get_page_access_token(FACEBOOK_PAGE_ID)
    if token:
        return token
    logging.error("Could not resolve page access token. Check APP_ID/APP_SECRET/SHORT_LIVED_TOKEN or USER_ACCESS_TOKEN.")
    sys.exit(1)

def upload(json_path):
    if not os.path.exists(json_path):
        logging.error(f"JSON file not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        metadata = json.load(f)

    page_token = _resolve_page_token()
    video_file = metadata.get('final_video')
    if not video_file or not os.path.exists(video_file):
        logging.error(f"Video file not found: {video_file}")
        sys.exit(1)

    social = metadata.get('social_media', {})
    title = social.get('title', '')
    description = social.get('description', '')

    yt_url = metadata.get("youtube_url") or social.get("youtube_url")
    if yt_url and f"Watch the full video on YouTube: {yt_url}" not in description:
        description += f"\n\nWatch the full video on YouTube: {yt_url}"

    data = {
        'title': title,
        'description': description,
        'published': 'true',
        'temporary': 'true',
        'access_token': page_token
    }
    files = {'source': open(video_file, 'rb')}
    resp = requests.post(f"{GRAPH_API_BASE}/v21.0/{FACEBOOK_PAGE_ID}/videos", data=data, files=files)
    if resp.status_code != 200:
        logging.error(f"Upload failed: {resp.status_code} {resp.text}")
        sys.exit(1)
    print(json.dumps(resp.json(), indent=2))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python fbupload.py <path_to_json>")
        sys.exit(1)
    upload(sys.argv[1])
