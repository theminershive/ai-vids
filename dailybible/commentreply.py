#!/usr/bin/env python3

import os
import time
import json
import logging
import requests

from dotenv import load_dotenv
import openai

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ── CONFIG ─────────────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# YouTube (existing workflow)
TOKEN_FILE            = './token2.json'                   # existing YouTube OAuth token
YOUTUBE_SCOPES        = ['https://www.googleapis.com/auth/youtube.force-ssl']
YOUTUBE_CHANNEL_ID    = os.getenv('YOUTUBE_CHANNEL_ID', 'UCmM96VHTjKYPcIKCT4LvR4w')
REPLIED_COMMENTS_FILE = 'replied_comments.json'           # existing state file
AUTHOR_HISTORY_FILE   = 'author_reply_counts.json'        # existing state file
REPLIED_VIDEOS_FILE   = 'replied_videos.json'             # existing state file

# Facebook Page setup
FB_ACCESS_TOKEN       = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
FB_GRAPH_URL          = 'https://graph.facebook.com/v17.0'

# Instagram Business/Creator setup (fallback to FB token if not provided)
IG_ACCESS_TOKEN       = os.getenv("INSTAGRAM_ACCESS_TOKEN", FB_ACCESS_TOKEN)
IG_GRAPH_URL          = 'https://graph.facebook.com/v17.0'

# runtime parameters
LOG_LEVEL      = logging.INFO
SLEEP_MINUTES  = 20
YT_MAX_VIDEOS  = 50
FB_MAX_POSTS   = 50
IG_MAX_MEDIA   = 50

# ── LOGGING ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ── FILE I/O HELPERS ───────────────────────────────────────────────────────────
def load_json_file(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed loading JSON {path}: {e}")
        return []

def save_json_file(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logging.error(f"Failed saving JSON {path}: {e}")

# ── FACEBOOK PAGE ID FALLBACK ─────────────────────────────────────────────────
def get_facebook_page_id():
    pid = os.getenv("FB_PAGE_ID")
    if not pid:
        try:
            resp = requests.get(
                f"{FB_GRAPH_URL}/me/accounts",
                params={"access_token": FB_ACCESS_TOKEN}
            ).json()
            pages = resp.get("data", [])
            if pages:
                pid = pages[0].get("id")
                logging.info(f"Derived Facebook Page ID: {pid}")
        except Exception as e:
            logging.error(f"Error deriving FB_PAGE_ID: {e}")
    if not pid:
        raise RuntimeError("Facebook Page ID not set and could not derive from token.")
    return pid

# ── INSTAGRAM ACCOUNT ID FALLBACK ──────────────────────────────────────────────
def get_instagram_account_id(page_id):
    iid = os.getenv("INSTAGRAM_ACCOUNT_ID")
    if not iid:
        try:
            resp = requests.get(
                f"{FB_GRAPH_URL}/{page_id}",
                params={"fields": "instagram_business_account", "access_token": FB_ACCESS_TOKEN}
            ).json()
            igacct = resp.get("instagram_business_account", {})
            iid = igacct.get("id")
            logging.info(f"Derived Instagram Account ID: {iid}")
        except Exception as e:
            logging.error(f"Error deriving INSTAGRAM_ACCOUNT_ID: {e}")
    if not iid:
        raise RuntimeError("Instagram Account ID not set and could not derive from Facebook page.")
    return iid

# derive FB and IG ids
try:
    fb_page_id = get_facebook_page_id()
    ig_user_id = get_instagram_account_id(fb_page_id)
except Exception as e:
    logging.critical(f"Startup failure deriving IDs: {e}")
    raise

# ── YOUTUBE AUTH & HELPERS ────────────────────────────────────────────────────
def get_youtube_service():
    if not os.path.exists(TOKEN_FILE):
        raise RuntimeError(f"Missing YouTube token file: {TOKEN_FILE}")
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, YOUTUBE_SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
            logging.info("Refreshed YouTube credentials.")
        elif not creds.valid:
            raise RuntimeError("Invalid YouTube credentials.")
        return build('youtube', 'v3', credentials=creds)
    except Exception as e:
        logging.critical(f"YouTube auth failure: {e}")
        raise

def get_youtube_comments(youtube):
    try:
        resp = youtube.commentThreads().list(
            part='snippet',
            allThreadsRelatedToChannelId=YOUTUBE_CHANNEL_ID,
            textFormat='plainText',
            maxResults=50
        ).execute()
        items = resp.get('items', [])
        logging.info(f"Retrieved {len(items)} YouTube comments.")
        return items
    except Exception as e:
        logging.error(f"Error retrieving YouTube comments: {e}")
        return []

def reply_to_youtube_comment(youtube, comment_id, text):
    try:
        youtube.comments().insert(
            part='snippet',
            body={'snippet': {'parentId': comment_id, 'textOriginal': text}}
        ).execute()
        logging.info(f"Replied to YouTube comment {comment_id}")
    except Exception as e:
        logging.error(f"Failed to reply to YouTube comment {comment_id}: {e}")

def get_youtube_videos(youtube):
    try:
        resp = youtube.search().list(
            part='id',
            channelId=YOUTUBE_CHANNEL_ID,
            order='date',
            type='video',
            maxResults=YT_MAX_VIDEOS
        ).execute()
        vids = [i['id']['videoId'] for i in resp.get('items', [])]
        logging.info(f"Retrieved {len(vids)} YouTube videos.")
        return vids
    except Exception as e:
        logging.error(f"Error retrieving YouTube videos: {e}")
        return []

def comment_on_youtube_video(youtube, video_id, text):
    try:
        youtube.commentThreads().insert(
            part='snippet',
            body={'snippet': {'videoId': video_id, 'topLevelComment': {'snippet': {'textOriginal': text}}}}
        ).execute()
        logging.info(f"Posted pastor comment on YouTube video {video_id}")
    except Exception as e:
        logging.error(f"Failed to post pastor comment on YouTube video {video_id}: {e}")

def get_youtube_video_title(youtube, video_id):
    try:
        resp = youtube.videos().list(part='snippet', id=video_id).execute()
        items = resp.get('items', [])
        if items:
            return items[0]['snippet']['title']
    except Exception as e:
        logging.error(f"Error retrieving title for YouTube video {video_id}: {e}")
    return ""

# ── FACEBOOK HELPERS ──────────────────────────────────────────────────────────
def get_facebook_posts():
    try:
        resp = requests.get(
            f"{FB_GRAPH_URL}/{fb_page_id}/posts",
            params={'access_token': FB_ACCESS_TOKEN, 'limit': FB_MAX_POSTS}
        ).json()
        posts = [p['id'] for p in resp.get('data', [])]
        logging.info(f"Retrieved {len(posts)} Facebook posts.")
        return posts
    except Exception as e:
        logging.error(f"Error retrieving Facebook posts: {e}")
        return []

def get_facebook_comments(post_id):
    try:
        resp = requests.get(
            f"{FB_GRAPH_URL}/{post_id}/comments",
            params={'access_token': FB_ACCESS_TOKEN}
        ).json()
        comments = resp.get('data', [])
        logging.info(f"Retrieved {len(comments)} comments on Facebook post {post_id}")
        return comments
    except Exception as e:
        logging.error(f"Error retrieving FB comments for {post_id}: {e}")
        return []

def reply_to_facebook_comment(comment_id, text):
    try:
        requests.post(
            f"{FB_GRAPH_URL}/{comment_id}/comments",
            params={'message': text, 'access_token': FB_ACCESS_TOKEN}
        ).raise_for_status()
        logging.info(f"Replied to Facebook comment {comment_id}")
    except Exception as e:
        logging.error(f"Failed to reply FB comment {comment_id}: {e}")

def comment_on_facebook_post(post_id, text):
    try:
        requests.post(
            f"{FB_GRAPH_URL}/{post_id}/comments",
            params={'message': text, 'access_token': FB_ACCESS_TOKEN}
        ).raise_for_status()
        logging.info(f"Posted pastor comment on Facebook post {post_id}")
    except Exception as e:
        logging.error(f"Failed to post pastor comment on FB post {post_id}: {e}")

def get_facebook_post_message(post_id):
    try:
        resp = requests.get(
            f"{FB_GRAPH_URL}/{post_id}",
            params={'fields': 'message', 'access_token': FB_ACCESS_TOKEN}
        ).json()
        return resp.get('message', '')
    except Exception as e:
        logging.error(f"Error retrieving FB post message {post_id}: {e}")
        return ""

# ── INSTAGRAM HELPERS ─────────────────────────────────────────────────────────
def get_instagram_media():
    try:
        resp = requests.get(
            f"{IG_GRAPH_URL}/{ig_user_id}/media",
            params={'access_token': IG_ACCESS_TOKEN, 'limit': IG_MAX_MEDIA}
        ).json()
        medias = [m['id'] for m in resp.get('data', [])]
        logging.info(f"Retrieved {len(medias)} Instagram media items.")
        return medias
    except Exception as e:
        logging.error(f"Error retrieving IG media: {e}")
        return []

def get_instagram_comments(media_id):
    try:
        resp = requests.get(
            f"{IG_GRAPH_URL}/{media_id}/comments",
            params={'access_token': IG_ACCESS_TOKEN}
        ).json()
        comments = resp.get('data', [])
        logging.info(f"Retrieved {len(comments)} comments on IG media {media_id}")
        return comments
    except Exception as e:
        logging.error(f"Error retrieving IG comments for {media_id}: {e}")
        return []

def reply_to_instagram_comment(comment_id, text):
    try:
        requests.post(
            f"{IG_GRAPH_URL}/{comment_id}/replies",
            params={'message': text, 'access_token': IG_ACCESS_TOKEN}
        ).raise_for_status()
        logging.info(f"Replied to Instagram comment {comment_id}")
    except Exception as e:
        logging.error(f"Failed to reply to IG comment {comment_id}: {e}")

def comment_on_instagram_media(media_id, text):
    try:
        requests.post(
            f"{IG_GRAPH_URL}/{media_id}/comments",
            params={'message': text, 'access_token': IG_ACCESS_TOKEN}
        ).raise_for_status()
        logging.info(f"Posted pastor comment on IG media {media_id}")
    except Exception as e:
        logging.error(f"Failed to post pastor comment on IG media {media_id}: {e}")

def get_instagram_media_caption(media_id):
    try:
        resp = requests.get(
            f"{IG_GRAPH_URL}/{media_id}",
            params={'fields': 'caption', 'access_token': IG_ACCESS_TOKEN}
        ).json()
        return resp.get('caption', '')
    except Exception as e:
        logging.error(f"Error retrieving caption for IG media {media_id}: {e}")
        return ""

# ── GPT FUNCTIONS ─────────────────────────────────────────────────────────────
def generate_reply_via_gpt(site, author, text, count, title):
    prompt = f"""You are a friendly Christian moderator replying on {site}.
Title/Caption: {title}
Comment author: {author}
Previous replies: {count}
Comment: "{text}"

Write a short, kind reply reflecting Christian values.
""" 
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": f"You reply to comments on {site}."},
                      {"role": "user",   "content": prompt}],
            temperature=0.7, max_tokens=100
        )
        return resp.choices[0].message['content'].strip()
    except Exception as e:
        logging.error(f"GPT error on {site} reply: {e}")
        return "Thank you for your thoughtful comment!"


def generate_pastor_comment(title):
    prompt = f"""You are a friendly pastor writing a brief reflection for content titled "{title}" covering Bible verses.
Write a short, uplifting pastor-style comment.
""" 
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You craft brief pastor reflections on Bible verse content."},
                      {"role": "user",   "content": prompt}],
            temperature=0.7, max_tokens=150
        )
        return resp.choices[0].message['content'].strip()
    except Exception as e:
        logging.error(f"GPT error on pastor comment: {e}")
        return "May God's peace guide you as you meditate on these verses."

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
def main():
    youtube    = get_youtube_service()
    yt_comments= set(load_json_file(REPLIED_COMMENTS_FILE))
    author_hist= load_json_file(AUTHOR_HISTORY_FILE)
    yt_media   = set(load_json_file(REPLIED_VIDEOS_FILE))
    fb_comments= set(load_json_file('replied_comments_facebook.json'))
    fb_media   = set(load_json_file('replied_media_facebook.json'))
    ig_comments= set(load_json_file('replied_comments_instagram.json'))
    ig_media   = set(load_json_file('replied_media_instagram.json'))

    logging.info("Bot started across YouTube, Facebook, Instagram")

    while True:
        # YouTube
        for thread in get_youtube_comments(youtube):
            top    = thread['snippet']['topLevelComment']
            cid    = top['id']
            author = top['snippet']['authorDisplayName']
            chan   = top['snippet'].get('authorChannelId', {}).get('value')
            text   = top['snippet']['textDisplay']
            vid    = thread['snippet']['videoId']

            if cid in yt_comments or chan == YOUTUBE_CHANNEL_ID:
                continue

            title = get_youtube_video_title(youtube, vid)
            count = author_hist.get(author, 0)
            reply = generate_reply_via_gpt("YouTube", author, text, count, title)
            reply_to_youtube_comment(youtube, cid, reply)

            yt_comments.add(cid)
            author_hist[author] = count + 1
            save_json_file(list(yt_comments), REPLIED_COMMENTS_FILE)
            save_json_file(author_hist,      AUTHOR_HISTORY_FILE)

        for vid in get_youtube_videos(youtube):
            if vid in yt_media:
                continue
            title = get_youtube_video_title(youtube, vid)
            comment = generate_pastor_comment(title)
            comment on_youtube_video(youtube, vid, comment)
        