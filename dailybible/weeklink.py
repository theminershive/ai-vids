import os
import datetime
import json
import openai
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from dotenv import load_dotenv
load_dotenv()

WEEKLY_VIDEO_URL = "https://youtu.be/VH51IAtbfj0"

# === CONFIGURATION ===
COMMENT_LOG_FILE = "commented_bible_videos.json"

def extract_video_id(url):
    return url.split("watch?v=")[-1].split("/")[-1].split("?")[0]

WEEKLY_VIDEO_ID = extract_video_id(WEEKLY_VIDEO_URL)

WEEKLY_VIDEO_URL = "https://youtu.be/VH51IAtbfj0"
CHANNEL_ID = "UCmM96VHTjKYPcIKCT4LvR4w"
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
openai.api_key = os.getenv("OPENAI_API_KEY")

def authenticate_youtube():
    if not os.path.exists("token2.json"):
        raise RuntimeError("Missing token2.json for YouTube API access.")
    creds = Credentials.from_authorized_user_file("token2.json", SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds.valid:
        raise RuntimeError("token2.json is invalid or expired.")
    return build("youtube", "v3", credentials=creds)

def detect_tone(title):
    lowered = title.lower()
    if any(kw in lowered for kw in ["miracle", "blessing", "faith", "hope", "grace"]):
        return "inspirational and reverent"
    elif any(kw in lowered for kw in ["warning", "prophecy", "judgment", "end times"]):
        return "serious and reflective"
    elif any(kw in lowered for kw in ["teaching", "parable", "lesson", "wisdom"]):
        return "thoughtful and informative"
    else:
        return "warm and devotional"

def get_recent_shorts(youtube, channel_id, days=7):
    now = datetime.datetime.utcnow()
    published_after = (now - datetime.timedelta(days=days)).isoformat("T") + "Z"

    shorts = []
    next_page_token = None

    while True:
        request = youtube.search().list(
            part="id",
            channelId=channel_id,
            publishedAfter=published_after,
            maxResults=50,
            type="video",
            videoDuration="short",
            order="date",
            pageToken=next_page_token,
        )
        response = request.execute()
        for item in response["items"]:
            shorts.append(item["id"]["videoId"])
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    return shorts

def update_description(youtube, video_id, weekly_url):
    video = youtube.videos().list(part="snippet", id=video_id).execute()
    if not video["items"]:
        print(f"Video {video_id} not found.")
        return None
    snippet = video["items"][0]["snippet"]
    old_description = snippet.get("description", "")
    title = snippet.get("title", "")
    
    if WEEKLY_VIDEO_URL in old_description:
        print(f"[SKIP] Already contains link: {video_id}")
    else:
        new_description = old_description + f"\n\nWatch the full recap: {weekly_url}"
        snippet["description"] = new_description
        youtube.videos().update(
            part="snippet",
            body={"id": video_id, "snippet": snippet}
        ).execute()
        print(f"[UPDATED] {video_id} with weekly link")
    return title

def generate_comment(title, weekly_url):
    tone = detect_tone(title)
    prompt = (
        f"You are a Bible teacher and the creator of a weekly video recap series on scripture. "
        f"This Short is a clip from a longer reflection. Write a short, {tone} YouTube comment (under 40 words) "
        f"inviting viewers to watch the full weekly Bible recap. Include 1â€“2 gentle emojis like ðŸ™ or ðŸ“–. "
        f"The Short is titled: '{title}'. End the comment with this link: {weekly_url}."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a kind, devotional Bible teacher who writes engaging YouTube comments with gentle emojis."},
                      {"role": "user", "content": prompt}],
            max_tokens=100
        )
        comment = response.choices[0].message.content.strip()
        if weekly_url not in comment:
            comment = f"{comment} ðŸ‘‰ {weekly_url}"
        return comment[:1000]
    except Exception as e:
        print(f"[ERROR] Failed to generate comment: {e}")
        return f"This verse is part of our weekly Bible recap ðŸ™ Watch the full message ðŸ‘‰ {weekly_url}"

def remove_outdated_comment(youtube, video_id, outdated_text):
    try:
        comments = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50,
            textFormat="plainText"
        ).execute()
        for item in comments.get("items", []):
            body = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            if outdated_text in body:
                comment_id = item["id"]
                youtube.commentThreads().delete(id=comment_id).execute()
                print(f"[DELETED] Outdated comment on {video_id}")
                return True
    except Exception as e:
        print(f"[ERROR] Failed to delete outdated comment: {e}")
    return False

def clean_up_commented_log(youtube):
    # Load the log of commented videos
    if os.path.exists(COMMENT_LOG_FILE):
        with open(COMMENT_LOG_FILE, "r") as f:
            commented = json.load(f)
    else:
        commented = {}

    # Go through the log and check if the comments exist on YouTube
    to_remove = []
    for video_id in commented:
        try:
            comments = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=50,
                textFormat="plainText"
            ).execute()

            # If no comments exist for the video, or no comment with the correct link, mark for removal
            if not any(WEEKLY_VIDEO_URL in item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                       for item in comments.get("items", [])):
                to_remove.append(video_id)
        except Exception as e:
            print(f"[ERROR] Failed to check comments for {video_id}: {e}")

    # Remove the missing comments from the log
    for video_id in to_remove:
        del commented[video_id]
        print(f"[CLEANED UP] Removed {video_id} from commented log (no comment found)")

    # Save the updated log
    with open(COMMENT_LOG_FILE, "w") as f:
        json.dump(commented, f, indent=2)


def post_comment(youtube, video_id, text):
    # First clean up the log to remove entries for videos without comments
    clean_up_commented_log(youtube)

    # Load the updated log
    if os.path.exists(COMMENT_LOG_FILE):
        with open(COMMENT_LOG_FILE, "r") as f:
            commented = json.load(f)
    else:
        commented = {}

    # If the video is in the log, skip it
    if video_id in commented:
        print(f"[SKIP] Already commented (log): {video_id}")
        return

    try:
        # Post the new comment with the updated link
        youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {
                            "textOriginal": text
                        }
                    }
                }
            }
        ).execute()
        print(f"[COMMENTED] on {video_id}")

        # Save the video ID to the log
        commented[video_id] = WEEKLY_VIDEO_URL
        with open(COMMENT_LOG_FILE, "w") as f:
            json.dump(commented, f, indent=2)

    except Exception as e:
        print(f"[ERROR] Could not comment on {video_id}: {e}")

def main():
    youtube = authenticate_youtube()
    shorts = get_recent_shorts(youtube, CHANNEL_ID)
    print(f"Found {len(shorts)} Shorts uploaded in the last 7 days.")
    for vid in shorts:
        if vid == WEEKLY_VIDEO_ID:
            print(f"[SKIP] Skipping weekly video itself: {vid}")
            continue
        title = update_description(youtube, vid, WEEKLY_VIDEO_URL)
        if title:
            comment = generate_comment(title, WEEKLY_VIDEO_URL)
            post_comment(youtube, vid, comment)

if __name__ == "__main__":
    main()
