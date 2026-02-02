#!/usr/bin/env python3
"""
facebook_comment_reply_llm.py

Facebook Page comment auto-replier powered by OpenAI (Responses API), with:

✅ Replies to different people on the same Reel/post
✅ Caps replies per PERSON per post (default: 2 total per user per post)
   - "we reply" -> they comment again -> "we reply once more" max
✅ Counts manual replies you make yourself (so the bot won’t reply again to that user)
✅ No double-posting to the same user beyond the cap
✅ Optional starter comment ONLY if the post has zero comments (engagement-free)
✅ Uses OpenAI for funny/smart replies without inventing facts

Deps:
  pip install requests

Secrets via env vars (recommended via .env loaded by wrapper):
  FB_PAGE_ID=...
  FB_PAGE_ACCESS_TOKEN=...   (must be a Page token, not user token)
  OPENAI_API_KEY=...
  OPENAI_MODEL=gpt-4.1-mini

Run:
  python facebook_comment_reply_llm.py
  python facebook_comment_reply_llm.py fb_llm_config.json
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

GRAPH_VERSION = "v24.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    # Facebook
    page_id: str
    fb_access_token: str

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4.1-mini"

    # Behavior (posts/comments scanning)
    lookback_hours: int = 72
    max_posts: int = 25
    max_comments_per_post: int = 50

    # pacing / safety
    min_delay_s: float = 2.0
    max_delay_s: float = 7.0
    openai_min_delay_s: float = 0.0
    max_reply_chars: int = 450
    dry_run: bool = False

    # Voice
    tone: str = "funny, smart, warm"
    brand_voice: str = (
        "You are replying as the official Facebook Page owner. Keep it friendly, slightly witty, never mean."
    )

    # Starter comment behavior
    enable_starter_comment: bool = True
    starter_max_chars: int = 220  # keep starter short

    # Per-user reply depth control (per post)
    user_state_store: str = "fb_user_thread_state.json"
    max_replies_per_user_per_post: int = 2  # 1 = only one reply ever; 2 = reply, then one more later
    require_followup_for_second_reply: bool = True  # only allow 2nd reply if user commented after our last reply
    manual_sync_scan_parents_per_post: int = 200  # how many parent comments to scan to count manual replies


def _die(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(code)


def load_config(path: Optional[str] = None) -> Config:
    data: Dict[str, Any] = {}
    if path:
        if not os.path.exists(path):
            _die(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

    page_id = data.get("page_id") or os.getenv("FB_PAGE_ID")
    fb_token = data.get("fb_access_token") or os.getenv("FB_PAGE_ACCESS_TOKEN")
    if not fb_token:
        try:
            from social_tokens import get_page_access_token
            fb_token = get_page_access_token(page_id)
        except Exception:
            fb_token = None

    oai_key = data.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    oai_model = data.get("openai_model") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"

    if not page_id:
        _die("Missing FB Page ID. Set FB_PAGE_ID or provide config JSON with page_id.")
    if not fb_token:
        _die("Missing FB Page access token. Set FB_PAGE_ACCESS_TOKEN or provide config JSON with fb_access_token.")
    if not oai_key:
        _die("Missing OpenAI API key. Set OPENAI_API_KEY or provide config JSON with openai_api_key.")

    default_cfg = Config(page_id=str(page_id), fb_access_token=str(fb_token), openai_api_key=str(oai_key))

    return Config(
        page_id=str(page_id),
        fb_access_token=str(fb_token),
        openai_api_key=str(oai_key),
        openai_model=str(oai_model),

        lookback_hours=int(data.get("lookback_hours", default_cfg.lookback_hours)),
        max_posts=int(data.get("max_posts", default_cfg.max_posts)),
        max_comments_per_post=int(data.get("max_comments_per_post", default_cfg.max_comments_per_post)),

        min_delay_s=float(data.get("min_delay_s", default_cfg.min_delay_s)),
        max_delay_s=float(data.get("max_delay_s", default_cfg.max_delay_s)),
        openai_min_delay_s=float(data.get("openai_min_delay_s", default_cfg.openai_min_delay_s)),
        max_reply_chars=int(data.get("max_reply_chars", default_cfg.max_reply_chars)),
        dry_run=bool(data.get("dry_run", default_cfg.dry_run)),

        tone=str(data.get("tone", default_cfg.tone)),
        brand_voice=str(data.get("brand_voice", default_cfg.brand_voice)),

        enable_starter_comment=bool(data.get("enable_starter_comment", default_cfg.enable_starter_comment)),
        starter_max_chars=int(data.get("starter_max_chars", default_cfg.starter_max_chars)),

        user_state_store=str(data.get("user_state_store", default_cfg.user_state_store)),
        max_replies_per_user_per_post=int(data.get("max_replies_per_user_per_post", default_cfg.max_replies_per_user_per_post)),
        require_followup_for_second_reply=bool(data.get("require_followup_for_second_reply", default_cfg.require_followup_for_second_reply)),
        manual_sync_scan_parents_per_post=int(data.get("manual_sync_scan_parents_per_post", default_cfg.manual_sync_scan_parents_per_post)),
    )


# -----------------------------
# JSON state: per-post per-user reply depth
# -----------------------------

def load_user_state(path: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Structure:
      {
        "POST_ID": {
          "USER_ID": {
            "page_replies": int,
            "last_page_reply_time": "ISO8601"
          }
        }
      }
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
        return {}
    except Exception:
        return {}


def save_user_state(path: str, state: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _get_user_entry(state: Dict[str, Dict[str, Dict[str, Any]]], post_id: str, user_id: str) -> Dict[str, Any]:
    post_map = state.setdefault(post_id, {})
    entry = post_map.setdefault(user_id, {"page_replies": 0, "last_page_reply_time": None})
    if "page_replies" not in entry:
        entry["page_replies"] = 0
    if "last_page_reply_time" not in entry:
        entry["last_page_reply_time"] = None
    return entry


# -----------------------------
# Facebook Graph API helpers
# -----------------------------

def graph_get(path: str, token: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}{path}"
    p = dict(params or {})
    p["access_token"] = token
    r = requests.get(url, params=p, timeout=60)
    try:
        payload = r.json()
    except Exception:
        _die(f"Non-JSON response from Graph API: {r.status_code} {r.text[:200]}")
    if r.status_code >= 400 or "error" in payload:
        _die(f"Graph GET error: {payload}")
    return payload


def graph_post(path: str, token: str, data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{GRAPH_BASE}{path}"
    d = dict(data)
    d["access_token"] = token
    r = requests.post(url, data=d, timeout=60)
    try:
        payload = r.json()
    except Exception:
        _die(f"Non-JSON response from Graph API: {r.status_code} {r.text[:200]}")
    if r.status_code >= 400 or "error" in payload:
        _die(f"Graph POST error: {payload}")
    return payload


def iter_paged(initial: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    page = initial
    while True:
        for item in page.get("data", []) or []:
            yield item
        next_url = (page.get("paging") or {}).get("next")
        if not next_url:
            break
        r = requests.get(next_url, timeout=60)
        page = r.json()
        if "error" in page:
            _die(f"Paging error: {page}")


# -----------------------------
# Fetch posts & comments
# -----------------------------

def get_recent_page_posts(cfg: Config) -> List[Dict[str, Any]]:
    since_dt = datetime.now(timezone.utc) - timedelta(hours=cfg.lookback_hours)
    fields = "id,message,story,created_time,permalink_url"
    first = graph_get(
        f"/{cfg.page_id}/feed",
        cfg.fb_access_token,
        params={
            "fields": fields,
            "limit": min(cfg.max_posts, 100),
            "since": int(since_dt.timestamp()),
        },
    )

    posts: List[Dict[str, Any]] = []
    for post in iter_paged(first):
        posts.append(post)
        if len(posts) >= cfg.max_posts:
            break
    return posts


def get_post_comments(cfg: Config, post_id: str, include_replies: bool = True) -> List[Dict[str, Any]]:
    fields = "id,message,from,created_time,comment_count"
    first = graph_get(
        f"/{post_id}/comments",
        cfg.fb_access_token,
        params={
            "fields": fields,
            "limit": min(cfg.max_comments_per_post, 100),
            "order": "reverse_chronological",
            "filter": "stream",  # top-level comments only
        },
    )

    comments: List[Dict[str, Any]] = []
    for c in iter_paged(first):
        comments.append(c)
        if include_replies and (c.get("comment_count") or 0) > 0:
            parent_id = str(c.get("id", "")).strip()
            if parent_id:
                replies_first = graph_get(
                    f"/{parent_id}/comments",
                    cfg.fb_access_token,
                    params={
                        "fields": fields,
                        "limit": min(cfg.max_comments_per_post, 100),
                        "order": "reverse_chronological",
                    },
                )
                for r in iter_paged(replies_first):
                    comments.append(r)
                    if len(comments) >= cfg.max_comments_per_post:
                        break

        if len(comments) >= cfg.max_comments_per_post:
            break
    return comments


def post_has_any_user_interaction(cfg: Config, post_id: str) -> bool:
    first = graph_get(
        f"/{post_id}/comments",
        cfg.fb_access_token,
        params={
            "fields": "id,from,comment_count",
            "limit": 50,
            "filter": "stream",
            "order": "reverse_chronological",
        },
    )
    for c in iter_paged(first):
        frm = c.get("from") or {}
        frm_id = str(frm.get("id", "")).strip() if isinstance(frm, dict) else ""
        if frm_id and frm_id != str(cfg.page_id):
            return True
        if (c.get("comment_count") or 0) > 0:
            parent_id = str(c.get("id", "")).strip()
            if not parent_id:
                continue
            replies_first = graph_get(
                f"/{parent_id}/comments",
                cfg.fb_access_token,
                params={
                    "fields": "id,from",
                    "limit": 50,
                    "order": "reverse_chronological",
                },
            )
            for r in iter_paged(replies_first):
                r_from = r.get("from") or {}
                r_from_id = str(r_from.get("id", "")).strip() if isinstance(r_from, dict) else ""
                if r_from_id and r_from_id != str(cfg.page_id):
                    return True
    return False


def add_starter_comment_to_post(cfg: Config, post_id: str, message: str) -> str:
    resp = graph_post(f"/{post_id}/comments", cfg.fb_access_token, {"message": message})
    return str(resp.get("id", ""))


def reply_to_comment(cfg: Config, comment_id: str, message: str) -> str:
    resp = graph_post(f"/{comment_id}/comments", cfg.fb_access_token, {"message": message})
    return str(resp.get("id", ""))


# -----------------------------
# Time parsing (for follow-up gating)
# -----------------------------

def parse_fb_time(s: str) -> Optional[datetime]:
    """
    FB created_time usually looks like:
      "2026-01-29T12:34:56+0000" OR "2026-01-29T12:34:56Z"
    Convert to aware datetime.
    """
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    s = s.replace("Z", "+00:00")
    # convert +0000 to +00:00
    if re.match(r".*[+-]\d{4}$", s):
        s = s[:-2] + ":" + s[-2:]
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# -----------------------------
# Manual-reply sync (counts your manual replies)
# -----------------------------

def sync_manual_replies_for_post(cfg: Config, state: Dict[str, Dict[str, Dict[str, Any]]], post_id: str) -> None:
    """
    Scan parent comments on this post and count how many replies the Page has made
    under each parent thread; map that to the parent commenter (user_id).

    This means if YOU manually reply in FB UI, the bot sees it and won’t reply again.
    """
    # Fetch top-level comments (parents)
    parents_first = graph_get(
        f"/{post_id}/comments",
        cfg.fb_access_token,
        params={
            "fields": "id,from,created_time",
            "limit": 50,
            "filter": "stream",
            "order": "reverse_chronological",
        },
    )

    scanned = 0
    for parent in iter_paged(parents_first):
        if scanned >= cfg.manual_sync_scan_parents_per_post:
            break

        parent_id = str(parent.get("id", "")).strip()
        parent_from = parent.get("from") or {}
        user_id = str(parent_from.get("id", "")).strip() if isinstance(parent_from, dict) else ""
        if not parent_id or not user_id:
            continue

        # Fetch replies under that parent
        replies_first = graph_get(
            f"/{parent_id}/comments",
            cfg.fb_access_token,
            params={
                "fields": "id,from,created_time",
                "limit": 50,
                "order": "reverse_chronological",
            },
        )

        page_reply_count = 0
        last_page_time: Optional[datetime] = None

        for r in iter_paged(replies_first):
            frm = r.get("from") or {}
            frm_id = str(frm.get("id", "")).strip() if isinstance(frm, dict) else ""
            if frm_id != str(cfg.page_id):
                continue
            page_reply_count += 1
            ct = parse_fb_time(str(r.get("created_time") or ""))
            if ct and (last_page_time is None or ct > last_page_time):
                last_page_time = ct

        if page_reply_count > 0:
            entry = _get_user_entry(state, post_id, user_id)
            # We only care up to max cap (e.g. 2)
            entry["page_replies"] = max(int(entry.get("page_replies", 0)), min(page_reply_count, cfg.max_replies_per_user_per_post))
            if last_page_time:
                # store latest page reply time (ISO)
                entry["last_page_reply_time"] = last_page_time.isoformat().replace("+00:00", "Z")

        scanned += 1


# -----------------------------
# Prompt assembly + OpenAI call
# -----------------------------

def jitter_sleep(min_s: float, max_s: float) -> None:
    if max_s <= 0:
        return
    delay = random.uniform(min_s, max(min_s, max_s))
    time.sleep(delay)


def derive_post_title_and_body(post: Dict[str, Any]) -> Tuple[str, str]:
    message = (post.get("message") or "").strip()
    story = (post.get("story") or "").strip()

    title = ""
    if message:
        for line in message.splitlines():
            line = line.strip()
            if line:
                title = line
                break
    if not title and story:
        title = story
    if not title:
        title = "Facebook post"

    return title[:140], message[:2500]


def clean_for_prompt(s: str) -> str:
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def clamp_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    m = re.search(r"(.+?[.!?])\s+[^.!?]*$", cut)
    return (m.group(1) if m else cut).strip()


def _openai_post(cfg: Config, payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {cfg.openai_api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=60)
    try:
        out = r.json()
    except Exception:
        _die(f"OpenAI returned non-JSON: {r.status_code} {r.text[:200]}")

    # Responses API includes "error": null on success — only fail if it's truthy
    if r.status_code >= 400 or out.get("error"):
        _die(f"OpenAI error: {out.get('error') or out}")
    return out


def _extract_output_text(out: Dict[str, Any]) -> str:
    text = out.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts: List[str] = []
    for item in out.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                parts.append(c["text"])
    result = "\n".join(p.strip() for p in parts if p.strip()).strip()
    if not result:
        _die(f"OpenAI response had no text output. Raw: {out}")
    return result


def openai_generate_reply(cfg: Config, post_title: str, post_body: str, comment_text: str) -> str:
    system_instructions = f"""
You are writing a reply as the official Facebook Page owner.
Tone: {cfg.tone}. {cfg.brand_voice}

Hard rules:
- No hate, harassment, or bullying.
- Do NOT invent facts. If unsure, keep it general or ask a light question.
- If you include factual claims, keep them widely-known and non-controversial.
- Keep it concise: 1–2 short paragraphs max.
- Do not mention you are an AI or that you used an API.
- Avoid emoji spam; 0–2 emojis max.
- Reply should directly address the commenter and the post context.
""".strip()

    user_content = f"""
POST TITLE:
{post_title}

POST DESCRIPTION/BODY:
{post_body or "(no body text)"}

COMMENT TO REPLY TO:
{comment_text}

Write the best possible reply now.
""".strip()

    payload = {
        "model": cfg.openai_model,
        "input": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content},
        ],
        "max_output_tokens": 220,
        "store": False,
    }

    out = _openai_post(cfg, payload)
    return _extract_output_text(out)


def openai_generate_starter_comment(cfg: Config, post_title: str, post_body: str) -> str:
    system_instructions = f"""
You are writing a SINGLE top-level comment as the official Facebook Page owner on your own post.
Tone: {cfg.tone}. {cfg.brand_voice}

Rules:
- 1–2 sentences max.
- Lightly witty and relevant.
- Ask ONE engaging question OR add ONE interesting non-controversial tidbit.
- No hashtags unless the post itself used them.
- No links.
- Do NOT invent facts. If unsure, keep it general.
- 0–1 emoji max.
""".strip()

    user_content = f"""
POST TITLE:
{post_title}

POST BODY:
{post_body or "(no body text)"}

Write the starter comment now.
""".strip()

    payload = {
        "model": cfg.openai_model,
        "input": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_content},
        ],
        "max_output_tokens": 90,
        "store": False,
    }

    out = _openai_post(cfg, payload)
    return _extract_output_text(out)


# -----------------------------
# Per-user gating logic
# -----------------------------

def should_reply_to_user_on_post(cfg: Config, state: Dict[str, Dict[str, Dict[str, Any]]], post_id: str, user_id: str, comment_created_time: str) -> bool:
    entry = _get_user_entry(state, post_id, user_id)
    page_replies = int(entry.get("page_replies", 0))
    last_page_reply_time = entry.get("last_page_reply_time")

    # Hard cap
    if page_replies >= cfg.max_replies_per_user_per_post:
        return False

    # If this would be the 2nd reply, optionally require that they commented after our last reply
    if cfg.require_followup_for_second_reply and page_replies == 1:
        last_dt = parse_fb_time(str(last_page_reply_time or ""))
        cur_dt = parse_fb_time(str(comment_created_time or ""))
        if last_dt and cur_dt:
            # Only reply if their comment is newer than our last reply time
            return cur_dt > last_dt
        # If timestamps can’t be parsed, be conservative: skip second reply
        return False

    return True


def record_page_reply(state: Dict[str, Dict[str, Dict[str, Any]]], post_id: str, user_id: str, now_iso: str) -> None:
    entry = _get_user_entry(state, post_id, user_id)
    entry["page_replies"] = int(entry.get("page_replies", 0)) + 1
    entry["last_page_reply_time"] = now_iso


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = load_config(config_path)

    user_state = load_user_state(cfg.user_state_store)
    print(f"[INFO] Loaded user-state from {cfg.user_state_store} (posts tracked={len(user_state)})")

    posts = get_recent_page_posts(cfg)
    print(f"[INFO] Found {len(posts)} recent posts (lookback={cfg.lookback_hours}h)")

    replied_now = 0
    skipped_depth = 0
    starter_posted = 0
    starter_skipped_has_comments = 0

    for post in posts:
        post_id = str(post.get("id", "")).strip()
        if not post_id:
            continue

        title, body = derive_post_title_and_body(post)
        permalink = post.get("permalink_url", "")
        print(f"\n[POST] {post_id}  {permalink}")

        # Sync manual replies so your own UI replies count toward the cap
        try:
            sync_manual_replies_for_post(cfg, user_state, post_id)
            save_user_state(cfg.user_state_store, user_state)
        except SystemExit:
            raise
        except Exception as e:
            print(f"  [WARN] Manual-reply sync failed (continuing): {e}")

        # Starter comment rule: ONLY if there is no user interaction yet
        if cfg.enable_starter_comment:
            try:
                if post_has_any_user_interaction(cfg, post_id):
                    starter_skipped_has_comments += 1
                    print("  [INFO] Starter skipped: post already has user interaction.")
                else:
                    if cfg.openai_min_delay_s > 0:
                        time.sleep(cfg.openai_min_delay_s)

                    starter = openai_generate_starter_comment(cfg, clean_for_prompt(title), clean_for_prompt(body))
                    starter = clamp_text(starter, cfg.starter_max_chars)

                    if cfg.dry_run:
                        print("  [DRY RUN] Would add starter comment:\n" + starter)
                    else:
                        starter_id = add_starter_comment_to_post(cfg, post_id, starter)
                        print(f"  [OK] Starter comment posted. id={starter_id}")
                    starter_posted += 1

                    jitter_sleep(max(1.0, cfg.min_delay_s), max(2.0, cfg.min_delay_s + 2.0))
            except SystemExit:
                raise
            except Exception as e:
                print(f"  [WARN] Starter-comment attempt failed: {e}")

        comments = get_post_comments(cfg, post_id, include_replies=True)
        print(f"[INFO]  {len(comments)} comments fetched")

        for c in comments:
            cid = str(c.get("id", "")).strip()
            if not cid:
                continue

            frm = c.get("from") or {}
            user_id = str(frm.get("id", "")).strip() if isinstance(frm, dict) else ""
            if not user_id:
                continue

            # Skip if the comment is from the Page itself
            if user_id == str(cfg.page_id):
                continue

            comment_text = clean_for_prompt(str(c.get("message") or "").strip())
            if not comment_text:
                continue

            comment_created = str(c.get("created_time") or "").strip()

            # Per-user-per-post depth gating
            if not should_reply_to_user_on_post(cfg, user_state, post_id, user_id, comment_created):
                skipped_depth += 1
                continue

            author = (c.get("from") or {}).get("name") if isinstance(c.get("from"), dict) else ""
            preview = comment_text[:140].replace("\n", " ")
            print(f"  [NEW] {cid} by {author!r}: {preview!r}")

            # Generate reply
            if cfg.openai_min_delay_s > 0:
                time.sleep(cfg.openai_min_delay_s)

            reply = openai_generate_reply(cfg, clean_for_prompt(title), clean_for_prompt(body), comment_text)
            reply = clamp_text(reply, cfg.max_reply_chars)

            if cfg.dry_run:
                print("    [DRY RUN] Reply would be:\n" + reply)
            else:
                new_id = reply_to_comment(cfg, cid, reply)
                print(f"    [OK] Replied. reply_id={new_id}")

            # Record we replied to THIS USER on THIS POST
            record_page_reply(user_state, post_id, user_id, iso_utc_now())
            save_user_state(cfg.user_state_store, user_state)

            replied_now += 1
            jitter_sleep(cfg.min_delay_s, cfg.max_delay_s)

    print(
        f"\n[DONE] replied_now={replied_now} skipped_depth={skipped_depth} "
        f"starter_posted={starter_posted} starter_skipped_has_comments={starter_skipped_has_comments} "
        f"user_state_posts={len(user_state)}"
    )


if __name__ == "__main__":
    main()
