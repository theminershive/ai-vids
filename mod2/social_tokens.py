#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

GRAPH_API_BASE = os.getenv("GRAPH_API_BASE", "https://graph.facebook.com")
GRAPH_API_VERSION = os.getenv("GRAPH_API_VERSION", "v21.0")
TOKEN_CACHE_FILE = os.getenv("FB_TOKEN_CACHE_FILE", "fb_token.json")

APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")
SHORT_LIVED_TOKEN = os.getenv("SHORT_LIVED_TOKEN")
USER_ACCESS_TOKEN = os.getenv("USER_ACCESS_TOKEN")


def _cache_path() -> Path:
    return Path(TOKEN_CACHE_FILE)


def _load_cache() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    tmp.replace(path)


def _is_valid(cache: dict, key: str) -> bool:
    token = cache.get(key)
    if not token:
        return False
    exp = cache.get(f"{key}_expires_at")
    if not exp:
        return True
    return time.time() < float(exp)


def _exchange_short_lived_token(app_id: str, app_secret: str, short_token: str) -> Tuple[str, int]:
    url = f"{GRAPH_API_BASE}/oauth/access_token"
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": short_token,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {resp.status_code} {resp.text}")
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Token exchange returned no access_token")
    expires_in = int(data.get("expires_in", 0))
    return token, expires_in


def get_long_lived_user_token() -> Optional[str]:
    cache = _load_cache()

    if _is_valid(cache, "long_lived_user_token"):
        return cache.get("long_lived_user_token")

    if APP_ID and APP_SECRET and SHORT_LIVED_TOKEN:
        try:
            token, expires_in = _exchange_short_lived_token(APP_ID, APP_SECRET, SHORT_LIVED_TOKEN)
            cache["long_lived_user_token"] = token
            if expires_in:
                cache["long_lived_user_token_expires_at"] = time.time() + expires_in
            _save_cache(cache)
            return token
        except Exception as exc:
            logging.warning("Short-lived token exchange failed: %s", exc)

    if USER_ACCESS_TOKEN:
        cache["long_lived_user_token"] = USER_ACCESS_TOKEN
        _save_cache(cache)
        return USER_ACCESS_TOKEN

    return None


def get_page_access_token(page_id: str) -> Optional[str]:
    if not page_id:
        return None

    cache = _load_cache()
    cached_page_id = cache.get("page_id")
    if cached_page_id == page_id and _is_valid(cache, "page_access_token"):
        token = cache.get("page_access_token")
        if token:
            os.environ["FB_PAGE_ACCESS_TOKEN"] = token
        return token

    user_token = get_long_lived_user_token()
    if not user_token:
        return None

    url = f"{GRAPH_API_BASE}/{GRAPH_API_VERSION}/me/accounts"
    resp = requests.get(url, params={"access_token": user_token}, timeout=30)
    if resp.status_code != 200:
        logging.warning("Failed to fetch page tokens: %s", resp.text)
        return None

    for page in resp.json().get("data", []):
        if str(page.get("id")) == str(page_id):
            token = page.get("access_token")
            if token:
                cache["page_id"] = str(page_id)
                cache["page_access_token"] = token
                _save_cache(cache)
                os.environ["FB_PAGE_ACCESS_TOKEN"] = token
                return token

    logging.warning("Page ID not found in /me/accounts")
    return None


def get_instagram_token() -> Optional[str]:
    return get_long_lived_user_token()
