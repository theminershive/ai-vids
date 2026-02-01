import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ------------------- CONFIG -------------------
TTS_BACKEND = os.getenv("TTS_BACKEND", "elevenlabs").strip().lower()  # elevenlabs | qwen
QWEN_TTS_API_BASE = os.getenv("QWEN_TTS_API_BASE", "http://192.168.1.94:9910").rstrip("/")

# Qwen compat API has fixed paths:
QWEN_VOICES_PATH = os.getenv("QWEN_VOICES_PATH", "/v1/voices").strip()
QWEN_TTS_PATH_TMPL = os.getenv("QWEN_TTS_PATH_TMPL", "/v1/text-to-speech/{voice_id}").strip()

TTS_FALLBACK_VOICE = os.getenv("TTS_FALLBACK_VOICE", "Bible Reader").strip()
TTS_FORMAT = os.getenv("TTS_FORMAT", "mp3").strip().lower()  # mp3|wav
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "Auto").strip()  # kept for backwards compat (not used by compat API)
TTS_STABILITY = float(os.getenv("TTS_STABILITY", "0.3"))
TTS_SIMILARITY = float(os.getenv("TTS_SIMILARITY", "0.7"))

# Optional: if your compat server enforces xi-api-key (it supports the header)
QWEN_XI_API_KEY = os.getenv("QWEN_XI_API_KEY", "").strip()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

CHUNK_SIZE = 1024
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", "audio"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_TTS = os.getenv("DEBUG_TTS", "1").strip().lower() in ("1", "true", "yes", "y")
MIN_AUDIO_BYTES = int(os.getenv("MIN_AUDIO_BYTES", "2048"))  # sanity check

def log(msg: str):
    if DEBUG_TTS:
        print(msg, flush=True)

# ------------------- VOICES -------------------
# For ElevenLabs: id = ElevenLabs voice_id
# For Qwen compat: id is optional; we will resolve via /v1/voices by matching names/ids
VOICE_OPTIONS = {
    "Valentino": {"id": "Nv8Euon5i3G2sBJM47fo", "description": "Deep narrator"},
    "David - American Narrator": {"id": "v9LgF91V36LGgbLX3iHW", "description": "American narrator"},
    "Christopher": {"id": "G17SuINrv2H9FC6nvetn", "description": "British narrator"},

    # These are logical “channel” names in your pipeline:
    "Bible Reader": {"id": "bible_reading", "description": "Warm, reverent reading voice"},
    "Science Channel": {"id": "science_channel", "description": "Clear, upbeat educational voice"},
}

def _safe_voice_name(requested: str) -> str:
    if requested and requested in VOICE_OPTIONS:
        return requested
    fb = TTS_FALLBACK_VOICE if TTS_FALLBACK_VOICE in VOICE_OPTIONS else "Bible Reader"
    return fb

# ------------------- BACKWARDS COMPAT: old helper -------------------
def get_voice_id(tone: str):
    """
    Backwards compatible. Old code expects a voice_id string.
    - ElevenLabs: return actual ElevenLabs ID
    - Qwen compat: return preferred id hint (we still resolve it against /v1/voices)
    """
    voice_name = _safe_voice_name(tone)
    return VOICE_OPTIONS[voice_name]["id"]

# ------------------- ELEVENLABS -------------------
def generate_tts_elevenlabs(narration_text: str, audio_path: Path, voice_id: str,
                            stability: float = 0.3, similarity_boost: float = 0.7) -> None:
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY is missing")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY,
    }
    data = {
        "text": narration_text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": stability, "similarity_boost": similarity_boost},
    }

    r = requests.post(url, json=data, headers=headers, stream=True, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {r.status_code}: {r.text}")

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audio_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# ------------------- QWEN ELEVENLABS-COMPAT -------------------
_qwen_voice_cache = None
_qwen_voice_cache_at = 0
QWEN_VOICE_CACHE_TTL = int(os.getenv("QWEN_VOICE_CACHE_TTL", "300"))

def _fetch_qwen_voices():
    global _qwen_voice_cache, _qwen_voice_cache_at
    now = time.time()
    if _qwen_voice_cache and (now - _qwen_voice_cache_at) < QWEN_VOICE_CACHE_TTL:
        return _qwen_voice_cache

    url = f"{QWEN_TTS_API_BASE}{QWEN_VOICES_PATH}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch Qwen voices {r.status_code}: {r.text[:500]}")
    data = r.json()
    voices = data.get("voices") or []
    # normalize to list of dicts with keys voice_id, name
    norm = []
    for v in voices:
        norm.append({
            "voice_id": v.get("voice_id") or v.get("id"),
            "name": (v.get("name") or "").strip(),
            "description": v.get("description") or "",
        })
    _qwen_voice_cache = norm
    _qwen_voice_cache_at = now
    return norm

def _resolve_qwen_voice_id(tone_name: str) -> str:
    """
    Resolve a requested tone name like "Bible Reader" into a real {voice_id} on the Qwen compat server.
    Matching strategy:
      1) exact match on voice_id using VOICE_OPTIONS[tone]['id']
      2) exact match on name
      3) contains match on name (case-insensitive)
      4) fallback to bible_reading if available, else first voice
    """
    voices = _fetch_qwen_voices()
    if not voices:
        raise RuntimeError("No voices returned by Qwen /v1/voices")

    voice_name = _safe_voice_name(tone_name)
    preferred_id = (VOICE_OPTIONS.get(voice_name, {}) or {}).get("id", "")

    # 1) exact id
    if preferred_id:
        for v in voices:
            if (v.get("voice_id") or "") == preferred_id:
                return preferred_id

    # 2) exact name
    for v in voices:
        if (v.get("name") or "") == voice_name:
            return v["voice_id"]

    # 3) contains match
    vn = voice_name.lower()
    for v in voices:
        if vn and vn in (v.get("name") or "").lower():
            return v["voice_id"]

    # 4) fallback
    for v in voices:
        if (v.get("voice_id") or "") == "bible_reading":
            return "bible_reading"

    return voices[0]["voice_id"]

def generate_tts_qwen_compat(narration_text: str, audio_path: Path, tone_name: str,
                             stability: float = 0.3, similarity_boost: float = 0.7) -> None:
    """
    Calls your Qwen3-TTS ElevenLabs Compat API:
      POST /v1/text-to-speech/{voice_id}
    JSON body is ElevenLabs style: {"text": "...", "model_id": null, "voice_settings": {...}}
    Audio format via Accept header:
      - mp3 -> Accept: audio/mpeg
      - wav -> Accept: audio/wav
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    voice_id = _resolve_qwen_voice_id(tone_name)

    accept = "audio/mpeg" if TTS_FORMAT == "mp3" else "audio/wav"
    url = f"{QWEN_TTS_API_BASE}{QWEN_TTS_PATH_TMPL}".format(voice_id=voice_id)

    headers = {
        "Content-Type": "application/json",
        "Accept": accept,
    }
    if QWEN_XI_API_KEY:
        headers["xi-api-key"] = QWEN_XI_API_KEY

    payload = {
        "text": narration_text,
        "model_id": None,
        "voice_settings": {"stability": float(stability), "similarity_boost": float(similarity_boost)},
    }

    r = requests.post(url, json=payload, headers=headers, timeout=600)

    ctype = (r.headers.get("content-type") or "").lower()

    # If server returns JSON, it's an error detail (FastAPI HTTPException)
    if "application/json" in ctype:
        try:
            j = r.json()
        except Exception:
            raise RuntimeError(f"Qwen TTS returned JSON but could not parse. HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError(f"Qwen TTS JSON error. HTTP {r.status_code}: {json.dumps(j)[:600]}")

    if r.status_code != 200:
        raise RuntimeError(f"Qwen TTS HTTP {r.status_code}: {r.text[:600]}")

    # write bytes
    with open(audio_path, "wb") as f:
        f.write(r.content)

    size = audio_path.stat().st_size if audio_path.exists() else 0
    if size < MIN_AUDIO_BYTES:
        raise RuntimeError(f"Qwen TTS produced too-small audio ({size} bytes) at {audio_path}")

# ------------------- Unified entry -------------------
def generate_tts(narration_text: str, audio_path: Path, tone: str, **kwargs) -> bool:
    voice_name = _safe_voice_name(tone)

    if TTS_BACKEND == "elevenlabs":
        voice_id = VOICE_OPTIONS[voice_name]["id"]
        try:
            generate_tts_elevenlabs(
                narration_text=narration_text,
                audio_path=audio_path,
                voice_id=voice_id,
                stability=float(kwargs.get("stability", TTS_STABILITY)),
                similarity_boost=float(kwargs.get("similarity_boost", TTS_SIMILARITY)),
            )
            return True
        except Exception as e:
            log(f"[ERR] ElevenLabs failed for voice='{voice_name}': {e}")

            fb_name = _safe_voice_name("Bible Reader")
            fb_id = VOICE_OPTIONS[fb_name]["id"]
            try:
                generate_tts_elevenlabs(
                    narration_text=narration_text,
                    audio_path=audio_path,
                    voice_id=fb_id,
                    stability=float(kwargs.get("stability", TTS_STABILITY)),
                    similarity_boost=float(kwargs.get("similarity_boost", TTS_SIMILARITY)),
                )
                log(f"[OK] Fell back to '{fb_name}' (ElevenLabs)")
                return True
            except Exception as e2:
                log(f"[ERR] ElevenLabs fallback failed: {e2}")
                return False

    # Qwen backend (ElevenLabs-compat API)
    try:
        generate_tts_qwen_compat(
            narration_text=narration_text,
            audio_path=audio_path,
            tone_name=voice_name,
            stability=float(kwargs.get("stability", TTS_STABILITY)),
            similarity_boost=float(kwargs.get("similarity_boost", TTS_SIMILARITY)),
        )
        return True
    except Exception as e:
        log(f"[ERR] Qwen compat failed for voice='{voice_name}': {e}")

        fb_name = _safe_voice_name("Bible Reader")
        try:
            generate_tts_qwen_compat(
                narration_text=narration_text,
                audio_path=audio_path,
                tone_name=fb_name,
                stability=float(kwargs.get("stability", TTS_STABILITY)),
                similarity_boost=float(kwargs.get("similarity_boost", TTS_SIMILARITY)),
            )
            log(f"[OK] Fell back to '{fb_name}' (Qwen compat)")
            return True
        except Exception as e2:
            log(f"[ERR] Qwen compat fallback failed: {e2}")
            return False

# ------------------- Backwards compatible processor -------------------
def process_tts(script_data: dict, audio_dir: Path = AUDIO_DIR) -> dict:
    """
    Backwards compatible:
    - uses script_data["tone"] as voice selector (defaults to Bible Reader)
    - writes section_{i}_segment_{j}_{tone}.(wav/mp3)
    - sets segment["narration"]["audio_path"]
    """
    tone = _safe_voice_name(script_data.get("tone", "Bible Reader"))

    sections = script_data.get("sections", [])
    if not sections:
        log("[WARN] No sections in script JSON.")
        return script_data

    ext = "mp3" if TTS_FORMAT == "mp3" else "wav"

    for section_idx, section in enumerate(sections, start=1):
        segments = section.get("segments", [])
        if not segments:
            log(f"[WARN] Section {section_idx} has no segments.")
            continue

        for segment_idx, segment in enumerate(segments, start=1):
            narration = segment.get("narration") or {}
            text = (narration.get("text") or "").strip()
            if not text:
                log(f"[WARN] Section {section_idx} Segment {segment_idx}: empty narration text; skipping.")
                segment.setdefault("narration", {})
                segment["narration"]["audio_path"] = None
                continue

            safe_tone_tag = tone.replace(" ", "_")
            audio_filename = f"section_{section_idx}_segment_{segment_idx}_{safe_tone_tag}.{ext}"
            audio_path = Path(audio_dir) / audio_filename

            log(f"[TTS] Section {section_idx} Segment {segment_idx} voice='{tone}' -> {audio_path}")
            ok = generate_tts(text, audio_path, tone)

            segment.setdefault("narration", {})
            segment["narration"]["audio_path"] = str(audio_path) if ok else None

            if ok:
                size = audio_path.stat().st_size if audio_path.exists() else 0
                log(f"[OK] Wrote {size} bytes: {audio_path}")
            else:
                log(f"[ERR] Failed TTS for section {section_idx} segment {segment_idx}")

    return script_data

def save_audio_paths(updated_script: dict, filename: str = "video_script_with_audio.json") -> Path:
    script_path = AUDIO_DIR.parent / filename
    with open(script_path, "w", encoding="utf-8") as f:
        json.dump(updated_script, f, indent=4)
    return script_path

def load_script_from_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    json_path = input("Enter the path to the JSON file to use: ").strip()
    if not os.path.exists(json_path):
        print(f"The specified JSON file does not exist: {json_path}")
        raise SystemExit(1)

    log(f"[CFG] TTS_BACKEND={TTS_BACKEND}")
    log(f"[CFG] QWEN_TTS_API_BASE={QWEN_TTS_API_BASE}")
    log(f"[CFG] QWEN_VOICES_PATH={QWEN_VOICES_PATH}")
    log(f"[CFG] QWEN_TTS_PATH_TMPL={QWEN_TTS_PATH_TMPL}")
    log(f"[CFG] TTS_FORMAT={TTS_FORMAT} stability={TTS_STABILITY} similarity={TTS_SIMILARITY}")
    log(f"[CFG] fallback={TTS_FALLBACK_VOICE}")

    data = load_script_from_json(json_path)
    updated = process_tts(data, audio_dir=AUDIO_DIR)
    out = save_audio_paths(updated)
    print(f"Saved: {out}")
