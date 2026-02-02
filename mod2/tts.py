import os
import json
import time
import requests
import shutil
import subprocess
import base64
from pathlib import Path
from dotenv import load_dotenv

from config import load_config
from tts_voice_resolver import resolve_voice

load_dotenv()
config = load_config()

# ------------------- CONFIG -------------------
TTS_BACKEND = os.getenv("TTS_BACKEND", "elevenlabs").strip().lower()  # elevenlabs | qwen
QWEN_TTS_API_BASE = os.getenv("QWEN_TTS_API_BASE", "http://192.168.1.94:9910").rstrip("/")

QWEN_VOICES_PATH = os.getenv("QWEN_VOICES_PATH", "/v1/voices").strip()
QWEN_TTS_PATH_TMPL = os.getenv("QWEN_TTS_PATH_TMPL", "/v1/text-to-speech/{voice_id}").strip()

TTS_FALLBACK_VOICE = os.getenv("TTS_FALLBACK_VOICE", "Bible Reader").strip()
TTS_FORMAT = os.getenv("TTS_FORMAT", "mp3").strip().lower()
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "Auto").strip()
TTS_STABILITY = float(os.getenv("TTS_STABILITY", "0.3"))
TTS_SIMILARITY = float(os.getenv("TTS_SIMILARITY", "0.7"))
TTS_PREPEND_SILENCE_MS = int(os.getenv("TTS_PREPEND_SILENCE_MS", "150"))
TTS_APPEND_SILENCE_MS = int(os.getenv("TTS_APPEND_SILENCE_MS", "200"))
TTS_SOURCE_WAV = os.getenv("TTS_SOURCE_WAV", "1").strip().lower() in ("1", "true", "yes", "y")

QWEN_USE_CLONE = os.getenv("QWEN_USE_CLONE", "1").strip().lower() in ("1", "true", "yes", "y")
QWEN_CLONE_VOICE_PATH = os.getenv("QWEN_CLONE_VOICE_PATH", "").strip()
QWEN_CLONE_FIELD = os.getenv("QWEN_CLONE_FIELD", "reference_audio").strip()
QWEN_CLONE_MIME = os.getenv("QWEN_CLONE_MIME", "audio/mpeg").strip()

DEFAULT_TONE_NAME = os.getenv("DEFAULT_TONE_NAME", "Valentino").strip()

QWEN_XI_API_KEY = os.getenv("QWEN_XI_API_KEY", "").strip()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

CHUNK_SIZE = 1024
AUDIO_DIR = Path(os.getenv("AUDIO_DIR", str(config.paths.audio_dir)))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_TTS = os.getenv("DEBUG_TTS", "1").strip().lower() in ("1", "true", "yes", "y")
TTS_DRY_RUN = os.getenv("TTS_DRY_RUN", "0").strip().lower() in ("1", "true", "yes", "y")


def log(msg: str):
    if DEBUG_TTS:
        print(msg, flush=True)


def _find_ffmpeg() -> str:
    return os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg") or "ffmpeg"


def _convert_wav_to_mp3_with_delay(wav_path: Path, mp3_path: Path, delay_ms: int, append_ms: int) -> None:
    ffmpeg = _find_ffmpeg()
    delay = max(delay_ms, 0)
    pad = max(append_ms, 0) / 1000.0
    filter_arg = f"adelay={delay}|{delay},apad=pad_dur={pad}"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-filter_complex",
        filter_arg,
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(mp3_path),
    ]
    log(f"[TTS] ffmpeg convert wav->mp3 with delay {delay}ms + tail {append_ms}ms")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


VOICE_OPTIONS = {
    "Valentino": {"id": "Nv8Euon5i3G2sBJM47fo", "description": "Deep narrator"},
    "David - American Narrator": {"id": "v9LgF91V36LGgbLX3iHW", "description": "American narrator"},
    "Christopher": {"id": "G17SuINrv2H9FC6nvetn", "description": "British narrator"},
    "Bible Reader": {"id": "bible_reading", "description": "Warm, reverent reading voice"},
    "Science Channel": {"id": "science_channel", "description": "Clear, upbeat educational voice"},
}

VOICE_ALIASES = {k.lower(): k for k in VOICE_OPTIONS.keys()}

DEFAULT_TONE_ALIASES = {
    "", "default", "default voice", "standard", "main", "primary", "narrator"
}

VALENTINO_FORCE_STABILITY = float(os.getenv("VALENTINO_FORCE_STABILITY", "0.95"))
VALENTINO_FORCE_SIMILARITY = float(os.getenv("VALENTINO_FORCE_SIMILARITY", "0.85"))
VALENTINO_FORCE_MODEL_ID = os.getenv(
    "VALENTINO_FORCE_MODEL_ID",
    "style:warm, steady, reverent, calm pacing"
).strip()

# Force these settings for Qwen backend (consistent voice + tone)
QWEN_FORCE_VOICE_ID = "valentino"
QWEN_FORCE_STABILITY = 0.95
QWEN_FORCE_SIMILARITY = 0.85
QWEN_FORCE_MODEL_ID = "style:warm, steady, reverent, calm pacing"


def _canonical_voice_name(requested: str) -> str:
    req = (requested or "").strip()

    if req.lower() in DEFAULT_TONE_ALIASES:
        if DEFAULT_TONE_NAME in VOICE_OPTIONS:
            return DEFAULT_TONE_NAME

    canon = VOICE_ALIASES.get(req.lower())
    if canon:
        return canon

    fb = TTS_FALLBACK_VOICE if TTS_FALLBACK_VOICE in VOICE_OPTIONS else "Bible Reader"
    return fb


def _is_valentino(voice_name: str) -> bool:
    return (voice_name or "").strip().lower() == "valentino"


def _apply_valentino_defaults(voice_name: str, stability: float, similarity_boost: float, model_id):
    if _is_valentino(voice_name):
        return VALENTINO_FORCE_STABILITY, VALENTINO_FORCE_SIMILARITY, (VALENTINO_FORCE_MODEL_ID or model_id)
    return stability, similarity_boost, model_id


def get_voice_id(tone: str):
    voice_name = _canonical_voice_name(tone)
    return VOICE_OPTIONS[voice_name]["id"]


def generate_tts_elevenlabs(
    narration_text: str,
    audio_path: Path,
    voice_id: str,
    stability: float = 0.3,
    similarity_boost: float = 0.7
) -> None:
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
    voices = _fetch_qwen_voices()
    if not voices:
        raise RuntimeError("No voices returned by Qwen /v1/voices")

    voice_name = _canonical_voice_name(tone_name)
    preferred_id = (VOICE_OPTIONS.get(voice_name, {}) or {}).get("id", "")

    if preferred_id:
        for v in voices:
            if (v.get("voice_id") or "") == preferred_id:
                return preferred_id

    for v in voices:
        if (v.get("name") or "") == voice_name:
            return v["voice_id"]

    vn = voice_name.lower()
    for v in voices:
        if vn and vn in (v.get("name") or "").lower():
            return v["voice_id"]

    for v in voices:
        if (v.get("voice_id") or "") == "bible_reading":
            return "bible_reading"

    return voices[0]["voice_id"]


def generate_tts_qwen_compat(
    narration_text: str,
    audio_path: Path,
    tone_name: str,
    stability: float = 0.3,
    similarity_boost: float = 0.7,
    model_id=None,
    clone_extras: dict | None = None,
) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    voice_id = _resolve_qwen_voice_id(tone_name)
    want_mp3 = TTS_FORMAT == "mp3"
    accept = "audio/wav" if (want_mp3 and TTS_SOURCE_WAV) else ("audio/mpeg" if want_mp3 else "audio/wav")
    url = f"{QWEN_TTS_API_BASE}{QWEN_TTS_PATH_TMPL}".format(voice_id=voice_id)

    headers = {"Content-Type": "application/json", "Accept": accept}
    if QWEN_XI_API_KEY:
        headers["xi-api-key"] = QWEN_XI_API_KEY

    payload = {
        "text": narration_text,
        "model_id": model_id,
        "language": (TTS_LANGUAGE or "English"),
        "voice_settings": {"stability": float(stability), "similarity_boost": float(similarity_boost)},
    }
    use_clone = QWEN_USE_CLONE or _is_valentino(tone_name)
    clone_path = None
    if QWEN_CLONE_VOICE_PATH and os.path.isfile(QWEN_CLONE_VOICE_PATH):
        clone_path = QWEN_CLONE_VOICE_PATH
    else:
        local_clone = Path(__file__).parent / "voiceclone.mp3"
        if local_clone.exists():
            clone_path = str(local_clone)
    if clone_extras:
        for k, v in clone_extras.items():
            payload[k] = v
    elif use_clone and clone_path:
        with open(clone_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload[QWEN_CLONE_FIELD or "reference_audio"] = b64
        payload["reference_audio"] = b64
        payload["ref_audio"] = b64
        payload["reference_audio_type"] = QWEN_CLONE_MIME

    if TTS_DRY_RUN:
        log(f"[TTS] DRY_RUN payload voice={tone_name} extras={list((clone_extras or {}).keys())}")
        audio_path.write_bytes(b"")
        return

    r = requests.post(url, json=payload, headers=headers, timeout=600)
    ctype = (r.headers.get("content-type") or "").lower()

    if "application/json" in ctype:
        try:
            j = r.json()
        except Exception:
            raise RuntimeError(f"Qwen TTS returned JSON but could not parse. HTTP {r.status_code}: {r.text[:300]}")
        raise RuntimeError(f"Qwen TTS JSON error. HTTP {r.status_code}: {json.dumps(j)[:600]}")

    if r.status_code != 200:
        raise RuntimeError(f"Qwen TTS HTTP {r.status_code}: {r.text[:600]}")

    if want_mp3 and TTS_SOURCE_WAV:
        wav_path = audio_path.with_suffix(".wav")
        with open(wav_path, "wb") as f:
            f.write(r.content)
        _convert_wav_to_mp3_with_delay(wav_path, audio_path, TTS_PREPEND_SILENCE_MS, TTS_APPEND_SILENCE_MS)
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        with open(audio_path, "wb") as f:
            f.write(r.content)


def generate_tts(narration_text: str, audio_path: Path, tone: str, **kwargs) -> bool:
    final_voice, clone_extras, reason = resolve_voice(
        requested_voice=tone,
        default_voice=DEFAULT_TONE_NAME,
        fallback_voice=TTS_FALLBACK_VOICE,
        clone_path=QWEN_CLONE_VOICE_PATH,
    )
    voice_name = _canonical_voice_name(final_voice)

    stability = float(kwargs.get("stability", TTS_STABILITY))
    similarity = float(kwargs.get("similarity_boost", TTS_SIMILARITY))
    model_id = kwargs.get("model_id", None)

    stability, similarity, model_id = _apply_valentino_defaults(voice_name, stability, similarity, model_id)

    if TTS_BACKEND == "elevenlabs":
        voice_id = VOICE_OPTIONS[voice_name]["id"]
        try:
            generate_tts_elevenlabs(narration_text, audio_path, voice_id, stability, similarity)
            return True
        except Exception as e:
            log(f"[ERR] ElevenLabs failed for voice='{voice_name}': {e}")
            return False

    # Force Qwen compat settings only when using Valentino
    if voice_name.lower() == "valentino":
        stability = QWEN_FORCE_STABILITY
        similarity = QWEN_FORCE_SIMILARITY
        model_id = QWEN_FORCE_MODEL_ID

    try:
        generate_tts_qwen_compat(
            narration_text=narration_text,
            audio_path=audio_path,
            tone_name=voice_name,
            stability=stability,
            similarity_boost=similarity,
            model_id=model_id,
            clone_extras=clone_extras,
        )
        return True
    except Exception as e:
        log(f"[ERR] Qwen compat failed for voice='{voice_name}': {e}")
        return False


def process_tts(script_data: dict, audio_dir: Path = AUDIO_DIR) -> dict:
    tone = _canonical_voice_name(script_data.get("tone", DEFAULT_TONE_NAME))

    sections = script_data.get("sections", [])
    if not sections:
        log("[WARN] No sections in script JSON.")
        return script_data

    ext = "mp3" if TTS_FORMAT == "mp3" else "wav"

    for section_idx, section in enumerate(sections, start=1):
        segments = section.get("segments", [])
        for segment_idx, segment in enumerate(segments, start=1):
            narration = segment.get("narration") or {}
            text = (narration.get("text") or "").strip()
            if not text:
                log(f"[WARN] Section {section_idx} Segment {segment_idx}: empty narration; skipping.")
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

    return script_data
