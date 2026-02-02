#!/usr/bin/env python3
import os
import io
import gc
import uuid
import time
import asyncio
import tempfile
import subprocess
import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

import torch
import soundfile as sf

from faster_whisper import WhisperModel
from qwen_tts import Qwen3TTSModel

# -----------------------------------------------------------------------------
# ENV
# -----------------------------------------------------------------------------
load_dotenv()

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_VISIBLE_DEVICES", "0"))
os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "1"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "1"))

APP_HOST = os.getenv("BROKER_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("BROKER_PORT", "9910"))

MAX_QUEUE_SECONDS = int(os.getenv("MAX_QUEUE_SECONDS", "1800"))  # 30 minutes
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "3600"))
MODEL_IDLE_TTL = int(os.getenv("MODEL_IDLE_TTL", "45"))  # seconds to keep last model warm

REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "0").lower() in ("1", "true", "yes")
API_KEY = os.getenv("API_KEY", "")

MODELS_DIR = Path(os.getenv("QWEN_TTS_MODELS_DIR", "./models"))
BASEVOICE_DIR = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-Base"
CUSTOMVOICE_DIR = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOICEDESIGN_DIR = MODELS_DIR / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_OUTPUT_FORMAT = os.getenv("QWEN_TTS_OUTPUT_FORMAT", "mp3").lower()
DEBUG_TTS = os.getenv("DEBUG_TTS", "0").lower() in ("1", "true", "yes", "y")

# Optional server-side clone defaults
CLONE_VOICE = os.getenv("CLONE_VOICE", "").strip()
CLONE_AUDIO = os.getenv("CLONE_AUDIO", "").strip()
CLONE_TEXT = os.getenv("CLONE_TEXT", "").strip()

# Add a small lead-in silence to avoid MP3 encoder/decoder priming clipping
TTS_LEADIN_MS = int(os.getenv("TTS_LEADIN_MS", "80"))
TTS_TAIL_MS = int(os.getenv("TTS_TAIL_MS", "200"))

# Whisper settings
STT_MODEL = os.getenv("STT_MODEL", "large-v3").strip()
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "float16").strip()
STT_DEVICE = os.getenv("STT_DEVICE", "cuda").strip()
STT_DEVICE_INDEX = int(os.getenv("STT_DEVICE_INDEX", "0"))
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "auto").strip()
STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "5"))
STT_VAD_FILTER = os.getenv("STT_VAD_FILTER", "true").strip().lower() in ("1", "true", "yes", "y")
STT_WORD_TIMESTAMPS = os.getenv("STT_WORD_TIMESTAMPS", "false").strip().lower() in ("1", "true", "yes", "y")

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class VoiceSettings(BaseModel):
    stability: float = Field(default=0.3, ge=0.0, le=1.0)
    similarity_boost: float = Field(default=0.7, ge=0.0, le=1.0)

class SamplingSettings(BaseModel):
    seed: Optional[int] = Field(default=None, ge=0, le=2_147_483_647)
    do_sample: Optional[bool] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0, le=500)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.5, le=2.0)

class TTSRequest(BaseModel):
    text: str
    model_id: Optional[str] = None
    language: Optional[str] = None
    voice_settings: Optional[VoiceSettings] = None
    sampling: Optional[SamplingSettings] = None

    # NEW (backwards compatible): base64-encoded reference audio for clone
    # Accepts raw base64 or data URLs like "data:audio/wav;base64,...."
    reference_audio: Optional[str] = None
    reference_audio_type: Optional[str] = None
    reference_audio_b64: Optional[str] = None
    reference_text: Optional[str] = None

class TranscribePathRequest(BaseModel):
    audio_path: str
    language: Optional[str] = None
    task: Optional[str] = "transcribe"
    response_format: Optional[str] = "verbose_json"
    prompt: Optional[str] = None

# -----------------------------------------------------------------------------
# Voice registry
# -----------------------------------------------------------------------------
@dataclass
class VoiceProfile:
    id: str
    name: str
    description: str
    mode: Literal["customvoice", "voicedesign"]
    language: str
    speaker: Optional[str] = None
    instruct: Optional[str] = None

VOICE_PROFILES: List[VoiceProfile] = [
    VoiceProfile(
        id="valentino",
        name="Valentino (Premium Narrator)",
        description="Deep, premium narrator tone.",
        mode="voicedesign",
        language="English",
        instruct="Male, 35-45, deep warm baritone, premium narrator tone, dry studio sound, medium pace."
    ),
    VoiceProfile(
        id="ryan_neutral",
        name="Ryan (CustomVoice)",
        description="Preset speaker Ryan.",
        mode="customvoice",
        language="English",
        speaker="Ryan",
        instruct="Neutral American accent. Calm, confident. Medium pace. Dry studio."
    ),
    VoiceProfile(
        id="adien",
        name="Adien (Built-in)",
        description="Built-in neutral narrator.",
        mode="voicedesign",
        language="English",
        speaker="adien",
        instruct="Neutral, clear, steady pace. Dry studio."
    ),
    VoiceProfile(
        id="sohee",
        name="Sohee (Built-in)",
        description="Built-in female voice.",
        mode="voicedesign",
        language="English",
        speaker="sohee",
        instruct="Warm, calm, clear. Dry studio."
    ),
    VoiceProfile(
        id="vivian",
        name="Vivian (Built-in)",
        description="Built-in female voice.",
        mode="voicedesign",
        language="English",
        speaker="vivian",
        instruct="Bright, friendly, clear. Dry studio."
    ),
]
VOICE_MAP = {v.id: v for v in VOICE_PROFILES}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found. Install: sudo apt install -y ffmpeg (or set QWEN_TTS_OUTPUT_FORMAT=wav)"
        )

def _encode_wav_bytes(wav, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()

def _wav_bytes_to_mp3_bytes(wav_bytes: bytes) -> bytes:
    _ensure_ffmpeg()
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
        input=wav_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if p.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg mp3 encode failed: {p.stderr.decode('utf-8','ignore')}")
    return p.stdout

def _prepend_silence(wav, sr: int, ms: int):
    if not ms or ms <= 0:
        return wav
    n = int(sr * (ms / 1000.0))
    if n <= 0:
        return wav

    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu()
        silence = torch.zeros((n,), dtype=wav.dtype)
        return torch.cat([silence, wav], dim=0)

    try:
        import numpy as np
        wav_np = np.asarray(wav)
        silence = np.zeros((n,), dtype=wav_np.dtype)
        return np.concatenate([silence, wav_np], axis=0)
    except Exception:
        return wav

def _append_silence(wav, sr: int, ms: int):
    if not ms or ms <= 0:
        return wav
    n = int(sr * (ms / 1000.0))
    if n <= 0:
        return wav

    if isinstance(wav, torch.Tensor):
        wav = wav.detach().cpu()
        silence = torch.zeros((n,), dtype=wav.dtype)
        return torch.cat([wav, silence], dim=0)

    try:
        import numpy as np
        wav_np = np.asarray(wav)
        silence = np.zeros((n,), dtype=wav_np.dtype)
        return np.concatenate([wav_np, silence], axis=0)
    except Exception:
        return wav

def _settings_to_sampling(voice_settings: Optional[VoiceSettings]) -> Dict[str, Any]:
    if not voice_settings:
        return {"do_sample": False}
    stability = float(voice_settings.stability)
    sim = float(voice_settings.similarity_boost)
    temperature = 1.0 - (0.8 * stability)
    top_p = 0.95 - (0.20 * sim)
    if stability >= 0.85:
        return {"do_sample": False}
    return {
        "do_sample": True,
        "temperature": round(max(0.2, min(1.0, temperature)), 2),
        "top_p": round(max(0.6, min(0.98, top_p)), 2),
    }

def _apply_sampling_overrides(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(base or {})
    if not overrides:
        return out

    for k in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty"):
        if k in overrides and overrides[k] is not None:
            out[k] = overrides[k]

    if out.get("do_sample") is False:
        out.pop("temperature", None)
        out.pop("top_p", None)
        out.pop("top_k", None)
        out.pop("repetition_penalty", None)

    return out

def _set_deterministic_seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    try:
        s = int(seed)
    except Exception:
        return None

    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s

def _segments_to_verbose_json(segments, info) -> Dict[str, Any]:
    out_segments = []
    full_text_parts = []
    for i, seg in enumerate(segments):
        txt = (seg.text or "").strip()
        full_text_parts.append(txt)
        item = {
            "id": i,
            "seek": 0,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": txt,
            "tokens": [],
            "temperature": None,
            "avg_logprob": None,
            "compression_ratio": None,
            "no_speech_prob": None,
        }
        if hasattr(seg, "words") and seg.words:
            item["words"] = [{"start": float(w.start), "end": float(w.end), "word": w.word} for w in seg.words]
        out_segments.append(item)

    return {
        "task": "transcribe",
        "language": getattr(info, "language", None),
        "duration": getattr(info, "duration", None),
        "text": " ".join([t for t in full_text_parts if t]),
        "segments": out_segments,
    }

def _gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _decode_reference_audio_to_temp_wav(reference_audio: str) -> Tuple[str, int]:
    """
    Accepts base64 audio. If it's a data URL, strips prefix.
    Writes bytes to a temp file (keeps original container), then converts to WAV with ffmpeg if needed.
    Returns (wav_path, bytes_len).
    """
    if not reference_audio:
        raise HTTPException(status_code=400, detail="reference_audio is empty")

    b64 = reference_audio.strip()
    if b64.startswith("data:") and "base64," in b64:
        b64 = b64.split("base64,", 1)[1].strip()

    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"reference_audio is not valid base64: {e}")

    # write raw bytes (unknown container)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        raw_path = tmp.name
        tmp.write(raw)

    # convert to wav so downstream is consistent
    _ensure_ffmpeg()
    wav_path = raw_path + ".wav"
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", raw_path, "-ar", "24000", "-ac", "1", wav_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        os.remove(raw_path)
    except Exception:
        pass

    if p.returncode != 0 or not os.path.exists(wav_path):
        raise HTTPException(status_code=400, detail=f"reference_audio decode/convert failed: {p.stderr.decode('utf-8','ignore')}")

    return wav_path, len(raw)


def _load_clone_audio_b64() -> str:
    if not CLONE_AUDIO:
        return ""
    p = Path(CLONE_AUDIO).expanduser()
    if not p.exists():
        return ""
    try:
        return base64.b64encode(p.read_bytes()).decode("utf-8")
    except Exception:
        return ""

# -----------------------------------------------------------------------------
# Model Swap Manager (the 8GB fix)
# -----------------------------------------------------------------------------
class ModelManager:
    def __init__(self):
        self.active: Optional[str] = None  # "stt"|"tts_custom"|"tts_design"|"tts_base"
        self.last_used_ts: float = 0.0

        self.whisper: Optional[WhisperModel] = None
        self.qwen_base: Optional[Qwen3TTSModel] = None
        self.qwen_custom: Optional[Qwen3TTSModel] = None
        self.qwen_design: Optional[Qwen3TTSModel] = None

    def _unload_all(self):
        self.whisper = None
        self.qwen_base = None
        self.qwen_custom = None
        self.qwen_design = None
        self.active = None
        _gpu_cleanup()

    def maybe_idle_unload(self):
        if self.active and (time.time() - self.last_used_ts) > MODEL_IDLE_TTL:
            self._unload_all()

    def get_whisper(self) -> WhisperModel:
        if self.active not in (None, "stt"):
            self._unload_all()
        if self.whisper is None:
            self.whisper = WhisperModel(
                STT_MODEL,
                device=STT_DEVICE,
                device_index=STT_DEVICE_INDEX,
                compute_type=STT_COMPUTE_TYPE,
            )
        self.active = "stt"
        self.last_used_ts = time.time()
        return self.whisper

    def get_qwen_custom(self) -> Qwen3TTSModel:
        if self.active not in (None, "tts_custom"):
            self._unload_all()
        if self.qwen_custom is None:
            if not CUSTOMVOICE_DIR.exists():
                raise RuntimeError(f"Missing model dir: {CUSTOMVOICE_DIR}")
            self.qwen_custom = Qwen3TTSModel.from_pretrained(
                str(CUSTOMVOICE_DIR),
                device_map={"": 0},
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        self.active = "tts_custom"
        self.last_used_ts = time.time()
        return self.qwen_custom

    def get_qwen_base(self) -> Qwen3TTSModel:
        if self.active not in (None, "tts_base"):
            self._unload_all()
        if self.qwen_base is None:
            if not BASEVOICE_DIR.exists():
                raise RuntimeError(f"Missing model dir: {BASEVOICE_DIR}")
            self.qwen_base = Qwen3TTSModel.from_pretrained(
                str(BASEVOICE_DIR),
                device_map={"": 0},
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        self.active = "tts_base"
        self.last_used_ts = time.time()
        return self.qwen_base

    def get_qwen_design(self) -> Qwen3TTSModel:
        if self.active not in (None, "tts_design"):
            self._unload_all()
        if self.qwen_design is None:
            if not VOICEDESIGN_DIR.exists():
                raise RuntimeError(f"Missing model dir: {VOICEDESIGN_DIR}")
            self.qwen_design = Qwen3TTSModel.from_pretrained(
                str(VOICEDESIGN_DIR),
                device_map={"": 0},
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        self.active = "tts_design"
        self.last_used_ts = time.time()
        return self.qwen_design

mm = ModelManager()

# -----------------------------------------------------------------------------
# Queue
# -----------------------------------------------------------------------------
JobType = Literal["tts", "stt"]

@dataclass
class Job:
    id: str
    kind: JobType
    est_seconds: int
    payload: Dict[str, Any]
    future: asyncio.Future

class GPUQueue:
    def __init__(self):
        self.q: asyncio.Queue[Job] = asyncio.Queue()
        self.queued_seconds: int = 0
        self.active_job: Optional[Job] = None
        self.lock = asyncio.Lock()

    async def try_enqueue(self, job: Job) -> None:
        async with self.lock:
            if self.queued_seconds + job.est_seconds > MAX_QUEUE_SECONDS:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "queue_full",
                        "max_queue_seconds": MAX_QUEUE_SECONDS,
                        "queued_seconds": self.queued_seconds,
                        "requested_est_seconds": job.est_seconds,
                    },
                )
            self.queued_seconds += job.est_seconds
            await self.q.put(job)

    async def worker_loop(self):
        while True:
            job = await self.q.get()
            async with self.lock:
                self.active_job = job

            try:
                mm.maybe_idle_unload()
                result = await asyncio.to_thread(run_job_blocking, job)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as e:
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                async with self.lock:
                    self.queued_seconds = max(0, self.queued_seconds - job.est_seconds)
                    self.active_job = None
                self.q.task_done()

gpuq = GPUQueue()

# -----------------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------------
def estimate_tts_seconds(text: str) -> int:
    n = len((text or "").strip())
    return int(12 + (n / 100.0) * 14)

def estimate_stt_seconds(file_bytes: int) -> int:
    mb = max(0.1, file_bytes / (1024 * 1024))
    return int(12 + mb * 22)

# -----------------------------------------------------------------------------
# Blocking work
# -----------------------------------------------------------------------------
def run_job_blocking(job: Job) -> Dict[str, Any]:
    if job.kind == "tts":
        return run_tts_blocking(job.payload)
    return run_stt_blocking(job.payload)

def _call_qwen_base_clone(
    model: Qwen3TTSModel,
    *,
    text: str,
    language: str,
    instruct: str,
    ref_wav_path: str,
    ref_text: Optional[str],
    sampling: Dict[str, Any],
):
    """
    Only base model supports generate_voice_clone; do NOT attempt clone on other models.
    """
    fn = getattr(model, "generate_voice_clone", None)
    if not callable(fn):
        raise HTTPException(
            status_code=503,
            detail="Base TTS model does not expose generate_voice_clone; clone is unavailable.",
        )

    if not ref_text:
        # Try x_vector_only_mode if supported; otherwise require ref_text
        try:
            return fn(
                text=text,
                language=language,
                instruct=instruct,
                ref_audio=ref_wav_path,
                x_vector_only_mode=True,
                **sampling,
            )
        except TypeError:
            try:
                return fn(
                    text=text,
                    language=language,
                    instruct=instruct,
                    ref_audio_path=ref_wav_path,
                    x_vector_only_mode=True,
                    **sampling,
                )
            except TypeError:
                try:
                    return fn(
                        text=text,
                        language=language,
                        instruct=instruct,
                        reference_audio_path=ref_wav_path,
                        x_vector_only_mode=True,
                        **sampling,
                    )
                except TypeError:
                    raise HTTPException(
                        status_code=400,
                        detail="reference_text required for cloning with this model.",
                    )

    try:
        return fn(
            text=text,
            language=language,
            instruct=instruct,
            ref_audio=ref_wav_path,
            ref_text=ref_text,
            **sampling,
        )
    except TypeError:
        try:
            return fn(
                text=text,
                language=language,
                instruct=instruct,
                ref_audio_path=ref_wav_path,
                ref_text=ref_text,
                **sampling,
            )
        except TypeError:
            return fn(
                text=text,
                language=language,
                instruct=instruct,
                reference_audio_path=ref_wav_path,
                reference_text=ref_text,
                **sampling,
            )

def run_tts_blocking(payload: Dict[str, Any]) -> Dict[str, Any]:
    voice_id = payload["voice_id"]
    text = payload["text"]
    accept = payload.get("accept") or ""
    model_id = payload.get("model_id")

    req_reference_audio = payload.get("reference_audio") or payload.get("reference_audio_b64")
    req_reference_text = payload.get("reference_text")

    has_req_audio = bool(req_reference_audio)
    has_req_text = bool(req_reference_text)
    has_srv_audio = bool(CLONE_AUDIO and Path(CLONE_AUDIO).expanduser().exists())
    has_srv_text = bool(CLONE_TEXT)

    clone_source = "none"
    reference_audio = None
    reference_text = None
    clone_voice_match = bool(CLONE_VOICE) and (voice_id or "").strip().lower() == CLONE_VOICE.lower()

    if clone_voice_match:
        if has_req_audio and has_req_text:
            reference_audio = req_reference_audio
            reference_text = req_reference_text
            clone_source = "request"
        elif has_srv_audio and has_srv_text:
            reference_audio = _load_clone_audio_b64()
            if reference_audio:
                reference_text = CLONE_TEXT
                clone_source = "server"
            else:
                reference_audio = None
                reference_text = None
                clone_source = "none"
        else:
            reference_audio = None
            reference_text = None
            clone_source = "none"
    else:
        # Backwards compatible: honor request-provided clone data for other voices
        reference_audio = req_reference_audio
        reference_text = req_reference_text
        clone_source = "request" if has_req_audio else "none"

    if DEBUG_TTS:
        logging.info(
            "[TTS] clone_flags voice_id=%s has_req_audio=%s has_req_text=%s has_srv_audio=%s has_srv_text=%s source=%s",
            voice_id,
            has_req_audio,
            has_req_text,
            has_srv_audio,
            has_srv_text,
            clone_source,
        )

    vs = payload.get("voice_settings") or {}
    voice_settings = VoiceSettings(**vs) if vs else VoiceSettings()

    sampling_overrides = payload.get("sampling") or {}
    used_seed = _set_deterministic_seed(sampling_overrides.get("seed"))

    profile = VOICE_MAP.get(voice_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Unknown voice_id: {voice_id}")

    sampling = _settings_to_sampling(voice_settings)
    sampling = _apply_sampling_overrides(sampling, sampling_overrides)

    extra_instruct = ""
    if model_id and isinstance(model_id, str) and model_id.lower().startswith("style:"):
        extra_instruct = model_id.split(":", 1)[1].strip()

    instruct = ((profile.instruct or "") + " " + extra_instruct).strip()

    out_fmt = DEFAULT_OUTPUT_FORMAT
    if "audio/wav" in (accept or "").lower():
        out_fmt = "wav"

    tmp_ref_wav = None
    selected_model = None
    try:
        # If reference_audio is present: try clone-capable path
        if reference_audio:
            if not payload.get("language"):
                raise HTTPException(status_code=400, detail="language is required when reference_audio is provided.")
            tmp_ref_wav, _ = _decode_reference_audio_to_temp_wav(reference_audio)

            try:
                model = mm.get_qwen_base()
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Base clone model unavailable: {e}")

            selected_model = "base_clone"
            wavs, sr = _call_qwen_base_clone(
                model,
                text=text,
                language=payload.get("language") or profile.language,
                instruct=instruct,
                ref_wav_path=tmp_ref_wav,
                ref_text=reference_text,
                sampling=sampling,
            )

        else:
            # Normal path (unchanged)
            if profile.mode == "customvoice":
                model = mm.get_qwen_custom()
                selected_model = "custom_voice"
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=profile.language,
                    speaker=profile.speaker,
                    instruct=instruct,
                    **sampling,
                )
            else:
                model = mm.get_qwen_design()
                selected_model = "voice_design"
                if profile.speaker:
                    try:
                        wavs, sr = model.generate_voice_design(
                            text=text,
                            language=profile.language,
                            speaker=profile.speaker,
                            instruct=instruct,
                            **sampling,
                        )
                    except TypeError:
                        wavs, sr = model.generate_voice_design(
                            text=text,
                            language=profile.language,
                            instruct=instruct,
                            **sampling,
                        )
                else:
                    wavs, sr = model.generate_voice_design(
                        text=text,
                        language=profile.language,
                        instruct=instruct,
                        **sampling,
                    )

        wav0 = wavs[0]
        if TTS_LEADIN_MS > 0:
            wav0 = _prepend_silence(wav0, sr, TTS_LEADIN_MS)
        if TTS_TAIL_MS > 0:
            wav0 = _append_silence(wav0, sr, TTS_TAIL_MS)

        wav_bytes = _encode_wav_bytes(wav0, sr)
        if out_fmt == "mp3":
            audio_bytes = _wav_bytes_to_mp3_bytes(wav_bytes)
            media_type = "audio/mpeg"
        else:
            audio_bytes = wav_bytes
            media_type = "audio/wav"

        meta = {
            "voice_id": voice_id,
            "mode": profile.mode,
            "model_selected": selected_model,
            "seed": used_seed,
            "do_sample": sampling.get("do_sample", None),
            "temperature": sampling.get("temperature", None),
            "top_p": sampling.get("top_p", None),
            "top_k": sampling.get("top_k", None),
            "repetition_penalty": sampling.get("repetition_penalty", None),
            "lead_in_ms": TTS_LEADIN_MS,
            "tail_ms": TTS_TAIL_MS,
            "used_reference_audio": bool(reference_audio),
        }

        return {"media_type": media_type, "audio_bytes": audio_bytes, "meta": meta}

    finally:
        if tmp_ref_wav:
            try:
                os.remove(tmp_ref_wav)
            except Exception:
                pass

def run_stt_blocking(payload: Dict[str, Any]) -> Dict[str, Any]:
    ap = str(Path(payload["audio_path"]).expanduser().resolve())
    if not os.path.exists(ap):
        raise HTTPException(status_code=400, detail=f"audio_path not found: {ap}")

    language = payload.get("language") or STT_LANGUAGE
    if isinstance(language, str) and language.lower() == "auto":
        language = None

    task = payload.get("task") or "transcribe"
    prompt = payload.get("prompt")

    model = mm.get_whisper()
    segments, info = model.transcribe(
        ap,
        task=task,
        language=language,
        beam_size=STT_BEAM_SIZE,
        vad_filter=STT_VAD_FILTER,
        word_timestamps=STT_WORD_TIMESTAMPS,
        initial_prompt=prompt,
    )
    return _segments_to_verbose_json(list(segments), info)

# -----------------------------------------------------------------------------
# FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="GPU Audio Broker (8GB-safe)", version="1.2.0")

# Example curl (normal TTS):
# curl -X POST "http://localhost:9910/v1/text-to-speech/valentino" \
#   -H "Content-Type: application/json" -H "Accept: audio/wav" \
#   -d '{"text":"Hello world","language":"English","voice_settings":{"stability":0.95,"similarity_boost":0.85}}' \
#   -o out.wav
#
# Example curl (clone TTS using base model):
# curl -X POST "http://localhost:9910/v1/text-to-speech/valentino" \
#   -H "Content-Type: application/json" -H "Accept: audio/wav" \
#   -d '{"text":"Hello world","language":"English","reference_audio":"<base64_wav>","reference_text":"<optional>","voice_settings":{"stability":0.95,"similarity_boost":0.85}}' \
#   -o out.wav

@app.on_event("startup")
async def _startup():
    asyncio.create_task(gpuq.worker_loop())

@app.get("/health")
async def health():
    async with gpuq.lock:
        return {
            "ok": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_count_visible": torch.cuda.device_count(),
            "queue": {
                "max_queue_seconds": MAX_QUEUE_SECONDS,
                "queued_seconds": gpuq.queued_seconds,
                "jobs_waiting": gpuq.q.qsize(),
                "active_job": (gpuq.active_job.id if gpuq.active_job else None),
            },
            "model_manager": {
                "active": mm.active,
                "idle_ttl": MODEL_IDLE_TTL,
            }
        }

@app.get("/v1/voices")
def list_voices():
    return {
        "voices": [
            {"voice_id": v.id, "name": v.name, "description": v.description, "category": "qwen3-tts"}
            for v in VOICE_PROFILES
        ]
    }

@app.post("/v1/text-to-speech/{voice_id}")
async def tts(
    voice_id: str,
    req: TTSRequest,
    request: Request,
    xi_api_key: Optional[str] = Header(default=None, alias="xi-api-key"),
    accept: Optional[str] = Header(default=None),
):
    if REQUIRE_API_KEY:
        if not API_KEY:
            raise HTTPException(status_code=500, detail="Server misconfigured: API_KEY not set.")
        if not xi_api_key or xi_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid xi-api-key.")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Missing 'text'.")

    if voice_id not in VOICE_MAP:
        raise HTTPException(status_code=404, detail=f"Unknown voice_id: {voice_id}. Try GET /v1/voices")

    body = None
    if not (req.reference_audio or req.reference_audio_b64):
        try:
            body = await request.json()
        except Exception:
            body = None

    reference_audio = req.reference_audio or req.reference_audio_b64
    if not reference_audio and isinstance(body, dict):
        reference_audio = (
            body.get("reference_audio")
            or body.get("reference_audio_b64")
            or body.get("reference_audio_base64")
            or body.get("clone_audio")
        )

    est = estimate_tts_seconds(req.text)
    fut = asyncio.get_running_loop().create_future()

    job = Job(
        id=str(uuid.uuid4()),
        kind="tts",
        est_seconds=est,
        payload={
            "voice_id": voice_id,
            "text": req.text,
            "model_id": req.model_id,
            "language": req.language,
            "voice_settings": (req.voice_settings.model_dump() if req.voice_settings else None),
            "sampling": (req.sampling.model_dump() if req.sampling else None),
            "reference_audio": reference_audio,  # NEW, optional
            "reference_audio_b64": None,
            "reference_text": req.reference_text,
            "accept": accept or "",
        },
        future=fut,
    )

    await gpuq.try_enqueue(job)

    try:
        result = await asyncio.wait_for(fut, timeout=JOB_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timed out waiting for GPU queue.")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"TTS failed: {type(e).__name__}: {e}")

    meta = result.get("meta") or {}
    headers = {
        "X-TTS-Voice-Id": str(meta.get("voice_id") or ""),
        "X-TTS-Seed": "" if meta.get("seed") is None else str(meta.get("seed")),
        "X-TTS-Do-Sample": "" if meta.get("do_sample") is None else str(meta.get("do_sample")).lower(),
        "X-TTS-Temperature": "" if meta.get("temperature") is None else str(meta.get("temperature")),
        "X-TTS-Top-P": "" if meta.get("top_p") is None else str(meta.get("top_p")),
        "X-TTS-Top-K": "" if meta.get("top_k") is None else str(meta.get("top_k")),
        "X-TTS-Repetition-Penalty": "" if meta.get("repetition_penalty") is None else str(meta.get("repetition_penalty")),
        "X-TTS-Lead-In-Ms": "" if meta.get("lead_in_ms") is None else str(meta.get("lead_in_ms")),
        "X-TTS-Used-Reference-Audio": "true" if meta.get("used_reference_audio") else "false",
    }

    async def streamer():
        chunk = 1024 * 64
        bio = io.BytesIO(result["audio_bytes"])
        while True:
            b = bio.read(chunk)
            if not b:
                break
            yield b

    return StreamingResponse(streamer(), media_type=result["media_type"], headers=headers)

@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form(default="faster-whisper"),
    language: str = Form(default="auto"),
    task: str = Form(default="transcribe"),
    response_format: str = Form(default="verbose_json"),
    prompt: Optional[str] = Form(default=None),
):
    if response_format != "verbose_json":
        raise HTTPException(status_code=400, detail="Only response_format=verbose_json supported")

    suffix = Path(file.filename or "audio").suffix or ".mp3"
    content = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(content)

    est = estimate_stt_seconds(len(content))
    fut = asyncio.get_running_loop().create_future()

    job = Job(
        id=str(uuid.uuid4()),
        kind="stt",
        est_seconds=est,
        payload={
            "audio_path": tmp_path,
            "language": language,
            "task": task,
            "prompt": prompt,
        },
        future=fut,
    )

    try:
        await gpuq.try_enqueue(job)
        result = await asyncio.wait_for(fut, timeout=JOB_TIMEOUT_SECONDS)
        return JSONResponse(result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timed out waiting for GPU queue.")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/v1/audio/transcriptions_path")
async def transcriptions_path(req: TranscribePathRequest):
    if req.response_format and req.response_format != "verbose_json":
        raise HTTPException(status_code=400, detail="Only response_format=verbose_json supported")

    est = 30
    fut = asyncio.get_running_loop().create_future()

    job = Job(
        id=str(uuid.uuid4()),
        kind="stt",
        est_seconds=est,
        payload={
            "audio_path": req.audio_path,
            "language": req.language,
            "task": req.task or "transcribe",
            "prompt": req.prompt,
        },
        future=fut,
    )

    await gpuq.try_enqueue(job)
    try:
        result = await asyncio.wait_for(fut, timeout=JOB_TIMEOUT_SECONDS)
        return JSONResponse(result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Timed out waiting for GPU queue.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
