import os
from pathlib import Path

os.environ.setdefault("TTS_DRY_RUN", "1")
os.environ.setdefault("DEBUG_TTS", "1")

import tts


def _out_path(label: str) -> Path:
    ext = "mp3" if tts.TTS_FORMAT == "mp3" else "wav"
    safe = label.lower().replace(" ", "_")
    return Path("audio") / f"_smoke_{safe}.{ext}"


def run_case(label: str) -> None:
    out = _out_path(label)
    ok = tts.generate_tts("Test line for smoke test.", out, label)
    size = out.stat().st_size if out.exists() else 0
    print(f"voice={label} ok={ok} exists={out.exists()} size={size}")


if __name__ == "__main__":
    run_case(getattr(tts, "DEFAULT_TONE_NAME", "Bible Reader"))
    run_case("Valentino")
