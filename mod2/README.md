# mod2 — topic‑driven dailybible‑style pipeline

This build combines the **topic selection + script generation** from `mod` with the **multi‑image, TTS, captions, and assembly** workflow from `dailybible`.

It will:
- pick a topic (or use a provided one)
- generate a multi‑segment script + visual prompts
- generate multiple images
- synthesize TTS per segment
- assemble a full video
- add on‑screen captions
- apply start/end text overlay

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="..."
# Optional (music)
export FREESOUND_API_KEY="..."
# Optional (TTS + captions)
export ELEVENLABS_API_KEY="..."
# or
export TTS_BACKEND=qwen
export QWEN_TTS_API_BASE="http://<tts-host>:9910"

python3 app.py
```

## Generate with a specific topic
```bash
python3 assemble.py --topic "the lost polar expedition" --structure timeline_reveal
python3 app.py
```

## Modular configuration
All settings live in `config.json`. Key areas:
- `prompts/*` — edit prompt templates to create new themes
- `topics/*` — edit topic seed lists/hooks/structures
- `image_backend` — ComfyUI + Leonardo prompt rewrite settings
- `video` — size, bg music, transitions
- `captions` — font and caption layout
- `overlay` — start/end text

## Outputs
- `ready/` — assembler JSON
- `visuals/` — generated images
- `audio/` — TTS audio
- `final/` — final videos
- `memory/` — topic/hook/structure history

## Notes
- ComfyUI is the default backend. Set `image_backend.visual_backend` to `leonardo` and configure `LEONARDO_API_KEY` to use Leonardo instead.
- If `FREESOUND_API_KEY` is missing, the pipeline falls back to `fallbacks/default_bg_music.mp3`.
