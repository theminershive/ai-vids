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

## TTS clone options (optional)
Add any of these to `.env` to enable voice cloning or safer fallback for clone voices:
- `TTS_VALENTINO_VOICE_CLONE_PROMPT` (string or file path)
- `TTS_VALENTINO_REF_AUDIO` (file path, URL, or base64)
- `TTS_VALENTINO_REF_TEXT` (string)
- `TTS_X_VECTOR_ONLY_MODE=1` (skip ref_text if supported)
- `TTS_DRY_RUN=1` (skip HTTP and write empty audio for smoke tests)

## Notes
- ComfyUI is the default backend. Set `image_backend.visual_backend` to `leonardo` and configure `LEONARDO_API_KEY` to use Leonardo instead.
- If `FREESOUND_API_KEY` is missing, the pipeline falls back to `fallbacks/default_bg_music.mp3`.

## ComfyUI API Debugging

Enable detailed ComfyUI API logging to compare GUI vs API workflows:

1. Set in `.env`:
   - `DEBUG_COMFY_API=1`
   - `ENFORCE_WORKFLOW_JSON=1` (optional: use exact workflow JSON)
   - `ENFORCED_WORKFLOW_PATH=./mod2/workflow.json`
   - `ENFORCED_PROMPT_NODE_ID=68:6`
   - `ENFORCED_PROMPT_FIELD=text`
2. Run the pipeline as normal.

Outputs:
- `logs/comfy_api_debug.log` (rotating log with full request/response + timings)
- `logs/last_api_workflow.json` (exact workflow JSON sent to ComfyUI)

Tip: Export the workflow from the ComfyUI GUI and diff it against `logs/last_api_workflow.json`
to confirm node IDs, models, steps, sampler, LoRA, resolution, and dtype match.

## Uploaders + Comment Bot
This repo includes:
- `ytuploader.py`
- `fbupload.py`
- `igupload.py`
- `comment.py`

Token refresh uses `social_tokens.py` and caches to `fb_token.json`.

### Enable auto uploads
Uncomment the upload block at the bottom of `app.py`.

### Required env
- `APP_ID`, `APP_SECRET`, `SHORT_LIVED_TOKEN`
- `FACEBOOK_PAGE_ID`, `INSTAGRAM_ACCOUNT_ID`, `PUBLIC_IP`
- `FB_PAGE_ACCESS_TOKEN` (optional; auto-derived)
- `token2.json` for YouTube (refreshes automatically)

## Upload toggles
Set these in `.env` to enable uploads:
- `UPLOAD_YOUTUBE=1`
- `UPLOAD_FACEBOOK=1`
- `UPLOAD_INSTAGRAM=1`
