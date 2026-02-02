# mod — modular short‑video pipeline

This is a minimal, modular port of the ai‑tthub workflow. It:
- Picks a topic and generates a script via ChatGPT
- Generates an image from the workflow backend (Flux or OpenAI)
- Builds a short video with background music
- Overlays on‑screen text

Everything is configurable via files so you can spin up new themed channels by editing prompts and config files.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="..."
# Optional (for background music)
export FREESOUND_API_KEY="..."

python3 pipeline.py
```

## Configuration
All settings live in `config.json`. Key sections:
- `openai` → model, temperature range, retry settings
- `image_backend` → `flux` or `openai`
- `video` → size, duration, music tag, fade in/out
- `overlay` → fonts, sizes, colors, CTA text
- `paths` → output folders

### Image backend
- Flux (workflow server):
  - `image_backend.type = "flux"`
  - Set `image_backend.flux_api_url` and `image_backend.flux_workflow_name`
- OpenAI images:
  - `image_backend.type = "openai"`

## Modular prompts (change these to create a new channel)
Prompts and seeds are loaded from files under `prompts/`:
- `prompts/script_system.txt` — system prompt
- `prompts/script_templates.json` — script templates
- `prompts/seo_system.txt` — SEO system prompt
- `prompts/seo_user.txt` — SEO user prompt template
- `prompts/topic_seeds.json` — topic seed lists
- `prompts/hooks.json` — hook patterns
- `prompts/structures.json` — structure names
- `prompts/twists.json` — twist phrases

You can copy the `prompts/` folder, edit text, and run the pipeline to generate a new themed channel without touching code.

## Outputs
- `ready/` — assembler JSON
- `visuals/` — generated images
- `final/` — completed videos
- `memory/` — recent topics/hooks/structures and history

## Notes
- Fonts live in `fonts/` and are referenced in `config.json`.
- Background music defaults to `fallbacks/default_bg_music.mp3` if Freesound is not configured.

## Uploaders + Comment Bot
This repo includes:
- `ytuploader.py`
- `fbupload.py`
- `igupload.py`
- `comment.py`

Token refresh uses `social_tokens.py` and caches to `fb_token.json`.

### Enable auto uploads
Uncomment the upload block at the bottom of `pipeline.py`.

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
