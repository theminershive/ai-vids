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

## Local Ollama LLM (optional)
Set these in `.env` to use your local Ollama server:
- `LLM_PROVIDER=ollama` (or `auto`)
- `OLLAMA_BASE_URL=http://192.168.1.176:11434`
- `LLM_MODEL=qwen2.5:14b-instruct-q4_K_M`
- `OLLAMA_NUM_CTX=8192` (optional)
- `OLLAMA_TEMPERATURE=0.7` (optional)

Verify Ollama:
```bash
curl http://192.168.1.176:11434/api/tags
curl http://192.168.1.176:11434/api/chat -d '{
  "model":"qwen2.5:14b-instruct-q4_K_M",
  "messages":[{"role":"user","content":"Hello"}],
  "stream":false
}'
```

Smoke test script:
```bash
python3 scripts/ollama_smoke_test.py
```

## Render existing book JSONs
If you already have assembler JSON files under `ready/` (e.g., from `book_json_generator.py`),
you can render them into captioned + overlayed videos without regenerating topics:

```bash
python3 render_ready.py --json ready/my_book_ep001_assembler.json
```

Render all:
```bash
python3 render_ready.py --all
```

## Queue mode (auto-advance + new book zips)
Process one ready JSON per run, then auto-generate from new .zip books:

```bash
python3 run_queue.py
```

Behavior:
- If there is a pending JSON in `ready/`, it renders the next one.
- If not, it looks for a new `.zip` (recursively), generates all episodes,
  then renders the first generated JSON.
- Progress is tracked in `memory/processed_ready.json` and `memory/processed_books.json`.

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
