# Reel Creator

This application automates the creation of short vertical videos (e.g., TikTok/Reels) using AI.

## Features

1. **Script Generation**: Uses OpenAI to create a video script.
2. **Text-to-Speech**: Generates narration audio via ElevenLabs.
3. **Visual Assets**: Fetches images and optional motion clips via Leonardo AI.
4. **Assembly**: Assembles clips, narration, transitions, and background music using MoviePy.
5. **Captions**: Generates captions via Whisper and overlays them.
6. **Overlays**: Adds header/footer call-to-action text.

## Setup

1. Clone this repository.
2. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the example environment file and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
   - `OPENAI_API_KEY`
   - `ELEVENLABS_API_KEY`
   - `LEONARDO_API_KEY`
   - `FREESOUND_API_KEY`

## Usage

Run the main script and follow prompts:

```bash
python app.py
```

Videos and assets will be saved under `output/`.

## TTS clone options (optional)
Add any of these to `.env` to enable voice cloning or safer fallback for clone voices:
- `TTS_VALENTINO_VOICE_CLONE_PROMPT` (string or file path)
- `TTS_VALENTINO_REF_AUDIO` (file path, URL, or base64)
- `TTS_VALENTINO_REF_TEXT` (string)
- `TTS_X_VECTOR_ONLY_MODE=1` (skip ref_text if supported)
- `TTS_DRY_RUN=1` (skip HTTP and write empty audio for smoke tests)

## ComfyUI API Debugging

Enable detailed ComfyUI API logging to compare GUI vs API workflows:

1. Set in `.env`:
   - `DEBUG_COMFY_API=1`
   - `ENFORCE_WORKFLOW_JSON=1` (optional: use exact workflow JSON)
   - `ENFORCED_WORKFLOW_PATH=./dailybible/workflow.json`
   - `ENFORCED_PROMPT_NODE_ID=68:6`
   - `ENFORCED_PROMPT_FIELD=text`
2. Run the pipeline as normal.

Outputs:
- `logs/comfy_api_debug.log` (rotating log with full request/response + timings)
- `logs/last_api_workflow.json` (exact workflow JSON sent to ComfyUI)

Tip: Export the workflow from the ComfyUI GUI and diff it against `logs/last_api_workflow.json`
to confirm node IDs, models, steps, sampler, LoRA, resolution, and dtype match.

## Upload toggles
Set these in `.env` to enable uploads:
- `UPLOAD_YOUTUBE=1`
- `UPLOAD_FACEBOOK=1`
- `UPLOAD_INSTAGRAM=1`
