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

