# Voice Registry

The broker exposes voice IDs via `GET /v1/voices`. This list is driven by `VOICE_PROFILES` in `app.py`.

## How it works
- `/v1/voices` returns the registered `VOICE_PROFILES`.
- `/v1/text-to-speech/{voice_id}` only accepts IDs present in `VOICE_PROFILES`.
- Each voice defines a `mode`:
  - `customvoice`: uses `generate_custom_voice(..., speaker=...)`
  - `voicedesign`: uses `generate_voice_design(...)` (optionally `speaker=` if supported)

## Add a built-in voice
1. Edit `VOICE_PROFILES` in `app.py`:
   ```python
   VoiceProfile(
       id="sohee",
       name="Sohee (Built-in)",
       description="Built-in female voice.",
       mode="voicedesign",
       language="English",
       speaker="sohee",
       instruct="Warm, calm, clear. Dry studio."
   )
   ```
2. Restart the broker.
3. Confirm with `GET /v1/voices`.

## Add a clone voice (server defaults)
Set env vars and point to a local mp3/wav + transcript:
```
CLONE_VOICE=valentino
CLONE_AUDIO=./voiceclone.mp3
CLONE_TEXT=Welcome to Daily Bible Passages. Today's verses are a short reading for reflection.
```

## Clone priority order
1. If the request provides BOTH `reference_audio` and `reference_text`, those are used.
2. Else if `voice_id == CLONE_VOICE` and `CLONE_AUDIO + CLONE_TEXT` are set, server defaults are used.
3. Else normal (non-clone) generation path is used.

Notes:
- Built-in voices never require reference audio.
- Unknown `voice_id` still returns 404.
