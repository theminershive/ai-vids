
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
VIDEO_SCRIPTS_DIR = OUTPUT_DIR / "video_scripts"
AUDIO_DIR = OUTPUT_DIR / "audio"
VISUALS_DIR = OUTPUT_DIR / "visuals"
CAPTIONS_DIR = OUTPUT_DIR / "captions"
FINAL_VIDEO_DIR = OUTPUT_DIR / "final"

# Create directories if they don't exist
for directory in [VIDEO_SCRIPTS_DIR, AUDIO_DIR, VISUALS_DIR, CAPTIONS_DIR, FINAL_VIDEO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Environment Variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LEONARDO_API_KEY = os.getenv('LEONARDO_API_KEY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
FREESOUND_API_KEY = os.getenv('FREESOUND_API_KEY')

# Video Settings
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
VIDEO_SIZE = (VIDEO_WIDTH, VIDEO_HEIGHT)
FPS = 24
SCRIPT_DURATION_SECONDS = 59

# Caption Settings
CAPTION_SETTINGS = {
    "TEXT_SIZE": 85,
    "FONT": "Bangers-Regular.ttf",
    "COLOR": 'white',
    "STROKE_COLOR": 'black',
    "STROKE_WIDTH": 1,
    "METHOD": 'caption',
    "CAPTION_POSITION": (120, 1240),
    "SUBTITLE_MAX_WIDTH": 0.8,
    "ENABLE_BACKGROUND_BOX": True,
    "BACKGROUND_COLOR": (0, 0, 0),
    "BACKGROUND_OPACITY": 0.5,
    "BACKGROUND_PADDING_WIDTH": 20,
    "BACKGROUND_PADDING_HEIGHT": 20,
    "ROUND_CORNER_RADIUS": 30,
    "BACKGROUND_BOX_POSITION": (120, 850),
    "CUSTOM_BOX_POSITION_OFFSET": (0, 0),
    "FADE_IN_DURATION": 0,
    "FADE_OUT_DURATION": 0,
}

# Leonardo AI Configuration
LEONARDO_MODEL_ID = "b24e16ff-06e3-43eb-8d33-4416c2d75876"
LEONARDO_WIDTH = 864
LEONARDO_HEIGHT = 1536
LEONARDO_NUM_IMAGES = 1
LEONARDO_ALCHEMY = False
LEONARDO_MOTION_STRENGTH = 5

# Other Configurations
MAX_SCRIPT_TOKENS = 5000
MAX_RETRIES = 3

# Music & Sound Settings
BACKGROUND_MUSIC_USER = "Nancy_Sinclair"
MUSIC_TYPES = ["cinematic", "ambient", "suspense", "upbeat", "melodic", "neutral", "inspiring", "dramatic"]
TRANSITION_EFFECTS = ["swoosh", "fade-in", "whoosh", "glimmer"]
