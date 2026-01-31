import os
import json
import requests
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env
load_dotenv()

# Constants
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 1024

if not ELEVENLABS_API_KEY:
    print("ElevenLabs API key is missing in .env or not being loaded.")
else:
    print("ElevenLabs API key loaded successfully.")

# Define the available voices with their names and corresponding IDs
VOICE_OPTIONS = {
    "David - American Narrator": {
        "id": "v9LgF91V36LGgbLX3iHW",
        "description": "American middle-aged male. Professional voice actor. Perfect for Narrations."
    },
    "Luca": {
        "id": "4JVOFy4SLQs9my0OLhEw",
        "description": "A young adult American male with a calm, almost sober, slightly breathy way of talking. Great for voiceovers and narrations of all kinds."
    },
    "Jessica Anne Bogart - Conversations": {
        "id": "g6xIsTj2HwM6VR4iXFCw",
        "description": "Friendly and Conversational Female voice. Articulate, Confident and Helpful. Works well for Conversations."
    },
    "Simeon": {
        "id": "alMSnmMfBQWEfTP8MRcX",
        "description": "A middle aged male authoritative, deep, and calm voice. Good for audiobooks and storytelling."
    },
    "Christopher": {
        "id": "G17SuINrv2H9FC6nvetn",
        "description": "British male narrator, English, well-spoken, gentle and trustworthy voice. Great for audiobooks, podcasts and voiceovers."
    },
    "Jessica2 Anne Bogart - Character and Animation": {
        "id": "flHkNRp1BlvT73UL6gyz",
        "description": "The Villain! Wickedly eloquent. Calculating. Cruel and calm."
    },
    "Brian Overturf": {
        "id": "ryn3WBvkCsp4dPZksMIf",
        "description": "Middle aged Male voice. Perfect for radio."
    },
    "Bill Oxley": {
        "id": "T5cu6IU92Krx4mh43osx",
        "description": "Middle aged American male with a clear and natural voice. Perfect for audio book and documentary narration."
    },
    "Bradley - Earnest narrator": {
        "id": "RexqLjNzkCjWogguKyff",
        "description": "Middle-aged American male, baritone voice, based on religious audiobook. Earnest and kind."
    },
    "Valentino": {
        "id": "Nv8Euon5i3G2sBJM47fo",
        "description": "A great voice with depth. The voice is deep with a great accent, and works well for meditations and narrations."
    },
    "Parker Springfield - Professional Broadcaster": {
        "id": "Dnd9VXpAjEGXiRGBf1O6",
        "description": "Middle-aged American male with a casual yet fun delivery that sounds natural. Voice is perfect if you are looking for a professional radio announcer."
    },
    "Frederick Surrey": {
        "id": "j9jfwdrw7BRfcR43Qohk",
        "description": "Professional, calm, well spoken British narrator full of intrigue and wonder. Suitable for Nature, Science, Mystery & History documentaries or audio books & narration projects that need a smooth & velvety tone."
    }
}

def get_voice_id(tone):
    """
    Retrieve the voice ID based on the tone name.
    If the tone is not found, return a default voice ID.
    """
    voice = VOICE_OPTIONS.get(tone)
    if voice:
        print(f"Selected voice: {tone} (ID: {voice['id']})")
        return voice['id']
    else:
        print(f"Tone '{tone}' not found. Using default voice.")
        return VOICE_OPTIONS["Valentino"]["id"]  # Default to Valentino if tone not found

def generate_tts_elevenlabs(narration_text, audio_path, voice_id, stability=0.3, similarity_boost=0.7):
    """
    Generate TTS audio using the ElevenLabs API and save it to a file.
    """
    try:
        # API URL
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

        # Headers
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
        }

        # Request Data
        data = {
            "text": narration_text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        }

        # Make the API request
        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            # Save the audio content
            with open(audio_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
            print(f"Audio content saved to {audio_path}")
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"An error occurred while generating TTS with ElevenLabs: {e}")
        return False

def process_tts(script_data, audio_dir=AUDIO_DIR):
    """
    Process the script JSON, generate audio for each narration segment,
    and update the JSON with audio paths.
    Supports both short and long video JSON structures.
    """
    if not ELEVENLABS_API_KEY:
        print("ElevenLabs API key is not available. Exiting process.")
        return script_data

    # Get the tone from the script settings
    tone = script_data.get("tone", "Valentino")  # Default to Valentino if tone not specified
    voice_id = get_voice_id(tone)

    sections = script_data.get("sections", [])
    for section_idx, section in enumerate(sections, start=1):
        segments = section.get("segments", [])
        if not segments:
            print(f"\nSection {section_idx} has no segments. Skipping.")
            continue

        for segment_idx, segment in enumerate(segments, start=1):
            narration = segment.get("narration", {})
            text = narration.get("text", "")

            if not text:
                print(f"\nSection {section_idx}, Segment {segment_idx} has no narration text. Skipping.")
                continue

            print(f"\nGenerating TTS for Section {section_idx}, Segment {segment_idx}:")
            print(f"Text: {text}")

            # Save audio with section and segment-specific filename
            audio_filename = f"section_{section_idx}_segment_{segment_idx}.mp3"
            audio_path = audio_dir / audio_filename

            # Call the ElevenLabs TTS API with the selected voice_id
            success = generate_tts_elevenlabs(text, audio_path, voice_id)
            if success:
                segment["narration"]["audio_path"] = str(audio_path)
            else:
                segment["narration"]["audio_path"] = None

    return script_data

def save_audio_paths(updated_script, filename="video_script_with_audio.json"):
    """
    Save the updated script JSON with audio paths to a file.
    """
    script_path = AUDIO_DIR.parent / filename
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(updated_script, f, indent=4)
        print(f"Updated script with audio paths saved to {script_path}")
        return script_path
    except Exception as e:
        print(f"An error occurred while saving the updated script: {e}")
        return None

def load_script_from_json(json_path):
    """
    Load script data from a specified JSON file.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            script_data = json.load(f)
        print(f"Loaded script from {json_path}")
        return script_data
    except Exception as e:
        print(f"Failed to load JSON file: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Prompt user for JSON file path
    json_path = input("Enter the path to the JSON file to use: ").strip()

    # Validate JSON file
    if not os.path.exists(json_path):
        print(f"The specified JSON file does not exist: {json_path}")
    else:
        script_data = load_script_from_json(json_path)
        if script_data:
            updated_script = process_tts(script_data)
            if updated_script:
                save_audio_paths(updated_script)
