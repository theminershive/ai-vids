import math
import openai
import json
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai.error import OpenAIError
import random
import re
from video_assembler import search_sounds, is_banned

from voices_and_styles import VOICES, MODELS
# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
logger = logging.getLogger(__name__)

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VIDEO_SCRIPTS_DIR = "./output/video_scripts/"
MAX_SCRIPT_TOKENS = 16000  # Initial value; will be adjusted based on video length

# Set OpenAI API Key
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY is not set. Check your .env file.")
    exit(1)
openai.api_key = OPENAI_API_KEY
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Define available voices


# Define models with their descriptions and example prompts


def generate_background_music(length):
    """
    Select background music types based on video length.
    Returns a comma-separated string of one or two music types.
    """
    music_types = ["cinematic", "ambient", "suspense", "upbeat", "melodic", "neutral", "inspiring", "dramatic"]
    selected = random.sample(music_types, 2) if length > 120 else random.sample(music_types, 1)
    logger.debug(f"Selected background music: {selected}")
    return ", ".join(selected)

def generate_transition_effect():
    """
    Select a transition effect type.
    """
    transition_effects = ["swoosh", "fade-in", "whoosh", "glimmer"]
    effect = random.choice(transition_effects)
    logger.debug(f"Selected transition effect: {effect}")
    return effect

def call_openai_api(messages, max_tokens, temperature):
    """
    Calls the OpenAI API with the given messages, max tokens, and temperature.
    """
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        logger.debug("OpenAI API call successful.")
        return response
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        return None

def calculate_max_tokens(length):
    """
    Calculate max tokens based on video length.
    Overridden to always allow ample token budget.
    """
    return 8000

def select_background_music_via_gpt(topic, music_options):
    """
    Selects background music randomly from options.
    """
    try:
        return random.choice(music_options)
    except Exception:
        return "neutral"

def call_openai_api_generate_script(prompt, max_tokens, temperature):
    """
    Calls OpenAI API to generate the video script based on the provided prompt.
    """
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        logger.debug("OpenAI API call for script generation successful.")
        return response
    except OpenAIError as e:
        logger.error(f"OpenAI API error during script generation: {e}")
        return None

def generate_video_script(topic, length, size, num_sections, num_segments):
    """
    Generates a comprehensive video script based on the provided parameters.
    Args:
        topic (str): The topic for the video script.
        length (int): Total length of the video in seconds.
        size (str): Size of the video (e.g., "1080x1920").
        num_sections (int): Number of sections in the video.
        num_segments (int): Number of segments per section.
    Returns:
        dict: The generated video script data.
    """
    try:
        
        # --- New Better Duration Calculation ---
        TARGET_MAIN_SEGMENT_DURATION = 4  # seconds per main segment
        TARGET_SEGUE_SEGMENT_DURATION = 2  # seconds per segue segment
        MIN_SEGMENT_DURATION = 3  # absolute minimum to avoid being too fast
        MAX_SEGMENT_DURATION = 5  # absolute maximum to avoid dragging

        # Total main segments = sections * segments
        total_main_segments = num_sections * num_segments

        # How many transitions/segues? One less than number of sections
        num_segue_sections = max(0, num_sections - 1)

        # How many hook + outro segments? 2
        num_intro_outro = 2

        # Total segments = main + segues + hook/outro
        total_segments = total_main_segments + num_segue_sections + num_intro_outro

        # Calculate how much time is available for main content
        total_segue_time = num_segue_sections * TARGET_SEGUE_SEGMENT_DURATION
        total_main_time = max(0, length - total_segue_time)

        # Calculate raw main segment duration
        if total_main_segments > 0:
            raw_main_segment_duration = total_main_time / (total_main_segments + 2)  # +2 for hook and outro
        else:
            raw_main_segment_duration = TARGET_MAIN_SEGMENT_DURATION

        # Clamp segment duration within safe limits
        main_segment_duration = max(MIN_SEGMENT_DURATION, min(MAX_SEGMENT_DURATION, int(raw_main_segment_duration)))

        logger.debug(f"Main segment duration: {main_segment_duration} seconds")
        logger.debug(f"Segue segment duration: {TARGET_SEGUE_SEGMENT_DURATION} seconds")

        sections = []
        sections = []

        # HOOK
        sections.append({
            "section_number": 1,
            "title": "Hook: Attention-Grabbing Opener",
            "section_duration": main_segment_duration,
            "segments": [{
                "segment_number": 1,
                "narration": {
                    "text": "Hook narration here.",
                    "start_time": 0,
                    "duration": main_segment_duration
                },
                "visual": {
                    "type": "image",
                    "prompt": f"High-impact visual for opening on topic: {topic}.",
                    "start_time": 0,
                    "duration": main_segment_duration,
                },
                "sound": {
                    "transition_effect": generate_transition_effect()
                }
            }]
        })

        # Main sections
        for i in range(num_sections):
            section = {}
            section_number = i + 2
            section["section_number"] = section_number
            section["title"] = f"Section {i + 1}: Title"
            section["section_duration"] = num_segments * main_segment_duration
            segments = []
            for j in range(num_segments):
                global_segment_index = 1 + i * num_segments + j  # shift start by 1 for HOOK
                segment_number = j + 1
                segment = {
                    "segment_number": segment_number,
                    "narration": {
                        "text": "Narration text here.",
                        "start_time": global_segment_index * main_segment_duration,
                        "duration": main_segment_duration
                    },
                    "visual": {
                        "type": "image",
                        "prompt": f"Detailed visual prompt tailored to the {topic} topic.",
                        "start_time": global_segment_index * main_segment_duration,
                        "duration": main_segment_duration,
                    },
                    "sound": {
                        "transition_effect": generate_transition_effect()
                    }
                }
                segments.append(segment)
            section["segments"] = segments
            sections.append(section)
            if i < num_sections - 1:
                segue_section = {
                    "section_number": section_number + 0.5,
                    "title": "Segue",
                    "section_duration": TARGET_SEGUE_SEGMENT_DURATION,
                    "segments": [
                        {
                            "segment_number": 1,
                            "narration": {
                                "text": "",
                                "start_time": section_number * main_segment_duration,
                                "duration": TARGET_SEGUE_SEGMENT_DURATION
                            },
                            "visual": {
                                "type": "image",
                                "prompt": "",
                                "start_time": section_number * main_segment_duration,
                                "duration": TARGET_SEGUE_SEGMENT_DURATION
                            },
                            "sound": {
                                "transition_effect": generate_transition_effect()
                            }
                        }
                    ]
                }
                sections.append(segue_section)

        # OUTRO
        outro_start = (total_segments - 1) * main_segment_duration
        sections.append({
            "section_number": num_sections + 2,
            "title": "Outro: Wrap-up and CTA",
            "section_duration": main_segment_duration,
            "segments": [{
                "segment_number": 1,
                "narration": {
                    "text": "Outro narration here.",
                    "start_time": outro_start,
                    "duration": main_segment_duration
                },
                "visual": {
                    "type": "image",
                    "prompt": f"Closing image to wrap up video on topic: {topic}.",
                    "start_time": outro_start,
                    "duration": main_segment_duration,
                },
                "sound": {
                    "transition_effect": generate_transition_effect()
                }
            }]
        })
# Construct social media metadata if length > 120
        social_media = {}
        if length > 10:
            social_media = {
                "title": "Suggested title for social media platforms.",
                "description": "Short and engaging description for social media platforms.",
                "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
            }

        # Construct the initial JSON structure without background_music
        script_json = {
            "settings": {
                "use_background_music": True,
                "use_transitions": True,
                "video_size": size
            },
            "sections": sections
        }

        if length > 10:
            script_json["social_media"] = social_media

        # Serialize the JSON for reference (optional)


        # Prepare dynamic sections to avoid backslash issues inside f-strings
        social_media_instructions = ""
        social_media_json_block = ""
        if length > 120:
            social_media_instructions = (
                "5. **Social Media Optimization:**\n"
                "   - Create a highly engaging, clickable title under 100 characters. Use action verbs, emotional triggers, curiosity, or surprising facts. Include 1â€“2 emojis naturally.\n"
                "   - Write a dynamic 1â€“2 sentence description (up to 500 characters) designed for YouTube Shorts, Instagram Reels, and Facebook Reels. Make it compelling, include one call-to-action (comment, share, or follow), and naturally weave in trending keywords.\n"
                "   - Provide a list of 8â€“12 high-performing hashtags blending popular general hashtags (#viral, #shorts, #trending) with topic-specific tags.\n"
                "   - Format exactly as:\n"
                "     {\n"
                "         \"title\": \"Optimized Title\",\n"
                "         \"description\": \"Optimized Description\",\n"
                "         \"tags\": [\"tag1\", \"tag2\", \"tag3\", ..., \"tag10\"]\n"
                "     }\n"
                "   - DO NOT add extra fields. Stay within the exact JSON structure."
            )
            social_media_json_block = f'"social_media": {json.dumps(social_media, indent=4)}'
        prompt_json = json.dumps(script_json, indent=4)

        prompt = f"""
        You are an expert video scriptwriter specializing in creating highly engaging, viral content for social media and YouTube Shorts.

        Your task: generate a ready-to-use JSON script for a {length}-second video on "{topic}". Follow **exactly** this template (no added, missing, or renamed keys):

        {prompt_json}

        Creative requirements:
        - Hook (0–3s): start with a surprising fact, provocative question, or intriguing statement to grab viewers immediately.
        - Main Content: split the remainder into {num_sections} sections × {num_segments} segments each. For each segment:
          • Begin with a punchy, curiosity-invoking line.
          • Write vivid, concise narration that sustains attention. 
          • Ensure the script flows smoothly over the entire length.
          • Smoothly segue into the next part.
        - Visual Guidance: for every segment, supply a descriptive `visual.prompt` that paints scroll-stopping imagery.
        - Transitions (Segues): between sections, insert a 1–2s “segue” segment with a brief voiceover tying ideas together and a matching visual prompt.
        - Outro (Closing): conclude with a strong call-to-action to like, comment, subscribe, or share.
        - Storytelling Flow: write in a lively, conversational tone—vary sentence length, use active voice, vivid verbs, and the occasional metaphor to avoid robotic monotony.
- Seamless Segues: craft engaging 1–2‑sentence transition voiceovers that explicitly connect the concepts between sections, using rhetorical questions or cliffhangers to keep viewers hooked.
- Viral Techniques: weave in humor, urgency, trending references, or cultural hooks to maximize shareability.

        Strict rules:
        • Fill **every** `"text"`, `"prompt"`, `"start_time"`, `"duration"`, and `"audio_path"` with concrete values.
        • Include the top-level `"social_media"` block **only if** `length > 10`, exactly as shown.
        • Do **not** add, remove, reorder, or rename any keys.
        • Output **only** the raw JSON—no markdown fences, no comments, nothing else.
        """

        response = call_openai_api_generate_script(
            prompt=prompt,
            max_tokens=MAX_SCRIPT_TOKENS,
            temperature=0.9
        )

        if not response:
            logger.error("Failed to retrieve video script.")
            return None

        script_content = response.choices[0].message['content']
        # Debug: save raw GPT response
        safe_topic_raw = re.sub(r'[^A-Za-z0-9]+', '_', topic.lower())[:50]
        script_content_clean = script_content.strip()
        script_content_clean = re.sub(r'^```(?:json)?\s*', '', script_content_clean)
        script_content_clean = re.sub(r'\s*```$', '', script_content_clean)
        # --- Crop to JSON object to remove any leading/trailing junk ---
        first = script_content_clean.find('{')
        last = script_content_clean.rfind('}')
        if first != -1 and last != -1:
            script_content_clean = script_content_clean[first:last+1]
        logger.debug(f"Cleaned response content length: {len(script_content_clean)}")
        # Save cleaned response for debugging
        script_content = script_content_clean
        # Auto-correct missing visual duration fields for malformed JSON
        import re as _re_c
        script_content = _re_c.sub(r'"duration"\s*:\s*\}', '"duration": 0}', script_content)

        # Balance braces and quotes in JSON to prevent decode errors
        open_braces = script_content.count('{')
        close_braces = script_content.count('}')
        if close_braces < open_braces:
            script_content += '}' * (open_braces - close_braces)
        # Ensure even number of quotation marks
        if script_content.count('"') % 2 != 0:
            script_content += '"'
        # --- Trim any extra closing braces or brackets ---
        while script_content.endswith('}') and script_content.count('}') > script_content.count('{'):
            script_content = script_content[:-1]
        while script_content.endswith(']') and script_content.count(']') > script_content.count('['):
            script_content = script_content[:-1]

        logger.debug(f"Raw response content:\n{script_content}")

        

        # --- begin added cleanup to prevent JSONDecodeError ---
        # 1. Extract only the JSON payload between the first and last brace
        start = script_content.find('{')
        end   = script_content.rfind('}')
        if start != -1 and end != -1 and end > start:
            script_content = script_content[start:end+1]
        
        # 2. Remove any trailing commas before a closing brace or bracket
        import re as _json_cleaner
        script_content = _json_cleaner.sub(r',\s*([\]}])', r'\1', script_content)
        # --- end added cleanup ---
        # 3. Replace any non-numeric start_time or duration values with 0
        script_content = _json_cleaner.sub(r'"(?:start_time|duration)"\s*:\s*(?!\d)([^,\}\n]+)', r'"\1": 0', script_content)
# Parse JSON
        try:
            script_data = json.loads(script_content.strip())
            logger.debug(f"Generated script data: {json.dumps(script_data, indent=2)}")
                        # Sanitize filename for raw JSON and limit length
            safe_topic_raw = re.sub(r'[^A-Za-z0-9]+', '_', topic.lower())
            safe_topic_raw = safe_topic_raw[:50]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode GPT script output: {e}")
            return None
        # Inject missing transition effects

        
        for sec in script_data.get("sections", []):

            for seg in sec.get("segments", []):

                seg.setdefault("sound", {})["transition_effect"] = seg.get("sound", {}).get("transition_effect", generate_transition_effect())


        # Combine narration texts for voice and style selection

        narration_texts = []

        for sec in script_data.get("sections", []):

            if "segments" in sec:

                for seg in sec["segments"]:

                    t = seg.get("narration", {}).get("text", "")

                    if t:

                        narration_texts.append(t)

            else:

                t = sec.get("narration", {}).get("text", "")

                if t:

                    narration_texts.append(t)


        combined_text = " ".join(narration_texts)


        # Select voice and style
        script_data["tone"] = select_voice(combined_text)
        selected_style, _ = select_style(combined_text)
        script_data["image_style"] = selected_style

        # --- Background music: ask GPT to pick the best fit from our known-good options ---
        MUSIC_OPTIONS = [
            "cinematic", "ambient", "suspense", "upbeat",
            "melodic", "neutral", "inspiring", "dramatic"
        ]
        selected_bg = select_background_music_via_gpt(topic, MUSIC_OPTIONS)
        script_data["background_music"] = selected_bg

        
        script_data["background_music_type"] = selected_bg# --- Ensure downstream compatibility keys ---
        # Derive safe base filename for video outputs
        safe_topic = re.sub(r'[^A-Za-z0-9]+', '_', topic)
        base_filename = f"{safe_topic}_script"
        video_output_dir = os.getenv("VIDEO_OUTPUT_DIR", "./output/final")
        raw_video_filename = f"{base_filename}_raw.mp4"
        final_video_filename = f"{base_filename}.mp4"

        script_data["background_music_name"] = selected_bg
        script_data["raw_video"] = os.path.join(video_output_dir, raw_video_filename)
        script_data["final_video"] = os.path.join(video_output_dir, final_video_filename)



        return script_data


    except Exception as e:
        logger.error(f"Unhandled error: {e}")
def select_voice(script_text):
    """
    Selects the most appropriate voice from the VOICES dictionary based on the complete script.

    Args:
        script_text (str): The complete narration text of the script.

    Returns:
        str: The name of the selected voice.
    """
    # Prepare the voice options information
    voice_options = "\n".join([
        f"- **{voice_name}**: {voice_info['description']}"
        for voice_name, voice_info in VOICES.items()
    ])

    prompt = f"""
Given the following script narration:

\"\"\"
{script_text}
\"\"\"

And the following list of available voices:

{voice_options}

Please analyze the script and select the most appropriate voice for the narration from the list above.

**Instructions:**
- Choose the voice that best matches the script's content and tone.
- Provide only the name of the selected voice without any additional text.
    """

    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are ChatGPT, an assistant that selects the most appropriate narration voice based on script content and provided voice options."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0  # Ensures consistent and deterministic output
        )

        selected_voice = response.choices[0].message['content'].strip()
        logger.debug(f"Selected voice by GPT: {selected_voice}")

        # Validate the selected voice
        if selected_voice in VOICES:
            return selected_voice
        else:
            logger.warning(f"GPT selected an unknown voice: '{selected_voice}'. Defaulting to 'Frederick Surrey'.")
            return "Frederick Surrey"

    except OpenAIError as e:
        logger.error(f"OpenAI API error during voice selection: {e}")
        # Default voice in case of error
        return "Frederick Surrey"
    except Exception as e:
        logger.error(f"An unexpected error occurred during voice selection: {e}")
        # Default voice in case of error
        return "Frederick Surrey"

def select_style(script_text):
    """
    Selects the most appropriate style by sending the script back to GPT along with the style list.
    Args:
        script_text (str): The entire script narration.
    Returns:
        tuple: Selected style name and its corresponding model info.
    """
    # Construct the styles description
    styles_description = "\n".join([
        f"- **{style_name}**: {info['description']} Keywords: {', '.join(info['keywords'])}"
        for style_name, info in MODELS.items()
    ])

    prompt = f"""
Given the following video script narration:

\"\"\"
{script_text}
\"\"\"

And the following list of styles:

{styles_description}

Please analyze the script narration and select the most appropriate style from the list above that best matches the content and tone of the script. Provide only the name of the selected style.
    """

    # Call the OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0.3  # Lower temperature for consistency
        )

        selected_style = response.choices[0].message['content'].strip()
        logger.debug(f"Selected style from GPT:\n{selected_style}")

        # Validate the selected style
        if selected_style in MODELS:
            model_info = MODELS[selected_style]
            return selected_style, model_info
        else:
            logger.error(f"GPT returned an unknown style: {selected_style}. Selecting a random style.")
            selected_style = random.choice(list(MODELS.keys()))
            model_info = MODELS[selected_style]
            return selected_style, model_info

    except OpenAIError as e:
        logger.error(f"OpenAI API error during style selection: {e}")
        # In case of error, select a random style
        selected_style = random.choice(list(MODELS.keys()))
        model_info = MODELS[selected_style]
        return selected_style, model_info

    except Exception as e:
        logger.error(f"An unexpected error occurred during style selection: {e}")
        selected_style = random.choice(list(MODELS.keys()))
        model_info = MODELS[selected_style]
        return selected_style, model_info

def update_visual_prompts(script_data, style_info):
    """
    Updates the visual prompts in the script data based on the selected style.
    Args:
        script_data (dict): The script data containing sections and segments.
        style_info (dict): The selected style information from MODELS.
    """
    style_description = style_info["description"]
    example_prompts = "\n".join(style_info["example_prompts"])

    for section in script_data.get("sections", []):
        # For sections with segments
        if "segments" in section:
            for segment in section["segments"]:
                narration_text = segment["narration"].get("text", "")
                if not narration_text:
                    continue
                # Generate new visual prompt
                prompt = f"""
Given the following narration text:

\"\"\"
{narration_text}
\"\"\"

And the following style description:

{style_description}

With these example prompts:

{example_prompts}

Generate a detailed visual prompt that complements the narration and adheres to the style guidelines.

Provide only the visual prompt text without any additional explanations.
                """

                try:
                    response = openai.ChatCompletion.create(
                        model=OPENAI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.9
                    )
                    visual_prompt = response.choices[0].message['content'].strip()
                    logger.debug(f"Generated visual prompt:\n{visual_prompt}")
                    segment["visual"]["prompt"] = visual_prompt
                except OpenAIError as e:
                    logger.error(f"OpenAI API error during visual prompt generation: {e}")
                except Exception as e:
                    logger.error(f"An unexpected error occurred during visual prompt generation: {e}")
        else:
            # For sections without segments (short videos)
            narration_text = section.get("narration", {}).get("text", "")
            if not narration_text:
                continue
            # Generate new visual prompt
            prompt = f"""
Given the following narration text:

\"\"\"
{narration_text}
\"\"\"

And the following style description:

{style_description}

With these example prompts:

{example_prompts}

Generate a detailed visual prompt that complements the narration and adheres to the style guidelines.

Provide only the visual prompt text without any additional explanations.
            """

            try:
                response = openai.ChatCompletion.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.9
                )
                visual_prompt = response.choices[0].message['content'].strip()
                logger.debug(f"Generated visual prompt:\n{visual_prompt}")
                section["visual"]["prompt"] = visual_prompt
            except OpenAIError as e:
                logger.error(f"OpenAI API error during visual prompt generation: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred during visual prompt generation: {e}")

def save_script(script_data, tone, style, topic, filename=None):
    """
    Saves the generated script, tone, and style to a JSON file.
    Args:
        script_data (dict): The generated script data.
        tone (str): The name of the selected voice.
        style (str): The selected style name (optional).
        topic (str): The topic of the video.
        filename (str, optional): The filename for the saved script. Defaults to None.
    Returns:
        str: The path to the saved script file.
    """
    if not filename:
        # Replace any invalid filename characters
        safe_topic = re.sub(r'[\\/*?:"<>|]', "", topic)
        filename = f"{safe_topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    script_path = os.path.join(VIDEO_SCRIPTS_DIR, filename)

    # Ensure the directory exists
    os.makedirs(VIDEO_SCRIPTS_DIR, exist_ok=True)

    # Add tone and style to the script data
    script_data["tone"] = tone
    if style:
        script_data["settings"]["image_generation_style"] = style
        script_data["settings"]["style_selection_reason"] = f"The {style} style was selected based on the script content."

    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            json.dump(script_data, f, indent=4)
        logger.info(f"Script, tone, and style saved to {script_path}")
        return script_path
    except Exception as e:
        logger.error(f"An error occurred while saving the script: {e}")
        return None

def select_voice_and_style(script_text):
    """
    Selects the most appropriate voice and style based on the complete script.

    Args:
        script_text (str): The complete narration text of the script.

    Returns:
        tuple: Selected voice name, selected style name, and style info.
    """
    selected_voice = select_voice(script_text)
    selected_style, style_info = select_style(script_text)
    return selected_voice, selected_style, style_info

def main():
    """
    Main function to generate, select voice and style, update visuals, and save a video script.
    """
    try:
        # Gather user inputs
        topic = input("Enter the topic of the video: ").strip()
        length_input = input("Enter the length of the video in seconds: ").strip()
        size = input("Enter the size of the video (e.g., 1080x1920): ").strip()
        num_sections_input = input("Enter the number of sections: ").strip()
        num_segments_input = input("Enter the number of segments per section: ").strip()

        # Validate numeric inputs
        try:
            length = int(length_input)
            num_sections = int(num_sections_input)
            num_segments = int(num_segments_input)
        except ValueError:
            logger.error("Invalid input. Length, number of sections, and number of segments must be integers.")
            print("Invalid input. Please ensure that length, number of sections, and number of segments are numbers.")
            return

        # Adjust MAX_SCRIPT_TOKENS based on video length
        global MAX_SCRIPT_TOKENS
        MAX_SCRIPT_TOKENS = calculate_max_tokens(length)
        logger.debug(f"Adjusted MAX_SCRIPT_TOKENS based on video length: {MAX_SCRIPT_TOKENS}")

        # Generate the video script
        script_data = generate_video_script(topic, length, size, num_sections, num_segments)

        if not script_data:
            print("Failed to generate the script. Please check the logs for more details.")
            return

        # Combine all narration texts for voice and style selection
        narration_texts = []
        for section in script_data.get("sections", []):
            if "segments" in section:
                for segment in section["segments"]:
                    narration_text = segment.get("narration", {}).get("text", "")
                    if narration_text:
                        narration_texts.append(narration_text)
            else:
                narration_text = section.get("narration", {}).get("text", "")
                if narration_text:
                    narration_texts.append(narration_text)

        combined_narration = " ".join(narration_texts)
        logger.debug(f"Combined narration text for voice and style selection:\n{combined_narration}")

        # Select the appropriate voice and style based on the combined narration
        selected_voice, selected_style, style_info = select_voice_and_style(combined_narration)
        logger.info(f"Selected voice: {selected_voice}")
        logger.info(f"Selected style: {selected_style}")

        # Update the visual prompts based on the selected style
        update_visual_prompts(script_data, style_info)

        # Save the script with tone (voice) and style information
        saved_path = save_script(script_data, selected_voice, selected_style, topic)

        if saved_path:
            print(f"Script generation, voice and style selection, and saving completed successfully.\nSaved to: {saved_path}")
        else:
            print("Script generated, but failed to save the file.")

    except Exception as e:
        logger.error(f"An unexpected error occurred in the main function: {e}")
        print("An unexpected error occurred. Please check the logs for more details.")

if __name__ == '__main__':
    main()