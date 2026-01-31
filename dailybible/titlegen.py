# titlegen.py

import os
import json
import re
import logging
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_social_media(script_data):
    """
    Generates a social media post (title, description, tags) based on the script.
    Applies custom hook titles for known verse ranges.
    """
    try:
        prompt = f"""
Based on the following video script JSON, generate a social media post JSON with fields:
- "title": Catchy title for the post.
- "description": Engaging description.
- "tags": List of 5 to 10 relevant hashtags.

Script JSON:
{json.dumps(script_data, indent=4)}
"""
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an assistant that creates social media posts from video scripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        content = response.choices[0].message['content']
        clean = re.sub(r'^```(?:json)?\s*', '', content.strip())
        clean = re.sub(r'\s*```$', '', clean)
        result = json.loads(clean)

        # Override title with known verse-based hooks
        verse_hooks = {
            "Genesis 1:27 – Genesis 1:28": "God’s First Blessing – Genesis 1:28",
            "Genesis 3:7 – Genesis 3:10": "Why Adam Hid From God – Genesis 3:7-10",
            "Genesis 3:14 – Genesis 3:15": "The First Prophecy – Genesis 3:14-15"
        }

        for key, value in verse_hooks.items():
            if key in result.get("title", ""):
                logger.info(f"Hook title applied: '{value}' (replacing '{result['title']}')")
                result["title"] = value
                break

        return result

    except Exception as e:
        logger.error(f"Error generating social media: {e}")
        return {
            "title": "New Video Release!",
            "description": "Check out our latest video now!",
            "tags": ["video", "release", "AI", "shorts"]
        }
