import os
import json
import re
import logging
from dotenv import load_dotenv
import openai
from narration_and_style import select_voice, select_style, update_visual_prompts, MODELS

# Load environment variables
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ——— Sanitization rules ———
FILTER_KEYWORDS = {
    r"\bchild\b": "figure",
    r"\bchildren\b": "figures",
    r"\bkid\b": "person",
    r"\bminor\b": "individual",
    r"\binfant\b": "figure",
    r"\btoddler\b": "person",
    r"\bbaby\b": "person",
    r"\bteen\b": "young person",
    r"\bteenager\b": "young person",
    r"\byouth\b": "individual",
    r"\bjuvenile\b": "individual",
    r"\bunderage\b": "young individual",
    r"\bchildlike\b": "figurative",
    r"\blittle\s+girl\b": "young person",
    r"\blittle\s+boy\b": "young person",
    r"\bnaked\b": "modest",
    r"\bnakedness\b": "modestness",

    r"\bcelebrity\b": "figure",
    r"\bfamous\b": "well-known",
    r"\bpublic\s+figure\b": "figure",

    r"\bpenis\b": "anatomical part",
    r"\bcock\b": "anatomical part",
    r"\bdick\b": "anatomical part",
    r"\bshaft\b": "anatomical part",
    r"\btesticle\b": "anatomical part",
    r"\btesticles\b": "anatomical parts",
    r"\bscrotum\b": "anatomical part",

    r"\bvagina\b": "anatomical part",
    r"\bvulva\b": "anatomical part",
    r"\blabia\b": "anatomical parts",
    r"\bclitoris\b": "anatomical part",
    r"\bpussy\b": "anatomical part",

    r"\bbreast\b": "anatomical part",
    r"\bbreasts\b": "anatomical parts",
    r"\bboob\b": "anatomical part",
    r"\bboobs\b": "anatomical parts",
    r"\btit\b": "anatomical part",
    r"\btits\b": "anatomical parts",
    r"\bnipple\b": "anatomical part",
    r"\bnipples\b": "anatomical parts",

    r"\bbutt\b": "anatomical part",
    r"\bbuttocks\b": "anatomical parts",
    r"\bass\b": "anatomical part",
    r"\banus\b": "anatomical part",
    r"\basshole\b": "anatomical part",

    r"\bcum\b": "fluid",
    r"\bsemen\b": "fluid",
    r"\bsperm\b": "fluid",
    r"\bmilf\b": "individual",
    r"\borgasm\b": "reaction",
    r"\barousal\b": "reaction",

    r"\bsex\b": "intimacy",
    r"\bsexual\b": "intimate",
    r"\bfuck\b": "act",
    r"\bfucking\b": "act",
    r"\bintercourse\b": "act",
    r"\bpenetration\b": "act",
    r"\bmasturbat(e|ion|ing)\b": "act",
    r"\bjerk\s*off\b": "act",
    r"\bhandjob\b": "act",
    r"\bblowjob\b": "act",
    r"\bsuck\b": "act",
    r"\bgrope\b": "act",
    r"\bstrip(per|ping)\b": "act",
    r"\bbukkake\b": "act",

    r"\bcp\b": "content",
    r"\bchild\s*porn\b": "content",
    r"\bincest\b": "content",
    r"\brape\b": "act",
    r"\bbeastiality\b": "content",
    r"\bzoophilia\b": "content",

    r"\bgore\b": "graphic",
    r"\bblood\b": "liquid",
    r"\bbleeding\b": "graphic",
    r"\bviolent\b": "aggressive",
}


def sanitize_prompt(prompt: str) -> str:
    sanitized = prompt
    for pattern, replacement in FILTER_KEYWORDS.items():
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def generate_social_media(script_data):
    try:
        verses = None
        for section in script_data.get("sections", []):
            for segment in section.get("segments", []):
                verses_list = segment.get("verses", [])
                if verses_list:
                    if len(verses_list) == 1:
                        verses = verses_list[0]
                    else:
                        verses = f"{verses_list[0]} – {verses_list[-1]}"
                    break
            if verses:
                break

        topic_stub = "Today's Bible Verse"
        title = f"{topic_stub} – {verses}" if verses else topic_stub

        social = script_data.get("social_media", {})
        social["title"] = title
        script_data["social_media"] = social
        return social

    except Exception as e:
        logger.error(f"Error updating social media title: {e}")
        return script_data.get("social_media", {"title": "Today's Bible Verse"})


def enrich_script(script_data):
    full_text = " ".join(
        seg.get("narration", {}).get("text", "")
        for sec in script_data.get("sections", [])
        for seg in sec.get("segments", [])
    )
    tone = select_voice(full_text)
    script_data["tone"] = tone

    style_name, style_info = select_style(full_text)
    script_data["settings"]["image_generation_style"] = style_name
    script_data["settings"]["style_selection_reason"] = (
        f"The {style_name} style was selected based on the script content."
    )

    for sec_idx, section in enumerate(script_data.get("sections", []), start=1):
        for seg_idx, seg in enumerate(section.get("segments", []), start=1):
            raw = seg.get("visual_prompt")
            if raw:
                safe = sanitize_prompt(raw)
                if raw != safe:
                    logger.info(f"Sanitized visual prompt (section {sec_idx}, seg {seg_idx}): {safe!r}")
                seg["visual_prompt"] = safe
                logger.debug(f"FINAL sanitized prompt for section {sec_idx}, segment {seg_idx}: {safe}")

    update_visual_prompts(script_data, style_info)

    social = generate_social_media(script_data)

    if "description" in social:
        desc = social["description"]
        hashtags = " #shorts #bible"
        if "#shorts" not in desc or "#bible" not in desc:
            if not desc.endswith(hashtags):
                social["description"] = desc.strip() + hashtags

    if "tags" not in social or not isinstance(social["tags"], list):
        social["tags"] = []

    for tag in ["#shorts", "#bible"]:
        if tag not in social["tags"]:
            social["tags"].append(tag)

    script_data["social_media"] = social

    return script_data
