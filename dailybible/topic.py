import openai
import datetime
import os
import json
from dotenv import load_dotenv  # <-- NEW: import dotenv

# --- LOAD .env ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY not found in .env file!")

# --- CONFIG ---
MODEL = "gpt-4-1106-preview"  # or "gpt-4o"

# --- SYSTEM PROMPT FOR IDEAS ---
SYSTEM_PROMPT = """
You are an expert viral video strategist.
Based on the current day of the week, generate FIVE clickable YouTube video title ideas following this daily style guide:

- Monday: Hot Debate Topics
- Tuesday: Hidden Truths
- Wednesday: Top Lists or Best/Worst
- Thursday: What If? or Future Shock
- Friday: Feel-Good or Incredible Stories
- Saturday: Wild Stories or Unbelievable Moments
- Sunday: Crime Stories or Unsolved Mystries

**Rules:**
- Titles must be short, curiosity-driven, emotionally charged
- Use trending/relevant topics when possible
- Use direct questions sometimes
- Use verbs that spark action: Revealed, Exposed, Transformed, Survived, Built, Destroyed
- Make the titles feel natural for YouTube
- Include emojis where fitting (but not forced)
- Add 10–15 hashtags, comma-separated, for each title
- Write 1 short, punchy YouTube description (2–3 sentences) for each title

**Output strictly format for EACH title:**
Title: <viral video title>
Hashtags: <hashtags>
Description: <youtube description>

Stay focused, creative, and keep it YouTube-viral-ready.
"""

# --- DAY MAPPING ---
DAY_STYLE_MAPPING = {
    0: "Hot Debate Topics",
    1: "Hidden Truths",
    2: "Top Lists or Best/Worst",
    3: "What If? or Future Shock",
    4: "Feel-Good or Incredible Stories",
    5: "Wild Stories or Unbelievable Moments",
    6: "Community Questions / Open-Ended Topics"
}

# --- FEASIBILITY SCORING FUNCTION ---
def calculate_feasibility(title: str) -> (int, str):
    title_lower = title.lower()
    difficulty_keywords = ["war", "rescue", "underwater", "space", "natural disaster", "plane crash", "earthquake", "live footage", "historic event"]
    easy_keywords = ["tips", "hacks", "facts", "stories", "list", "apps", "tools", "shortcuts", "secrets", "revealed", "exposed", "tech", "social media", "products"]

    if any(word in title_lower for word in difficulty_keywords):
        score = 2
        reason = "Challenging topic: Real-world footage or difficult scenes needed (e.g., rescue, war, space). Harder to produce with AI."
    elif any(word in title_lower for word in easy_keywords):
        score = 5
        reason = "Simple topic: Stock videos, AI-generated visuals, and basic narration are enough. Easy production."
    else:
        score = 4
        reason = "Moderate topic: Some creative stock or AI visuals needed, but manageable without real-world complex footage."

    return score, reason

# --- PARSING FUNCTION ---
def parse_titles(raw_output):
    titles = []
    current = {"title": "", "hashtags": "", "description": ""}

    for line in raw_output.splitlines():
        if line.startswith("Title:"):
            if current["title"]:
                titles.append(current)
                current = {"title": "", "hashtags": "", "description": ""}
            current["title"] = line.replace("Title:", "").strip()
        elif line.startswith("Hashtags:"):
            current["hashtags"] = line.replace("Hashtags:", "").strip()
        elif line.startswith("Description:"):
            current["description"] = line.replace("Description:", "").strip()

    if current["title"]:
        titles.append(current)

    return titles

# --- GET EXTRA INFO FUNCTION ---
def get_extra_info(title):
    prompt = f"Provide extra helpful information for a YouTube video titled '{title}'. Include interesting facts, context, quick history, or important points that could enrich the script or video. Be punchy and relevant."

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a viral video researcher helping add extra useful info for a video."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return response['choices'][0]['message']['content'].strip()

# --- MAIN FUNCTION ---
def generate_daily_video_idea():
    openai.api_key = OPENAI_API_KEY

    today = datetime.datetime.now().weekday()  # Monday = 0
    day_style = DAY_STYLE_MAPPING.get(today, "Hot Debate Topics")
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")

    user_prompt = f"Today is {datetime.datetime.now().strftime('%A')}. Focus on: {day_style}. Generate five ideas."

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8,
        max_tokens=1500
    )

    output = response['choices'][0]['message']['content']
    print("\n--- Raw Output ---\n")
    print(output)

    # Parse the 5 ideas
    all_titles = parse_titles(output)

    # Score each title
    scored_titles = []
    for idea in all_titles:
        score, reason = calculate_feasibility(idea["title"])
        idea["score"] = score
        idea["reason"] = reason
        scored_titles.append(idea)

    # Select the best (highest score, break ties by first appearance)
    best_title = max(scored_titles, key=lambda x: x["score"])

    print("\n--- Best Selected Idea ---\n")
    print(f"Title: {best_title['title']}")
    print(f"Score: {best_title['score']} - {best_title['reason']}")
    print(f"Hashtags: {best_title['hashtags']}")
    print(f"Description: {best_title['description']}")

    # Get Extra Info
    extra_info = get_extra_info(best_title["title"])

    # Build Final Plan JSON
    final_plan = {
        "title": best_title["title"],
        "feasibility": {
            "score": best_title["score"],
            "details": best_title["reason"]
        },
        "structure": {
            "length": 45,
            "sections": 3,
            "segments_per_section": 2
        },
        "resolution": "1080x1920",
        "hashtags": best_title["hashtags"],
        "description": best_title["description"],
        "extra_info": extra_info
    }

    # Save Plan
    save_path = f"video_plan_{today_str}.json"
    with open(save_path, "w") as f:
        json.dump(final_plan, f, indent=4)

    print(f"\n✅ Saved Final Plan to {save_path}\n")

# --- RUN ---
if __name__ == "__main__":
    generate_daily_video_idea()
