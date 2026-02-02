#!/usr/bin/env python3
"""
book_json_generator.py

Turn Project Gutenberg HTML-in-ZIP books into mod2 "assembler JSON" files,
sized for ~60-second vertical videos, using a local OpenAI-compatible LLM
(e.g. Ollama running qwen2.5:14b).

Key features:
- Excerpt chunking sized for qwen14b: default 1200–1800 words per excerpt
- Built-in estimator: --estimate (no LLM calls; works even if openai not installed)
- Robust JSON extraction (handles extra text around JSON)
- Auto-retry + repair when model returns wrong segment count
- Guarantees EXACTLY 4 segments per episode (trim/pad + optional repair call)
- Outputs schema matches mod2 assemble.py output so the rest of mod2 works unchanged

Typical (Ollama):
  export OPENAI_BASE_URL="http://192.168.1.176:11434/v1"
  export OPENAI_API_KEY="ollama"
  python3 book_json_generator.py --book-zip ../pg84-h.zip --estimate
  python3 book_json_generator.py --book-zip ../pg84-h.zip --episodes 25 --model qwen2.5:14b

Outputs:
  ./ready/<bookslug>_ep###_assembler.json
"""
from __future__ import annotations

import argparse
import html as htmlmod
import json
import logging
import os
import random
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

call_llm = None  # set in main() to keep --estimate working without openai installed

BASE_DIR = Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Local mod2 imports
from config import load_config, ensure_dirs  # noqa: E402


# -------------------------
# Gutenberg extraction utils
# -------------------------
def _read_first_html_from_zip(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        html_names = [n for n in z.namelist() if n.lower().endswith((".html", ".htm"))]
        if not html_names:
            raise FileNotFoundError("No .html/.htm found in zip")

        # Prefer *-images.html when present (common in Gutenberg)
        html_names.sort(key=lambda n: (0 if "images" in n.lower() else 1, len(n)))
        name = html_names[0]
        return z.read(name).decode("utf-8", errors="ignore")


def _strip_html_to_text(raw_html: str) -> str:
    # Remove scripts/styles
    s = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)

    # Replace line-ish tags with newlines
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p\s*>", "\n\n", s)
    s = re.sub(r"(?i)</div\s*>", "\n", s)
    s = re.sub(r"(?i)</h[1-6]\s*>", "\n\n", s)

    # Drop remaining tags
    s = re.sub(r"(?s)<[^>]+>", " ", s)

    # Unescape entities and normalize whitespace
    s = htmlmod.unescape(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def _remove_gutenberg_boilerplate(text: str) -> str:
    """
    Remove Project Gutenberg header/footer blocks if present.
    """
    # Normalize for matching
    upper = text.upper()
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF PROJECT GUTENBERG EBOOK",
    ]

    start_idx = None
    for m in start_markers:
        idx = upper.find(m)
        if idx != -1:
            start_idx = idx
            break

    end_idx = None
    for m in end_markers:
        idx = upper.find(m)
        if idx != -1:
            end_idx = idx
            break

    if start_idx is not None:
        text = text[start_idx:]
        # Remove the marker line itself
        text = re.sub(r"(?is)\\*\\*\\*\\s*START[\\s\\S]*?\\n", "", text, count=1)

    if end_idx is not None:
        # Find in current text to avoid mismatch with earlier slicing
        upper2 = text.upper()
        cut = None
        for m in end_markers:
            idx = upper2.find(m)
            if idx != -1:
                cut = idx
                break
        if cut is not None:
            text = text[:cut]

    return text.strip()


def _clean_title_author(title: str, author: str) -> Tuple[str, str]:
    t = (title or "").strip()
    a = (author or "").strip()
    t = re.sub(r"(?i)project gutenberg", "", t).strip(" ,.-")
    t = re.sub(r"(?i)the project gutenberg ebook of", "", t).strip(" ,.-")
    t = re.sub(r"(?i)project gutenberg ebook of", "", t).strip(" ,.-")
    t = re.sub(r"(?i)ebook of", "", t).strip(" ,.-")
    if a.lower().startswith("by "):
        a = a[3:].strip()
    return t, a


def _extract_title_author(first_lines: List[str], fallback_title: str) -> Tuple[str, str]:
    lines = [ln.strip() for ln in first_lines if ln.strip()]
    if not lines:
        return _clean_title_author(fallback_title, "")

    joined = " ".join(lines[:3])
    m = re.search(r"(?i)project gutenberg ebook of\s+(.+?)(?:,\s*by\s+(.+))?$", joined)
    if m:
        title = m.group(1) or fallback_title
        author = m.group(2) or ""
        return _clean_title_author(title, author)

    title = lines[0]
    author = ""
    if len(lines) > 1:
        if re.match(r"(?i)^by\\s+", lines[1]):
            author = lines[1]
        elif re.match(r"(?i)^by\\s+", title):
            author = title
            title = lines[1] if len(lines) > 1 else fallback_title

    return _clean_title_author(title, author)


def _remove_front_back_matter(paras: List[str]) -> List[str]:
    """
    Drop likely front/back matter: contents, index, illustrations, contributor/license blocks.
    """
    drop_block = False
    cleaned: List[str] = []
    drop_headings = {
        "contents",
        "table of contents",
        "index",
        "illustrations",
        "list of illustrations",
        "contributors",
        "contributions",
        "bibliography",
        "notes",
        "appendix",
        "acknowledgments",
        "acknowledgements",
        "copyright",
        "project gutenberg",
        "preface",
        "foreword",
        "introduction",
        "translator's note",
        "editor's note",
    }

    for p in paras:
        low = p.strip().lower()
        # Start dropping when we hit obvious headings
        if low in drop_headings:
            drop_block = True
            continue
        # End drop block when we hit a likely chapter heading
        if drop_block and re.match(r"^(chapter|book|part|prologue)\\b", low):
            drop_block = False

        if drop_block:
            continue

        # Drop dot-leader table of contents lines or chapter lists
        if re.search(r"\\.{3,}\\s*\\d+$", p.strip()):
            continue
        if re.match(r"^(chapter|book|part)\\s+[ivxlcdm0-9]+", low) and len(p.split()) <= 6:
            # Likely TOC entry, not body
            continue

        cleaned.append(p)

    # Trim obvious Gutenberg license tail if present
    tail_markers = [
        "end of project gutenberg",
        "end of the project gutenberg",
        "project gutenberg license",
    ]
    for i, p in enumerate(cleaned):
        low = p.lower()
        if any(m in low for m in tail_markers):
            return cleaned[:i]

    return cleaned


def _split_into_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    cleaned: List[str] = []
    for p in paras:
        # Drop very short paras (noise/page headers)
        if len(p.split()) < 6:
            continue
        cleaned.append(p)
    return cleaned


def _make_excerpts(paras: List[str], episodes: int, min_words: int = 1200, max_words: int = 1800) -> List[str]:
    """
    Build rolling excerpt windows between min_words..max_words words, each excerpt -> 1 episode.
    """
    excerpts: List[str] = []
    i = 0
    n = len(paras)

    while i < n and len(excerpts) < episodes:
        words = 0
        chunk: List[str] = []

        while i < n and words < max_words:
            p = paras[i]
            w = len(p.split())

            # If we'd exceed max_words and we already have enough, stop chunk
            if words + w > max_words and words >= min_words:
                break

            chunk.append(p)
            words += w
            i += 1

        if words < min_words:
            # Not enough words to make a solid chunk; stop
            break

        excerpts.append("\n\n".join(chunk))

    return excerpts


# -------------------------
# LLM response utils
# -------------------------
def _extract_json_obj(s: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from a string, allowing extra text around it.
    """
    # common: model prints leading commentary then JSON
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        raise ValueError("LLM response contained no JSON object")
    raw = m.group(0).strip()

    # First attempt
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Some models double-escape JSON or add trailing garbage.
        # Try to clean a bit:
        raw2 = raw.strip("` \n\t")
        # Remove trailing characters after last }
        last = raw2.rfind("}")
        if last != -1:
            raw2 = raw2[: last + 1]
        return json.loads(raw2)


def _normalize_segments(payload: Dict[str, Any], fallback_visual: str | None = None) -> List[Dict[str, str]]:
    raw = payload.get("segments")
    segs: List[Dict[str, str]] = []

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    else:
        items = []

    for it in items:
        if not isinstance(it, dict):
            continue
        narration = str(it.get("narration", "")).strip()
        visual_prompt = str(it.get("visual_prompt", "")).strip()
        if not visual_prompt and fallback_visual:
            visual_prompt = fallback_visual
        if narration:
            segs.append({"narration": narration, "visual_prompt": visual_prompt})

    return segs


def _count_words(s: str) -> int:
    return len([w for w in s.strip().split() if w])


def _enforce_word_targets(segs: List[Dict[str, str]], total_range: Tuple[int, int] = (115, 145)) -> List[Dict[str, str]]:
    """
    Light-touch enforcement:
    - Trim segments that are too long
    - If too short overall, we leave it (the model retries usually fix it)
    """
    lo, hi = total_range
    # Trim individual segments to ~40 words max (keep meaning)
    for seg in segs:
        words = seg["narration"].split()
        if len(words) > 42:
            seg["narration"] = " ".join(words[:42]).strip()

    total = sum(_count_words(s["narration"]) for s in segs)
    if total > hi:
        # Reduce from the longest segments until <= hi
        while total > hi:
            idx = max(range(len(segs)), key=lambda i: _count_words(segs[i]["narration"]))
            words = segs[idx]["narration"].split()
            if len(words) <= 26:
                break
            segs[idx]["narration"] = " ".join(words[:-3]).strip()
            total = sum(_count_words(s["narration"]) for s in segs)

    # If total < lo, we don't fabricate too much here; retries/repair call handles it better.
    return segs


def _ollama_friendly_messages(book_title: str, book_author: str, excerpt: str) -> List[Dict[str, str]]:
    system = (
        "You are a script adapter for short vertical videos.\n"
        "You will be given an excerpt from a public-domain book.\n"
        "Produce ONLY valid JSON, no markdown, no commentary.\n"
        "Schema:\n"
        "{\n"
        '  "title": string,\n'
        '  "segments": [\n'
        '    {"narration": string, "visual_prompt": string},\n'
        "    ... exactly 4 items total ...\n"
        "  ]\n"
        "}\n"
        "Rules (MANDATORY):\n"
        "- You MUST output exactly 4 segments. If you output fewer or more, the answer is invalid.\n"
        "- Total narration length: 115 to 145 words across all 4 segments.\n"
        "- Each segment narration: 25 to 40 words.\n"
        "- Narration must be modern, engaging, and faithful to the excerpt (no invented plot points).\n"
        "- Visual prompts: cinematic, descriptive, no text overlays, no logos, no brand names.\n"
        "- Avoid spoilers beyond the excerpt.\n"
        "- Title should be punchy and under 70 characters.\n"
    )
    byline = f"{book_title} by {book_author}".strip() if book_author else book_title
    user = (
        f"BOOK: {byline}\n\n"
        "EXCERPT:\n"
        f"{excerpt}\n\n"
        "Return ONLY JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _theme_messages(book_title: str) -> List[Dict[str, str]]:
    system = (
        "You are a creative director. Given a book title, generate a short visual theme "
        "for images. Return ONLY JSON, no commentary.\n"
        "Schema:\n"
        '{ "image_theme": "string (12-30 words)" }'
    )
    user = f"BOOK TITLE: {book_title}\nReturn ONLY JSON."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _safe_llm_generate_theme(model: str, book_title: str) -> str:
    try:
        if call_llm is None:
            raise RuntimeError("LLM client not initialized. Call main() or set call_llm first.")
        content = call_llm(
            messages=_theme_messages(book_title),
            model=model,
            temperature=0.5,
            max_completion_tokens=120,
            timeout=45,
        )
        payload = _extract_json_obj(content)
        theme = str(payload.get("image_theme", "")).strip()
        return theme
    except Exception as exc:
        logging.warning("Theme generation failed, continuing without theme: %s", exc)
        return ""


# -------------------------
# Assembler JSON builder (mod2 schema)
# -------------------------
def _duration_for_text(text: str) -> int:
    # Keep consistent with mod2 assemble.py heuristic: ~2 words/sec, clamp 5..15
    words = text.split()
    return min(15, max(5, len(words) // 2))


def _to_assembler(config, book_title: str, book_author: str, payload: Dict[str, Any], seo: Optional[Dict[str, Any]] = None, image_theme: str = "") -> Dict[str, Any]:
    fallback_visual = ""
    if image_theme:
        fallback_visual = f"{image_theme}. cinematic scene, high detail, dramatic lighting, no text."
    clean_segments = _normalize_segments(payload, fallback_visual=fallback_visual)
    clean_segments = _enforce_word_targets(clean_segments)

    # Guarantee exactly 4 segments (trim/pad)
    if len(clean_segments) > 4:
        clean_segments = clean_segments[:4]
    while len(clean_segments) < 4:
        # Pad with safe continuation segment
        clean_segments.append(
            {
                "narration": "The moment hangs in the air, and you can feel the stakes rising as the scene shifts.",
                "visual_prompt": fallback_visual
                or "cinematic continuation scene, moody lighting, shallow depth of field, dramatic atmosphere, high detail",
            }
        )
    clean_segments = clean_segments[:4]

    sections: List[Dict[str, Any]] = []
    heading_title = f"{book_title} / {book_author}".strip(" /") if book_author else book_title
    for idx, seg in enumerate(clean_segments, start=1):
        narration = seg["narration"].strip()
        visual_prompt = seg.get("visual_prompt", "").strip() or fallback_visual
        duration = _duration_for_text(narration)

        sections.append(
            {
                "section_number": idx,
                "original_name": book_title,
                "title": heading_title,
                "section_duration": duration,
                "segments": [
                    {
                        "segment_number": 1,
                        "narration": {
                            "text": narration,
                            "start": 0,
                            "duration": duration,
                            "audio_path": f"audio/section_{idx}_segment_1.mp3",
                        },
                        "visual": {
                            "type": "image",
                            "prompt": visual_prompt,
                            "start": 0,
                            "duration": duration,
                            "image_path": f"visuals/section_{idx}_segment_1.png",
                        },
                        "sound": {"transition_effect": "fade_in"},
                    }
                ],
            }
        )

    narration_full = "\n\n".join(s["segments"][0]["narration"]["text"] for s in sections)
    seo = seo or {}

    return {
        "settings": {
            "video_size": config.video.size,
            "use_transitions": config.video.use_transitions,
            "use_background_music": config.video.use_background_music,
            "background_music_type": config.video.bg_music_tag,
            "image_generation_style": config.channel.image_generation_style,
            "style_selection_reason": "Book excerpt adaptation.",
            "bg_music_volume": config.video.bg_music_volume,
            "transition_volume": config.video.transition_volume,
            "image_theme": image_theme,
        },
        "sections": sections,
        "social_media": {
            "title": heading_title,
            "description": str(seo.get("description", "")).strip(),
            "tags": seo.get("tags") if seo.get("tags") else ["#booktok", "#classicbooks", "#audiobook", "#storytime"],
        },
        "background_music_type": config.video.bg_music_tag,
        "background_music": config.video.bg_music_tag,
        "tone": "Valentino",
        "image_style": config.channel.image_generation_style,
        "reference": f"{book_title} by {book_author}".strip() if book_author else book_title,
        "narration_full": narration_full,
        "image_theme": image_theme,
    }


# -------------------------
# LLM call + repair logic
# -------------------------
def _safe_llm_generate_payload(
    model: str,
    messages: List[Dict[str, str]],
    max_attempts: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Call model and ensure we get valid JSON with exactly 4 segments.
    Retries with stronger instruction if needed.
    """
    content: Optional[str] = None
    last_err: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            if call_llm is None:
                raise RuntimeError("LLM client not initialized. Call main() or set call_llm first.")
            content = call_llm(
                messages=messages,
                model=model,
                temperature=0.65,
                max_completion_tokens=900,
                timeout=60,
            )
            payload = _extract_json_obj(content)
            segs = _normalize_segments(payload)

            if len(segs) == 4:
                return payload

            logging.warning("Attempt %s: model returned %s segments (need 4).", attempt, len(segs))

            # Retry with correction prompt
            messages = messages + [
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was invalid.\n"
                        "Return ONLY valid JSON and include EXACTLY 4 segments.\n"
                        "Total narration 115–145 words; 25–40 words per segment.\n"
                        "No extra text."
                    ),
                }
            ]
        except Exception as exc:
            last_err = exc
            logging.warning("Attempt %s: LLM call/extract failed: %s", attempt, exc)

    if last_err:
        logging.error("Failed to get valid payload after %s attempts: %s", max_attempts, last_err)
    else:
        logging.error("Failed to get valid payload after %s attempts.", max_attempts)
    if content:
        logging.error("Last model output (truncated): %s", content[:500].replace("\n", " "))
    return None


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mod2 assembler JSONs from a Gutenberg book zip.")
    parser.add_argument("--book-zip", required=True, help="Path to Project Gutenberg book zip (html).")
    parser.add_argument("--episodes", type=int, default=10, help="How many 60s JSONs to generate (max; may stop earlier).")
    parser.add_argument("--estimate", action="store_true", help="Estimate how many episodes the book can produce (no LLM calls).")
    parser.add_argument("--chunk-min", type=int, default=1200, help="Min words per excerpt chunk (estimator + generation).")
    parser.add_argument("--chunk-max", type=int, default=1800, help="Max words per excerpt chunk (estimator + generation).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed (used only for naming/shuffle).")
    parser.add_argument("--model", default=None, help="Model name for OpenAI-compatible server (e.g. qwen2.5:14b).")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle paragraphs before chunking (less coherent; more variety).")
    parser.add_argument("--out-dir", default=None, help="Output directory (defaults to config.paths.ready_dir).")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max LLM attempts per episode to enforce 4 segments.")
    args = parser.parse_args()

    if args.chunk_min < 200 or args.chunk_max < args.chunk_min:
        raise SystemExit("--chunk-min must be >= 200 and --chunk-max must be >= --chunk-min")

    config = load_config()
    ensure_dirs(config)
    out_dir = Path(args.out_dir) if args.out_dir else config.paths.ready_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(args.book_zip).expanduser().resolve()
    raw_html = _read_first_html_from_zip(zip_path)
    text = _strip_html_to_text(raw_html)
    text = _remove_gutenberg_boilerplate(text)

    # Title/author guess: first non-empty lines after cleanup
    first_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    book_title, book_author = _extract_title_author(first_lines, zip_path.stem)

    paras = _split_into_paragraphs(text)
    paras = _remove_front_back_matter(paras)
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(paras)

    total_words = sum(len(p.split()) for p in paras)
    approx_chunk = max(1, (args.chunk_min + args.chunk_max) // 2)
    approx_episodes = max(0, total_words // approx_chunk)

    # Simulate chunking once (no LLM)
    simulated = _make_excerpts(paras, episodes=10**9, min_words=args.chunk_min, max_words=args.chunk_max)
    sim_count = len(simulated)

    if args.estimate:
        print("=== Episode estimate ===")
        print(f"Book zip: {zip_path.name}")
        print(f"Detected title: {book_title}")
        if book_author:
            print(f"Detected author: {book_author}")
        print(f"Paragraphs kept: {len(paras)}")
        print(f"Total words (cleaned): {total_words:,}")
        print(f"Chunk range: {args.chunk_min}–{args.chunk_max} words (target ~{approx_chunk})")
        print(f"Estimated episodes (rough): {approx_episodes}")
        print(f"Episodes possible (simulated chunking): {sim_count}")
        if sim_count > 0:
            sizes = [len(ex.split()) for ex in simulated[:5]]
            tail = [len(ex.split()) for ex in simulated[-5:]] if sim_count > 5 else []
            print(f"Example chunk sizes (first): {sizes}")
            if tail:
                print(f"Example chunk sizes (last): {tail}")
        print("\nTip: generate with --episodes N (N <= simulated count).")
        return

    # Build excerpts for actual generation
    excerpts = _make_excerpts(paras, episodes=args.episodes, min_words=args.chunk_min, max_words=args.chunk_max)
    if not excerpts:
        raise SystemExit("Failed to build excerpts (book too short after cleanup).")

    # Lazy import so --estimate works without openai installed
    from openai_client import call_llm as _call_llm  # local helper (OpenAI or Ollama)
    global call_llm
    call_llm = _call_llm
    model = args.model or config.openai.model

    # Generate a consistent visual theme from the book title
    image_theme = _safe_llm_generate_theme(model=model, book_title=book_title)
    if image_theme:
        logging.info("Image theme: %s", image_theme)

    # Generate episodes
    for i, excerpt in enumerate(excerpts, start=1):
        messages = _ollama_friendly_messages(book_title=book_title, book_author=book_author, excerpt=excerpt)

        logging.info(
            "Episode %s/%s: calling model=%s (excerpt words=%s)",
            i,
            len(excerpts),
            model,
            len(excerpt.split()),
        )

        payload = _safe_llm_generate_payload(
            model=model,
            messages=messages,
            max_attempts=args.max_attempts,
        )

        if not payload:
            logging.error("Skipping episode %s due to repeated invalid outputs.", i)
            continue

        # Assemble (always enforces exactly 4 segments with trim/pad as final safety)
        assembler = _to_assembler(config, book_title, book_author, payload, image_theme=image_theme)

        bookslug = re.sub(r"[^a-zA-Z0-9]+", "_", zip_path.stem).strip("_").lower()
        out_path = out_dir / f"{bookslug}_ep{i:03d}_assembler.json"
        out_path.write_text(json.dumps(assembler, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Wrote %s", out_path)

    logging.info("Done. Produce a video with:")
    logging.info("  python3 app.py --script %s", str(next(out_dir.glob(f"{bookslug}_ep*_assembler.json"), None)))


if __name__ == "__main__":
    main()
