#!/usr/bin/env python3
"""
Process ready/*.json in order and generate new JSONs from unseen book zips.

Behavior:
- If there is an unprocessed JSON in ready/, render exactly one per run.
- If none, look for an unprocessed .zip (recursively) and generate all episodes,
  then render the first new JSON.
- Tracks progress in memory/processed_ready.json and memory/processed_books.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable
import re

from config import load_config, ensure_dirs
from render_ready import _run_pipeline
import book_json_generator as bjg

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).parent.resolve()


def _load_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [str(x) for x in data] if isinstance(data, list) else []
    except Exception:
        return []


def _save_list(path: Path, items: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(set(items)), indent=2), encoding="utf-8")


def _find_zip_candidates(root: Path) -> list[Path]:
    return sorted(root.rglob("*.zip"))

def _episode_sort_key(path: Path) -> tuple[int, str]:
    name = path.stem.lower()
    m = re.search(r"_ep(\\d+)_", name)
    if m:
        return (int(m.group(1)), name)
    m = re.search(r"(\\d+)", name)
    if m:
        return (int(m.group(1)), name)
    return (0, name)


def _estimate_episodes(zip_path: Path, chunk_min: int, chunk_max: int) -> int:
    raw_html = bjg._read_first_html_from_zip(zip_path)
    text = bjg._strip_html_to_text(raw_html)
    paras = bjg._split_into_paragraphs(text)
    excerpts = bjg._make_excerpts(paras, episodes=10**9, min_words=chunk_min, max_words=chunk_max)
    return len(excerpts)


def _generate_from_zip(zip_path: Path, out_dir: Path, episodes: int, chunk_min: int, chunk_max: int) -> None:
    args = [
        "python3",
        str(BASE_DIR / "book_json_generator.py"),
        "--book-zip",
        str(zip_path),
        "--episodes",
        str(episodes),
        "--chunk-min",
        str(chunk_min),
        "--chunk-max",
        str(chunk_max),
        "--out-dir",
        str(out_dir),
    ]
    logging.info("Generating %s episodes from %s", episodes, zip_path.name)
    import subprocess
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process ready JSONs and generate from new zips.")
    parser.add_argument("--chunk-min", type=int, default=1200)
    parser.add_argument("--chunk-max", type=int, default=1800)
    parser.add_argument("--zip-root", default=str(BASE_DIR), help="Root to scan for book zips.")
    args = parser.parse_args()

    config = load_config()
    ensure_dirs(config)

    processed_ready_path = config.paths.memory_dir / "processed_ready.json"
    processed_books_path = config.paths.memory_dir / "processed_books.json"

    processed_ready = set(_load_list(processed_ready_path))
    processed_books = set(_load_list(processed_books_path))

    ready_dir = config.paths.ready_dir
    ready_dir.mkdir(parents=True, exist_ok=True)

    pending = [p for p in sorted(ready_dir.glob("*.json"), key=_episode_sort_key) if str(p.resolve()) not in processed_ready]
    if pending:
        next_json = pending[0]
        logging.info("Rendering pending JSON: %s", next_json.name)
        _run_pipeline(next_json, config)
        processed_ready.add(str(next_json.resolve()))
        _save_list(processed_ready_path, processed_ready)
        return

    zip_root = Path(args.zip_root).resolve()
    zip_candidates = _find_zip_candidates(zip_root)
    new_zips = [z for z in zip_candidates if str(z.resolve()) not in processed_books]
    if not new_zips:
        logging.info("No pending JSONs and no new zips found.")
        return

    zip_path = new_zips[0]
    episodes = _estimate_episodes(zip_path, args.chunk_min, args.chunk_max)
    if episodes <= 0:
        logging.warning("No episodes could be generated for %s", zip_path.name)
        processed_books.add(str(zip_path.resolve()))
        _save_list(processed_books_path, processed_books)
        return

    _generate_from_zip(zip_path, ready_dir, episodes, args.chunk_min, args.chunk_max)
    processed_books.add(str(zip_path.resolve()))
    _save_list(processed_books_path, processed_books)

    pending = [p for p in sorted(ready_dir.glob("*.json"), key=_episode_sort_key) if str(p.resolve()) not in processed_ready]
    if not pending:
        logging.warning("No JSONs were generated for %s", zip_path.name)
        return

    next_json = pending[0]
    logging.info("Rendering generated JSON: %s", next_json.name)
    _run_pipeline(next_json, config)
    processed_ready.add(str(next_json.resolve()))
    _save_list(processed_ready_path, processed_ready)


if __name__ == "__main__":
    main()
