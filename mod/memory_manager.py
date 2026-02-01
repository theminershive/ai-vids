#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from config import MemoryConfig


@dataclass
class MemoryStore:
    path: Path
    max_items: int
    items: List[str] = field(default_factory=list)

    def load(self) -> None:
        if not self.path.exists():
            self.items = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.items = [str(item).strip() for item in data if str(item).strip()]
            else:
                self.items = []
        except json.JSONDecodeError:
            logging.warning("Failed to parse %s; starting fresh.", self.path)
            self.items = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items[-self.max_items :], indent=2), encoding="utf-8")

    def add(self, value: str) -> None:
        cleaned = value.strip()
        if not cleaned:
            return
        self.items = [item for item in self.items if item != cleaned]
        self.items.append(cleaned)
        if len(self.items) > self.max_items:
            self.items = self.items[-self.max_items :]
        self.save()


@dataclass
class MemoryManager:
    topics: MemoryStore
    styles: MemoryStore
    hooks: MemoryStore

    @classmethod
    def build(cls, memory: MemoryConfig) -> "MemoryManager":
        manager = cls(
            topics=MemoryStore(memory.topics_file, memory.history_depth),
            styles=MemoryStore(memory.styles_file, memory.history_depth),
            hooks=MemoryStore(memory.hooks_file, memory.history_depth),
        )
        manager.load_all()
        return manager

    def load_all(self) -> None:
        self.topics.load()
        self.styles.load()
        self.hooks.load()

    def remember(self, topic: str, style: str, hook: str) -> None:
        self.topics.add(topic)
        self.styles.add(style)
        self.hooks.add(hook)
