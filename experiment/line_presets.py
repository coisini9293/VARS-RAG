"""Experiment line definitions (L0–L6): compose RAG + prompt + decode settings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from experiment.knowledge_rag import RagSettings


@dataclass(frozen=True)
class LineSpec:
    key: str
    rag: RagSettings
    use_expert_prompt: bool
    use_cot: bool
    max_new_tokens: int
    do_sample: bool
    temperature: float
    n_vote: int


RAG_NONE = RagSettings(False, 0, 0, None, 0)
RAG_BASE = RagSettings(True, 3, 1, 200, 800)
RAG_DEEP = RagSettings(True, 3, 3, None, 2500)

LINE_PRESETS: Dict[str, LineSpec] = {
    "L0": LineSpec("L0", RAG_NONE, False, False, 64, False, 0.2, 1),
    "L1": LineSpec("L1", RAG_BASE, False, False, 64, False, 0.2, 1),
    "L2": LineSpec("L2", RAG_DEEP, False, False, 64, False, 0.2, 1),
    "L3": LineSpec("L3", RAG_NONE, True, False, 64, False, 0.2, 1),
    "L4": LineSpec("L4", RAG_NONE, False, True, 192, False, 0.2, 1),
    "L5": LineSpec("L5", RAG_NONE, False, False, 64, True, 0.7, 5),
    "L6": LineSpec("L6", RAG_DEEP, True, True, 192, False, 0.2, 1),
}
