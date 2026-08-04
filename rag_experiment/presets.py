"""RAG-only experiment lines (R0–R4). Non-RAG techniques are out of scope here."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class RagSettings:
    """How retrieved rule chunks are selected and merged into the user message."""

    enabled: bool
    search_top_k: int
    merge_top_n: int
    truncate_chars: Optional[int]
    max_context_chars: int
    # fixed: same query for all clips; adaptive: include Action class (never Offence label)
    query_mode: str = "fixed"


@dataclass(frozen=True)
class LineSpec:
    key: str
    name: str
    description: str
    rag: RagSettings
    # Longer answers: paper asks foul? + card? with why (explanations ~dozens of words)
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.2


RAG_OFF = RagSettings(False, 0, 0, None, 0, "fixed")
# Matches current Streamlit generate_answer: search top_k=3, use Top1, truncate ~200
RAG_BASE = RagSettings(True, 3, 1, 200, 800, "fixed")
RAG_TOPK = RagSettings(True, 3, 3, 200, 800, "fixed")
# 中文规则 token 密度高；2500 字常使总 prompt >2048 导致空生成。预算压到 ~900。
RAG_DEEP = RagSettings(True, 3, 3, None, 900, "fixed")
RAG_ADAPTIVE = RagSettings(True, 3, 3, None, 900, "adaptive")

LINE_PRESETS: Dict[str, LineSpec] = {
    "R0": LineSpec(
        "R0",
        "Baseline",
        "Original X-VARS without RAG (control).",
        RAG_OFF,
    ),
    "R1": LineSpec(
        "R1",
        "RAG-Base",
        "Production-like RAG: Top1 chunk, ~200-char truncate, fixed query.",
        RAG_BASE,
    ),
    "R2": LineSpec(
        "R2",
        "RAG-TopK",
        "Merge Top3 chunks with per-chunk truncate (200 chars), fixed query.",
        RAG_TOPK,
    ),
    "R3": LineSpec(
        "R3",
        "RAG-Deep",
        "Merge Top3 full chunks, longer context budget, fixed query.",
        RAG_DEEP,
    ),
    "R4": LineSpec(
        "R4",
        "RAG-Adaptive",
        "Same merge as R3, but retrieval query includes Action class (no Offence leak).",
        RAG_ADAPTIVE,
    ),
}

DEFAULT_LINES = "R0,R1,R2,R3,R4"
