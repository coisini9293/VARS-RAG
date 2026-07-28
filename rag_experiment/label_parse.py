"""Parse free-form model text into paper 4-way offence+severity labels."""
from __future__ import annotations

import re
from collections import Counter

from rag_experiment.labels import (
    LABEL_NO_CARD,
    LABEL_NO_OFFENCE,
    LABEL_RED,
    LABEL_YELLOW,
    PAPER_LABELS,
)


def parse_paper_label(text: str) -> str:
    """
    Extract one of the 4 paper labels (or Unknown).

    Priority:
    1) Explicit canonical phrases / Final: line
    2) Card keywords (red / yellow / no card) implying Offence+*
    3) No-offence / no-foul phrases
    4) Generic \"foul/offence\" -> Offence + No card (weak default)
    """
    raw = (text or "").strip()
    if not raw:
        return "Unknown"
    low = raw.lower()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    head = lines[0].lower() if lines else low

    # Explicit Final: ...
    m = re.search(
        r"final\s*:\s*(no offence|no offense|offence\s*\+\s*no card|"
        r"offence\s*\+\s*yellow(?:\s*card)?|offence\s*\+\s*red(?:\s*card)?|"
        r"yellow(?:\s*card)?|red(?:\s*card)?|no card)",
        low,
    )
    if m:
        return _normalize_match(m.group(1))

    for scope in (head, low):
        mapped = _match_scope(scope)
        if mapped != "Unknown":
            return mapped

    # Chinese keywords (model may mix languages)
    if "红牌" in raw:
        return LABEL_RED
    if "黄牌" in raw:
        return LABEL_YELLOW
    if "无犯规" in raw or "不犯规" in raw or "没有犯规" in raw:
        return LABEL_NO_OFFENCE
    if "犯规" in raw and "无犯规" not in raw and "不犯规" not in raw:
        if "不出示牌" in raw or "无牌" in raw or "不罚牌" in raw:
            return LABEL_NO_CARD
        return LABEL_NO_CARD

    return "Unknown"


def _normalize_match(s: str) -> str:
    s = s.lower().strip()
    if "no offence" in s or "no offense" in s:
        return LABEL_NO_OFFENCE
    if "red" in s:
        return LABEL_RED
    if "yellow" in s:
        return LABEL_YELLOW
    if "no card" in s:
        return LABEL_NO_CARD
    return "Unknown"


def _match_scope(scope: str) -> str:
    # Order matters: check no-offence before offence; red before yellow before no card
    if "no offence" in scope or "no offense" in scope or "no foul" in scope:
        return LABEL_NO_OFFENCE
    if re.search(r"\bred\s*card\b", scope) or re.search(r"\bsending\s*off\b", scope):
        return LABEL_RED
    if re.search(r"\byellow\s*card\b", scope) or "caution" in scope:
        return LABEL_YELLOW
    if "offence + no card" in scope or "offense + no card" in scope:
        return LABEL_NO_CARD
    if "offence + yellow" in scope or "offense + yellow" in scope:
        return LABEL_YELLOW
    if "offence + red" in scope or "offense + red" in scope:
        return LABEL_RED
    if re.search(r"\bno\s*card\b", scope):
        # "no card" alone usually means foul without card in X-VARS answers
        if "offence" in scope or "offense" in scope or "foul" in scope:
            return LABEL_NO_CARD
        # Still often "What card? No card." after a foul — treat as No card class
        return LABEL_NO_CARD
    if re.search(r"\boffence\b", scope) or re.search(r"\boffense\b", scope) or re.search(
        r"\bfoul\b", scope
    ):
        return LABEL_NO_CARD
    return "Unknown"


def majority_label(votes: list[str]) -> str:
    valid = [x for x in votes if x in PAPER_LABELS]
    if not valid:
        return "Unknown"
    return Counter(valid).most_common(1)[0][0]


def get_parse_fn(mode: str = "rule"):
    """
    Parameters
    ----------
    mode:
        ``rule`` — regex/keyword (fast, collapses many free-form answers to No card)
        ``llm`` — DashScope/Qwen extraction (paper uses ChatGPT-3.5 for this step)
        ``hybrid`` — llm with rule fallback on API errors
    """
    mode = (mode or "rule").strip().lower()
    if mode == "rule":
        return parse_paper_label
    if mode == "llm":
        from rag_experiment.llm_parse import parse_paper_label_llm

        return parse_paper_label_llm
    if mode == "hybrid":
        from rag_experiment.llm_parse import parse_paper_label_hybrid

        return parse_paper_label_hybrid
    raise ValueError(f"Unknown parse mode: {mode}")
