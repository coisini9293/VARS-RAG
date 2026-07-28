"""Retrieve rule chunks and merge them; return text plus retrieval diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rag_experiment.presets import RagSettings

if TYPE_CHECKING:
    import x_vars_demo_with_streamlit as xv


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    hits: List[Dict[str, Any]]
    empty: bool
    top1_score: Optional[float]


def fetch_rag_context(
    rag: "xv.FootballRAGSystem",
    query: str,
    settings: RagSettings,
) -> RetrievalResult:
    if not settings.enabled:
        return RetrievalResult("", [], True, None)
    if not getattr(rag, "initialized", False):
        return RetrievalResult("", [], True, None)

    hits = rag.search(query, top_k=settings.search_top_k) or []
    if not hits:
        return RetrievalResult("", [], True, None)

    use = hits[: settings.merge_top_n]
    chunks: List[str] = []
    total = 0
    for h in use:
        full = str(h.get("text", "") or "")
        t = full
        if settings.truncate_chars is not None:
            lim = int(settings.truncate_chars)
            t = full[:lim] + ("..." if len(full) > lim else "")
        chunks.append(t)
        total += len(t)
        if settings.max_context_chars and total >= settings.max_context_chars:
            break

    merged = "\n---\n".join(chunks)
    if settings.max_context_chars and len(merged) > settings.max_context_chars:
        merged = merged[: settings.max_context_chars]

    top1 = float(hits[0]["score"]) if hits and "score" in hits[0] else None
    # Lightweight hit records for logging (truncate text for disk size)
    slim_hits = [
        {
            "score": float(h.get("score", 0.0)),
            "text_preview": str(h.get("text", ""))[:160],
            "text_len": len(str(h.get("text", ""))),
        }
        for h in use
    ]
    return RetrievalResult(merged, slim_hits, False, top1)
