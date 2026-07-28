"""Knowledge RAG: retrieve rule chunks and merge into context (L1 / L2 style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import x_vars_demo_with_streamlit as xv


@dataclass(frozen=True)
class RagSettings:
    enabled: bool
    search_top_k: int
    merge_top_n: int
    truncate_chars: Optional[int]
    max_context_chars: int


def fetch_rag_context(
    rag: "xv.FootballRAGSystem",
    query: str,
    settings: RagSettings,
) -> str:
    if not settings.enabled or not rag.initialized:
        return ""
    hits = rag.search(query, top_k=settings.search_top_k)
    if not hits:
        return ""
    use = hits[: settings.merge_top_n]
    chunks: list[str] = []
    total = 0
    for h in use:
        full = h.get("text", "")
        t = full
        if settings.truncate_chars is not None:
            t = full[: settings.truncate_chars] + (
                "..." if len(full) > settings.truncate_chars else ""
            )
        chunks.append(t)
        total += len(t)
        if total >= settings.max_context_chars:
            break
    merged = "\n---\n".join(chunks)
    if settings.max_context_chars and len(merged) > settings.max_context_chars:
        merged = merged[: settings.max_context_chars]
    return merged
