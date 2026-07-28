"""Build retrieval queries. Never include the Offence ground-truth label."""
from __future__ import annotations

from typing import Any, Dict, Optional

from rag_experiment.prompts import FIXED_RETRIEVAL_QUERY


def build_retrieval_query(
    query_mode: str,
    action_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Parameters
    ----------
    query_mode:
        ``fixed`` — same query for every sample.
        ``adaptive`` — append Action class / Contact (observable attributes), never Offence.
    action_meta:
        Optional annotation fields for the action (not the clip).
    """
    mode = (query_mode or "fixed").strip().lower()
    if mode == "fixed":
        return FIXED_RETRIEVAL_QUERY

    if mode != "adaptive":
        raise ValueError(f"Unknown query_mode: {query_mode}")

    parts = [FIXED_RETRIEVAL_QUERY]
    meta = action_meta or {}
    action_class = str(meta.get("Action class", "") or "").strip()
    contact = str(meta.get("Contact", "") or "").strip()
    bodypart = str(meta.get("Bodypart", "") or "").strip()
    if action_class:
        parts.append(action_class)
    if contact:
        parts.append(contact)
    if bodypart:
        parts.append(bodypart)
    return " ".join(parts)
