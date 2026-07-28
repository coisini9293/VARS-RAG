"""
Paper-aligned label space for X-VARS / SoccerNet-MVFoul offence+severity.

X-VARS (Held et al., CVPRW 2024) evaluates a 4-way severity task:
  No offence | Offence + No card | Offence + Yellow card | Offence + Red card

Annotation Severity codes used in this repo / SoccerNet-MVFoul style JSON:
  1.0 -> No card,  3.0 -> Yellow card,  5.0 -> Red card
Ambiguous codes 2.0 / 4.0 and Offence==\"Between\" are excluded from the
primary paper protocol (reported as skipped).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# Canonical labels (exact strings used in metrics / plots / prompts)
LABEL_NO_OFFENCE = "No offence"
LABEL_NO_CARD = "Offence + No card"
LABEL_YELLOW = "Offence + Yellow card"
LABEL_RED = "Offence + Red card"

PAPER_LABELS = [LABEL_NO_OFFENCE, LABEL_NO_CARD, LABEL_YELLOW, LABEL_RED]
PAPER_LABELS_EXT = PAPER_LABELS + ["Unknown"]

# Legacy 3-way Offence field (MVFoul annotation), kept for optional mode
LEGACY_OFFENCE_LABELS = ["Offence", "No offence", "Between"]
LEGACY_OFFENCE_LABELS_EXT = LEGACY_OFFENCE_LABELS + ["Unknown"]

SEVERITY_TO_CARD = {
    "1.0": LABEL_NO_CARD,
    "1": LABEL_NO_CARD,
    "3.0": LABEL_YELLOW,
    "3": LABEL_YELLOW,
    "5.0": LABEL_RED,
    "5": LABEL_RED,
}


def map_annotation_to_paper_label(
    offence: str,
    severity: Any,
) -> Tuple[Optional[str], str]:
    """
    Map annotation fields to a paper 4-way label.

    Returns
    -------
    (label_or_None, skip_reason)
        label is None when the sample should be excluded from primary metrics.
    """
    off = (offence or "").strip()
    sev = str(severity if severity is not None else "").strip()

    if off == "Between":
        return None, "between"
    if off == "No offence":
        return LABEL_NO_OFFENCE, ""
    if off != "Offence":
        return None, f"unknown_offence:{off}"

    mapped = SEVERITY_TO_CARD.get(sev)
    if mapped is None:
        if sev in ("2.0", "2", "4.0", "4"):
            return None, f"ambiguous_severity:{sev}"
        if not sev:
            return None, "missing_severity"
        return None, f"unknown_severity:{sev}"
    return mapped, ""


def paper_protocol_notes() -> Dict[str, str]:
    return {
        "task": "offence_severity_4way",
        "reference": "X-VARS (Held et al., CVPRW 2024) Table 4: Offence+Severity",
        "labels": ", ".join(PAPER_LABELS),
        "frames": (
            "16 frames @ 224px; foul-centered 8 before + 8 after at ~3.0s "
            "(create_features convention; annotation Timestamp is match-absolute)"
        ),
        "questions": (
            "Is it a foul or not? Why? / What card would you give? Why? "
            "(paper primary refereeing questions; combined in one generation)"
        ),
        "view": "single view per action (prefer Main camera center, else first clip)",
        "classifier_text": (
            "CLIPNetwork + 14_model.pth.tar predictions prepended as "
            "'The prediction for this video is ...' before the questions"
        ),
        "metrics": "Accuracy + Balanced Accuracy (+ Macro-F1 auxiliary)",
        "label_parse": "LLM extract from free-form answers (paper: ChatGPT-3.5; here: Qwen via .env)",
        "excluded": "Offence==Between; Severity in {2.0,4.0,\"\"}",
    }
