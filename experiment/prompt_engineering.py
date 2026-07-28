"""Prompt templates: base task wording + expert role / format (L3-style)."""
from __future__ import annotations

TASK_CLASSIFY_OFFENCE = (
    "Based on the video, choose exactly one label: Offence, No offence, or Between. "
    "Answer with the label on the first line, then one short sentence of justification."
)

EXPERT_ROLE_AND_FORMAT = (
    "You are a professional football video referee analyst.\n"
    "Judge the clip using the rules. Output format:\n"
    "Line 1: exactly one of Offence / No offence / Between\n"
    "Line 2: one sentence of justification.\n\n"
)

RAG_RULES_HEADER = "\n\nReference rules:\n{rag_text}\n"


def prompt_templates_for_log() -> dict[str, str]:
    return {
        "task_classify_offence": TASK_CLASSIFY_OFFENCE,
        "expert_role_and_format": EXPERT_ROLE_AND_FORMAT,
        "rag_rules_header": RAG_RULES_HEADER,
    }


def apply_expert_prompt_style(user_message: str) -> str:
    return EXPERT_ROLE_AND_FORMAT + user_message


def attach_rag_block(user_message: str, rag_text: str) -> str:
    if not rag_text.strip():
        return user_message
    return user_message + RAG_RULES_HEADER.format(rag_text=rag_text)
