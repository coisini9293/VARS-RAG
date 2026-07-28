"""Compose final user message from LineSpec + RAG text (order: CoT → expert wrap → RAG)."""
from __future__ import annotations

from experiment.cot import prepend_cot
from experiment.line_presets import LineSpec
from experiment.prompt_engineering import (
    TASK_CLASSIFY_OFFENCE,
    apply_expert_prompt_style,
    attach_rag_block,
)


def build_user_question(spec: LineSpec, rag_text: str) -> str:
    text = TASK_CLASSIFY_OFFENCE
    if spec.use_cot:
        text = prepend_cot(text)
    if spec.use_expert_prompt:
        text = apply_expert_prompt_style(text)
    return attach_rag_block(text, rag_text)


def example_question_for_settings(spec: LineSpec) -> str:
    return build_user_question(spec, "")
