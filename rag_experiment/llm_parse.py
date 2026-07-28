"""
LLM-based label extraction from free-form X-VARS answers.

Paper (X-VARS, CVPRW 2024): after asking the model foul + severity, they
\"asked ChatGPT-3.5 to extract the classification predictions from the generated
explanations.\" This module mirrors that step with an OpenAI-compatible API
(default: DashScope Qwen via ``.env``).
"""
from __future__ import annotations

import os
from typing import Optional

from rag_experiment.label_parse import parse_paper_label
from rag_experiment.labels import PAPER_LABELS

# Default cheap/fast chat model on DashScope compatible mode
DEFAULT_LLM_PARSE_MODEL = os.environ.get("QWEN_PARSE_MODEL", "qwen-plus")

_SYSTEM = (
    "You extract football referee classification labels from free-form answers "
    "to the questions: 'Is it a foul or not? Why?' and 'What card would you give? Why?'.\n"
    "This matches the X-VARS paper evaluation step (ChatGPT extracts predictions "
    "from generated explanations).\n"
    "Return EXACTLY one of these four strings and nothing else:\n"
    + "\n".join(f"- {x}" for x in PAPER_LABELS)
    + "\n"
    "Rules:\n"
    "- No foul / not a foul / no offence → No offence\n"
    "- Foul + no card / no caution / only a free kick → Offence + No card\n"
    "- Yellow card / caution / reckless → Offence + Yellow card\n"
    "- Red card / sending-off / serious foul play / violent conduct → Offence + Red card\n"
    "- If it is a foul but the card is not stated → Offence + No card\n"
    "- Do not invent a yellow/red card that the text does not imply.\n"
)

_USER_TMPL = (
    "The model was asked: Is it a foul or not? Why? What card would you give? Why?\n"
    "Extract the combined offence+severity label from its answer:\n"
    "----- BEGIN -----\n{text}\n----- END -----\n"
    "Reply with exactly one label string."
)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # Project root .env
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(root, ".env"), override=False)


def get_llm_client():
    """OpenAI-compatible client for DashScope / Qwen."""
    _load_dotenv()
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = (
        os.environ.get("QWEN_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY (or OPENAI_API_KEY). Put it in project .env"
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


def _canonicalize_llm_reply(reply: str) -> str:
    text = (reply or "").strip().strip('"').strip("'")
    # Take first non-empty line
    for line in text.splitlines():
        line = line.strip().strip("-").strip()
        if not line:
            continue
        # Exact match
        for lab in PAPER_LABELS:
            if line.lower() == lab.lower():
                return lab
        # Fuzzy containment
        low = line.lower()
        if "no offence" in low or "no offense" in low or "no foul" in low:
            return PAPER_LABELS[0]
        if "red" in low:
            return PAPER_LABELS[3]
        if "yellow" in low:
            return PAPER_LABELS[2]
        if "no card" in low or "offence" in low or "offense" in low or "foul" in low:
            return PAPER_LABELS[1]
    # Fallback: rule parser on the LLM reply itself
    return parse_paper_label(text)


def parse_paper_label_llm(
    text: str,
    *,
    client=None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """Ask an LLM to map free-form text → one of PAPER_LABELS."""
    raw = (text or "").strip()
    if not raw:
        return "Unknown"
    cli = client or get_llm_client()
    model_name = model or DEFAULT_LLM_PARSE_MODEL
    resp = cli.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER_TMPL.format(text=raw[:4000])},
        ],
    )
    reply = (resp.choices[0].message.content or "").strip()
    return _canonicalize_llm_reply(reply)


def parse_paper_label_hybrid(text: str, *, client=None, model: Optional[str] = None) -> str:
    """
    Prefer LLM extraction (paper-style). If LLM fails, fall back to rules.
    """
    try:
        return parse_paper_label_llm(text, client=client, model=model)
    except Exception as e:
        print(f"[llm_parse] fallback to rules: {type(e).__name__}: {e}", flush=True)
        return parse_paper_label(text)


def smoke_parse_batch(texts: list[str], *, model: Optional[str] = None) -> list[dict]:
    """Parse a small list; returns dicts with rule vs llm labels."""
    client = get_llm_client()
    out = []
    for t in texts:
        rule = parse_paper_label(t)
        try:
            llm = parse_paper_label_llm(t, client=client, model=model)
            err = None
        except Exception as e:
            llm = "Unknown"
            err = f"{type(e).__name__}: {e}"
        out.append({"raw": t, "rule": rule, "llm": llm, "error": err})
    return out
