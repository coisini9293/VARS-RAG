"""
Task prompts aligned with X-VARS paper questions (Held et al., CVPRW 2024).

Paper evaluation (Table 4): ask whether it is a foul and the corresponding severity,
then use an LLM (paper: ChatGPT-3.5) to extract the 4-way label from the free-form answer.

Paper training prompt shape (simplified):
  USER: <P_foul><P_sev> <Question> <video_tokens>
"""
from __future__ import annotations

# Exact wording from the paper for the two primary refereeing questions.
PAPER_Q_FOUL = "Is it a foul or not? Why?"
PAPER_Q_CARD = "What card would you give? Why?"

# Combined query used for offence+severity recognition (single generation pass).
# Mirrors: ask foul + severity, then LLM-extract the 4-way class.
TASK_PAPER_FOUL_AND_CARD = (
    f"{PAPER_Q_FOUL}\n"
    f"{PAPER_Q_CARD}\n"
    "Answer both questions briefly, referring to the Laws of the Game when helpful."
)

# Backward-compatible aliases
TASK_CLASSIFY_OFFENCE_SEVERITY = TASK_PAPER_FOUL_AND_CARD
TASK_CLASSIFY_OFFENCE = TASK_PAPER_FOUL_AND_CARD

RAG_RULES_HEADER = "\n\nReference rules:\n{rag_text}\n"

FIXED_RETRIEVAL_QUERY = (
    "football foul offence no offence yellow card red card tackling charging "
    "careless reckless serious foul play "
    "犯规 黄牌 红牌 冲撞 铲球 不正当行为 鲁莽 严重犯规"
)


def attach_rag_block(user_message: str, rag_text: str) -> str:
    if not rag_text.strip():
        return user_message
    return user_message + RAG_RULES_HEADER.format(rag_text=rag_text)


def attach_classifier_text(user_message: str, classifier_text: str) -> str:
    """
    Prepend X-VARS-style CLIP multi-task prediction sentence.

    Paper puts <P_foul><P_sev> before <Question>. ``classifier_text`` is already
    formatted like custom_dataset.py:
      \"The prediction for this video is a tackle , foul and a yellow card.\"
    """
    text = (classifier_text or "").strip()
    if not text:
        return user_message
    return text + "\n" + user_message.lstrip()


def build_user_question(rag_text: str, classifier_text: str = "") -> str:
    """
    Assemble the user message (video tokens added later by ``build_prompt``).

    Order (paper-aligned + optional RAG):
      1) CLIP classifier prediction text  (<P_foul>/<P_sev>)
      2) Paper questions (foul? + card?)
      3) Optional retrieved Laws of the Game snippets (RAG)
    """
    text = TASK_PAPER_FOUL_AND_CARD
    text = attach_classifier_text(text, classifier_text)
    return attach_rag_block(text, rag_text)


def prompt_templates_for_log() -> dict[str, str]:
    return {
        "paper_q_foul": PAPER_Q_FOUL,
        "paper_q_card": PAPER_Q_CARD,
        "task_paper_foul_and_card": TASK_PAPER_FOUL_AND_CARD,
        "rag_rules_header": RAG_RULES_HEADER,
        "assembly_order": "classifier_text -> paper_questions -> rag_rules -> video_tokens",
    }
