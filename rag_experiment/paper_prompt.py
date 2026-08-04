"""Football referee prompt assembly (custom_dataset / paper USER block)."""
from __future__ import annotations

from video_chatgpt.constants import (
    DEFAULT_VIDEO_PATCH_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
)
from video_chatgpt.video_conversation import SeparatorStyle, conv_templates

# Matches custom_dataset.py training template intent
DEFAULT_XVARS_CONV_MODE = "xvars_football_v1"


def build_xvars_prompt(
    user_message: str,
    video_token_len: int,
    *,
    conv_mode: str = DEFAULT_XVARS_CONV_MODE,
) -> str:
    """USER message + video placeholder tokens + ASSISTANT: (generation)."""
    vid_tokens = (
        DEFAULT_VID_START_TOKEN
        + DEFAULT_VIDEO_PATCH_TOKEN * int(video_token_len)
        + DEFAULT_VID_END_TOKEN
    )
    qs = user_message.strip() + "\n" + vid_tokens
    mode = conv_mode if conv_mode in conv_templates else DEFAULT_XVARS_CONV_MODE
    conv = conv_templates[mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()
