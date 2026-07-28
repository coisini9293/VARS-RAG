"""Single-pass generation (greedy or optional sampling)."""
from __future__ import annotations

import x_vars_demo_with_streamlit as xv

from experiment.label_parse import parse_offence_label


def predict_greedy(
    model,
    tokenizer,
    video_feat,
    prompt: str,
    *,
    temperature: float,
    max_new_tokens: int,
    do_sample: bool,
) -> tuple[str, str]:
    out = xv.model_generate_simple(
        model=model,
        tokenizer=tokenizer,
        video_spatio_temporal_features=video_feat,
        prompt=prompt,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        debug=False,
    )
    return parse_offence_label(out), out
