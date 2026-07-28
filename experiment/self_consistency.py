"""Self-consistency: multiple stochastic decodes + majority vote on parsed labels."""
from __future__ import annotations

from typing import Callable

import torch
import x_vars_demo_with_streamlit as xv

from experiment.label_parse import majority_label, parse_offence_label


def predict_self_consistency(
    model,
    tokenizer,
    video_feat: torch.Tensor,
    prompt: str,
    *,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    seed0: int,
    parse_fn: Callable[[str], str] = parse_offence_label,
) -> tuple[str, str]:
    votes: list[str] = []
    texts: list[str] = []
    for v in range(n_samples):
        torch.manual_seed(seed0 + v)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed0 + v)
        out = xv.model_generate_simple(
            model=model,
            tokenizer=tokenizer,
            video_spatio_temporal_features=video_feat,
            prompt=prompt,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            debug=False,
        )
        texts.append(out)
        votes.append(parse_fn(out))
    pred = majority_label(votes)
    return pred, "\n---\n".join(texts)
