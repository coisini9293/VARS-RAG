"""
CLIP multi-task classifier (P_foul / P_sev / action) → text injected into the LLM prompt.

Matches ``custom_dataset.py`` wording and ``create_features.py`` label mapping from
``visual_encoder/14_model.pth.tar`` + ``CLIPNetwork``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from visual_encoder.clip_model import CLIPNetwork
from visual_encoder.config.classes import INVERSE_EVENT_DICTIONARY


@dataclass(frozen=True)
class ClassifierPrediction:
    action_class: str
    offence: str
    severity: str
    prompt_text: str
    offence_logits_argmax: int
    action_logits_argmax: int


_ACTION_PHRASE = {
    "Tackling": "a tackle ",
    "Standing tackling": "a foot duel ",
    "Elbowing": "using his elbows or arms ",
    "Holding": "holding ",
    "High leg": "a high leg ",
    "Pushing": "pushing ",
    "Challenge": "a shoulder challenge ",
    "Dive": "a simulation ",
}


def format_prediction_prompt_text(action_class: str, offence: str, severity: str) -> str:
    """Build the exact-style sentence used by original X-VARS ``VARS_Explain`` dataset."""
    pred_action = _ACTION_PHRASE.get(action_class, (action_class + " ").lower() if action_class else "")

    if offence == "Offence":
        pred_off = ", foul and "
    elif offence == "No offence":
        pred_off = "and no foul."
    else:
        pred_off = ", "

    if offence == "Offence":
        if severity == "1.0":
            pred_off += "no card."
        elif severity == "3.0":
            pred_off += "a yellow card."
        elif severity == "5.0":
            pred_off += "a red card."
        else:
            pred_off += "no card."

    return "The prediction for this video is " + pred_action + pred_off


def _map_offence_severity(argmax_sev: int) -> tuple[str, str]:
    # create_features.py / CLIPNetwork fc_offence (4-way)
    if argmax_sev == 0:
        return "No offence", ""
    if argmax_sev == 1:
        return "Offence", "1.0"
    if argmax_sev == 2:
        return "Offence", "3.0"
    if argmax_sev == 3:
        return "Offence", "5.0"
    return "No offence", ""


def load_clip_classifier(
    *,
    weights_path: str,
    clip_dir: str,
    device: torch.device,
) -> CLIPNetwork:
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"CLIP classifier weights not found: {weights_path}")
    if not os.path.isdir(clip_dir):
        raise FileNotFoundError(f"CLIP vision dir not found: {clip_dir}")

    # CLIPNetwork reads LOCAL_CLIP_DIR when clip_model_id is None
    prev = os.environ.get("LOCAL_CLIP_DIR")
    os.environ["LOCAL_CLIP_DIR"] = clip_dir
    try:
        net = CLIPNetwork()
    finally:
        if prev is None:
            os.environ.pop("LOCAL_CLIP_DIR", None)
        else:
            os.environ["LOCAL_CLIP_DIR"] = prev

    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing:
        print(f"[clip_classifier] missing_keys={len(missing)} e.g. {missing[:3]}", flush=True)
    if unexpected:
        print(f"[clip_classifier] unexpected_keys={len(unexpected)} e.g. {unexpected[:3]}", flush=True)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    net = net.to(device=device, dtype=dtype).eval()
    print(f"[clip_classifier] loaded {weights_path} on {device}", flush=True)
    return net


@torch.inference_mode()
def predict_from_frames(
    net: CLIPNetwork,
    frames: torch.Tensor,
) -> ClassifierPrediction:
    """
    Parameters
    ----------
    frames : (T, 3, 224, 224) already CLIP-normalized, on the classifier device.
    """
    if frames.ndim != 4:
        raise ValueError(f"expected (T,3,H,W), got {tuple(frames.shape)}")
    device = next(net.parameters()).device
    dtype = next(net.parameters()).dtype
    x = frames.to(device=device, dtype=dtype)
    out_off, out_act, _feats = net(x)

    # out_* may be 1D (C,) after squeeze
    if out_off.ndim == 0:
        raise RuntimeError("fc_offence returned scalar unexpectedly")
    if out_off.ndim > 1:
        out_off = out_off.mean(dim=0) if out_off.shape[0] > 1 else out_off.squeeze(0)
    if out_act.ndim > 1:
        out_act = out_act.mean(dim=0) if out_act.shape[0] > 1 else out_act.squeeze(0)

    sev_i = int(torch.argmax(out_off.detach().float().cpu()).item())
    act_i = int(torch.argmax(out_act.detach().float().cpu()).item())
    action_map = INVERSE_EVENT_DICTIONARY["action_class"]
    action = action_map.get(act_i, "Dont know")
    offence, severity = _map_offence_severity(sev_i)
    text = format_prediction_prompt_text(action, offence, severity)
    return ClassifierPrediction(
        action_class=action,
        offence=offence,
        severity=severity,
        prompt_text=text,
        offence_logits_argmax=sev_i,
        action_logits_argmax=act_i,
    )


def prediction_to_dict(pred: ClassifierPrediction) -> Dict[str, Any]:
    return {
        "Action class": pred.action_class,
        "Offence": pred.offence,
        "Severity": pred.severity,
        "prompt_text": pred.prompt_text,
        "offence_argmax": pred.offence_logits_argmax,
        "action_argmax": pred.action_logits_argmax,
    }
