"""Load SoccerNet-MVFoul / XFoul-style annotations with paper-aligned sampling."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rag_experiment.labels import map_annotation_to_paper_label


@dataclass(frozen=True)
class EvalItem:
    action_id: str
    clip_idx: int
    video_path: str
    y_true: str
    action_meta: Dict[str, Any]
    video_exists: bool
    camera_type: str = ""


def resolve_clip_video(dataset_root: str, url: str) -> str:
    norm = url.replace("\\", "/")
    parts = norm.split("/")
    low = [p.lower() for p in parts]
    if len(parts) >= 2 and low[0] == "dataset" and low[1] == "test":
        rel_parts = parts[2:]
    else:
        rel_parts = parts[-2:] if len(parts) >= 2 else parts
    base = os.path.join(dataset_root, *rel_parts)
    mp4 = base + ".mp4"
    if os.path.isfile(mp4):
        return mp4
    if os.path.isdir(base):
        cand = sorted(glob.glob(os.path.join(base, "*.mp4")))
        if cand:
            return cand[0]
    return mp4


def _pick_clip_index(clips: List[Dict[str, Any]], prefer_main: bool) -> int:
    """Prefer Main camera center (paper: single-view eval); else first clip."""
    if not prefer_main:
        return 0
    for i, clip in enumerate(clips):
        cam = str(clip.get("Camera type", "") or "").lower()
        if "main camera" in cam and "center" in cam:
            return i
    for i, clip in enumerate(clips):
        cam = str(clip.get("Camera type", "") or "").lower()
        if "main camera" in cam:
            return i
    return 0


def load_eval_items(
    annotations_path: str,
    dataset_root: str,
    max_action: int,
    *,
    skip_missing_video: bool = True,
    unit: str = "action",
    paper_protocol: bool = True,
    prefer_main_camera: bool = True,
) -> List[EvalItem]:
    """
    Parameters
    ----------
    unit:
        ``action`` — one clip per action (paper-aligned single-view default).
        ``clip`` — every available clip (legacy; inflates N via multi-view).
    paper_protocol:
        If True, map Offence+Severity to 4-way labels and skip Between / ambiguous severity.
    """
    if not os.path.isfile(annotations_path):
        raise FileNotFoundError(f"annotations not found: {annotations_path}")

    raw = json.load(open(annotations_path, encoding="utf-8"))
    actions = raw["Actions"]
    out: List[EvalItem] = []
    missing = 0
    skipped: Dict[str, int] = {}

    unit = (unit or "action").strip().lower()
    if unit not in ("action", "clip"):
        raise ValueError(f"Unknown unit={unit}; use action|clip")

    for aid in range(max_action + 1):
        key = str(aid)
        if key not in actions:
            continue
        act = actions[key]
        offence = act.get("Offence", "")
        severity = act.get("Severity", "")

        if paper_protocol:
            y_true, reason = map_annotation_to_paper_label(offence, severity)
            if y_true is None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
        else:
            y_true = str(offence)

        meta = {
            "Action class": act.get("Action class", ""),
            "Contact": act.get("Contact", ""),
            "Bodypart": act.get("Bodypart", ""),
            "Severity": severity,
            "Offence_raw": offence,
        }
        clips = act.get("Clips") or []
        if not clips:
            skipped["no_clips"] = skipped.get("no_clips", 0) + 1
            continue

        if unit == "action":
            indices = [_pick_clip_index(clips, prefer_main_camera)]
        else:
            indices = list(range(len(clips)))

        for ci in indices:
            clip = clips[ci]
            path = resolve_clip_video(dataset_root, clip["Url"])
            exists = os.path.isfile(path)
            if not exists:
                missing += 1
                if skip_missing_video:
                    continue
            out.append(
                EvalItem(
                    action_id=key,
                    clip_idx=ci,
                    video_path=path,
                    y_true=y_true,
                    action_meta=meta,
                    video_exists=exists,
                    camera_type=str(clip.get("Camera type", "") or ""),
                )
            )

    if not out:
        raise RuntimeError(
            f"No evaluable items (missing_videos={missing}, skipped={skipped}). "
            f"Check --dataset-root={dataset_root}"
        )
    print(
        f"[dataset] protocol={'paper_4way' if paper_protocol else 'legacy_offence'} "
        f"unit={unit} items={len(out)} missing_videos_skipped="
        f"{missing if skip_missing_video else 0} skipped_actions={skipped}",
        flush=True,
    )
    return out


def action_meta_for_item(item: EvalItem) -> Optional[Dict[str, Any]]:
    return item.action_meta


def summarize_label_distribution(items: List[EvalItem]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for it in items:
        dist[it.y_true] = dist.get(it.y_true, 0) + 1
    return dist
