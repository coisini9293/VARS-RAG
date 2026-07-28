"""Resolve clip paths and load annotation index."""
from __future__ import annotations

import glob
import json
import os
from typing import List, Tuple


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


def load_eval_items(
    annotations_path: str, dataset_root: str, max_action: int
) -> List[Tuple[str, int, str, str]]:
    raw = json.load(open(annotations_path, encoding="utf-8"))
    actions = raw["Actions"]
    out: List[Tuple[str, int, str, str]] = []
    for aid in range(max_action + 1):
        key = str(aid)
        if key not in actions:
            continue
        act = actions[key]
        y_true = act["Offence"]
        for ci, clip in enumerate(act["Clips"]):
            path = resolve_clip_video(dataset_root, clip["Url"])
            out.append((key, ci, path, y_true))
    return out
