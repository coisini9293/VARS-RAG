"""
Foul-centered frame sampling aligned with X-VARS / this repo's create_features.

Paper: 16 frames @ 224, 8 before + 8 after the foul.
Repo convention (create_features.py): foul is typically at ~3.0s in the ~5s clip.
Annotation ``Timestamp`` is absolute match time (ms) and cannot locate the foul
inside the short clip without the clip's start time — so we center at ``foul_time_sec``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from decord import bridge as decord_bridge


def paper_xvars_indices(
    total_frames: int,
    *,
    start_frame: int = 63,
    end_frame: int = 87,
    source_fps: float = 25.0,
    target_fps: float = 17.0,
) -> List[int]:
    """
    Frame indices used by ``visual_encoder/create_features.py`` (63–87 @ 25fps,
    subsampled to ~17fps → 16 frames).
    """
    if total_frames <= 0:
        return []
    start = min(int(start_frame), max(0, total_frames - 1))
    end = min(int(end_frame), total_frames)
    if end <= start:
        return foul_centered_indices(total_frames, fps=source_fps, foul_time_sec=3.0)
    window = end - start
    factor = window / ((window / float(source_fps)) * float(target_fps))
    return [start + j for j in range(window) if j % factor < 1]


def foul_centered_indices(
    total_frames: int,
    *,
    fps: float,
    foul_time_sec: float = 3.0,
    n_before: int = 8,
    n_after: int = 8,
) -> List[int]:
    """Return ``n_before + n_after`` frame indices centered on the foul time."""
    if total_frames <= 0:
        return []
    n = int(n_before) + int(n_after)
    if total_frames <= n:
        # Fall back to all frames (caller may pad / use as-is)
        return list(range(total_frames))

    center = int(round(float(foul_time_sec) * float(fps)))
    center = max(0, min(total_frames - 1, center))
    start = center - int(n_before)
    end = center + int(n_after)  # exclusive → n_before + n_after frames
    # Shift window into [0, total_frames)
    if start < 0:
        end -= start
        start = 0
    if end > total_frames:
        shift = end - total_frames
        start -= shift
        end = total_frames
        start = max(0, start)
    indices = list(range(start, end))
    # Guard length (should already be n)
    if len(indices) < n:
        # Pad by expanding toward available side
        while len(indices) < n and indices[0] > 0:
            indices.insert(0, indices[0] - 1)
        while len(indices) < n and indices[-1] < total_frames - 1:
            indices.append(indices[-1] + 1)
    return indices[:n]


def uniform_indices(total_frames: int, num_frames: int) -> List[int]:
    if total_frames <= 0:
        return []
    num_frames = min(int(num_frames), total_frames)
    if num_frames <= 1:
        return [0]
    seg = float(total_frames - 1) / float(num_frames)
    return [int(round(seg * i + seg / 2.0)) for i in range(num_frames)]


def _preprocess_batch(
    batch: torch.Tensor,
    target_size: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if batch.dtype != torch.uint8:
        batch = batch.to(torch.uint8)
    batch = batch.permute(0, 3, 1, 2).contiguous().float() / 255.0
    if batch.shape[-2:] != target_size:
        batch = F.interpolate(batch, size=target_size, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=batch.device).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    tgt_dtype = torch.float16 if device.type == "cuda" else torch.float32
    return batch.to(device, dtype=tgt_dtype)


def load_frames_tensor(
    path: str,
    *,
    mode: str = "foul_centered",
    max_frames: int = 16,
    foul_time_sec: float = 3.0,
    n_before: int = 8,
    n_after: int = 8,
    target_size: Tuple[int, int] = (224, 224),
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Load and preprocess frames.

    Returns
    -------
    frames : (T, 3, 224, 224)
    meta : sampling diagnostics (fps, center, indices, mode)
    """
    device = device or torch.device("cpu")
    decord_bridge.set_bridge("torch")
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    fps = float(vr.get_avg_fps()) if total > 0 else 25.0
    if fps <= 1e-6:
        fps = 25.0

    mode = (mode or "paper_xvars").strip().lower()
    if mode in ("paper_xvars", "paper", "create_features", "xvars"):
        idx = paper_xvars_indices(total, source_fps=fps)
    elif mode in ("foul_centered", "paper_8_8", "8_8"):
        # Paper: 8 before + 8 after. If max_frames != 16, split evenly.
        if int(max_frames) != int(n_before + n_after):
            half = int(max_frames) // 2
            n_before, n_after = half, int(max_frames) - half
        idx = foul_centered_indices(
            total, fps=fps, foul_time_sec=foul_time_sec, n_before=n_before, n_after=n_after
        )
    elif mode in ("uniform", "even"):
        idx = uniform_indices(total, int(max_frames))
    else:
        raise ValueError(f"Unknown frame sample mode: {mode}")

    batch = vr.get_batch(idx)
    frames = _preprocess_batch(batch, target_size, device)
    meta = {
        "mode": mode,
        "fps": fps,
        "total_frames": total,
        "foul_time_sec": foul_time_sec,
        "indices": idx,
        "center_index": idx[len(idx) // 2] if idx else None,
        "n_frames": len(idx),
    }
    return frames, meta
