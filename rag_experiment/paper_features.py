"""Paper-aligned CLIP → spatio-temporal features (create_features.py / X-VARS Table 4)."""
from __future__ import annotations

from typing import List

import torch

# create_features.get_spatio_temporal_features: pad temporal to 44 + spatial 256 = 300
PAPER_NUM_TEMPORAL_TOKENS = 44
PAPER_SPATIAL_TOKENS = 256  # ViT-L/14 @ 224
PAPER_VIDEO_TOKEN_LEN = PAPER_NUM_TEMPORAL_TOKENS + PAPER_SPATIAL_TOKENS


def paper_spatio_temporal_features(
    frame_features: torch.Tensor,
    *,
    num_temporal_tokens: int = PAPER_NUM_TEMPORAL_TOKENS,
) -> torch.Tensor:
    """
    Match ``visual_encoder/create_features.py::get_spatio_temporal_features``.

    Parameters
    ----------
    frame_features : (T, S, C) CLIP hidden states (no CLS), e.g. T=16, S=256, C=1024
    """
    if frame_features.dim() != 3:
        raise ValueError(f"expected (T,S,C), got {tuple(frame_features.shape)}")
    t, _s, c = frame_features.shape
    temporal = frame_features.mean(dim=1)
    pad = int(num_temporal_tokens) - int(t)
    if pad > 0:
        padding = torch.zeros(pad, c, device=frame_features.device, dtype=frame_features.dtype)
        temporal = torch.cat([temporal, padding], dim=0)
    elif pad < 0:
        temporal = temporal[: int(num_temporal_tokens)]
    spatial = frame_features.mean(dim=0)
    return torch.cat([temporal, spatial], dim=0)


def encode_paper_video_features(
    frames_tensor: torch.Tensor,
    vision_tower_hf: torch.nn.Module,
    clip_device: torch.device,
    clip_bs: int,
) -> torch.Tensor:
    """CLIP frames → (1, 300, C) tensor for X-VARS LLM."""
    use_gpu = clip_device.type == "cuda"
    tgt_dtype = torch.float16 if use_gpu else torch.float32
    vision_model = vision_tower_hf.to(clip_device, dtype=tgt_dtype).eval()
    chunks: List[torch.Tensor] = []
    for start in range(0, frames_tensor.shape[0], int(clip_bs)):
        end = min(start + int(clip_bs), frames_tensor.shape[0])
        batch = frames_tensor[start:end].to(clip_device, dtype=tgt_dtype)
        with torch.no_grad():
            if use_gpu:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outs = vision_model(batch, output_hidden_states=True)
            else:
                outs = vision_model(batch, output_hidden_states=True)
        hs = outs.hidden_states[-2][:, 1:]
        chunks.append(hs.to(dtype=tgt_dtype))
        del outs, hs, batch
    frame_features = torch.cat(chunks, dim=0)
    seq = paper_spatio_temporal_features(frame_features.to(device=clip_device, dtype=tgt_dtype))
    if seq.shape[0] != PAPER_VIDEO_TOKEN_LEN:
        raise RuntimeError(f"paper feature len {seq.shape[0]} != {PAPER_VIDEO_TOKEN_LEN}")
    return seq.unsqueeze(0)
