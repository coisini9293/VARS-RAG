"""CLIP frame batch → X-VARS video token sequence."""
from __future__ import annotations

from typing import List

import torch
import x_vars_demo_with_streamlit as xv


def encode_video_features(
    frames_tensor: torch.Tensor,
    vision_tower_hf: torch.nn.Module,
    clip_device: torch.device,
    clip_bs: int,
    video_token_len: int,
) -> torch.Tensor:
    use_gpu = clip_device.type == "cuda"
    tgt_dtype = torch.float16 if use_gpu else torch.float32
    vision_model = vision_tower_hf.to(clip_device, dtype=tgt_dtype).eval()
    collected: List[torch.Tensor] = []
    for start in range(0, frames_tensor.shape[0], int(clip_bs)):
        end = min(start + int(clip_bs), frames_tensor.shape[0])
        batch = frames_tensor[start:end].to(clip_device).to(dtype=tgt_dtype)
        with torch.no_grad():
            if use_gpu:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outs = vision_model(batch, output_hidden_states=True)
            else:
                outs = vision_model(batch, output_hidden_states=True)
        hs = outs.hidden_states[-2][:, 1:]
        collected.append(hs.to(dtype=tgt_dtype))
        del outs, hs, batch
    frame_features = torch.cat(collected, dim=0)
    seq = xv.make_video_features_from_clip_hidden(
        frame_features.to(device=clip_device, dtype=tgt_dtype),
        target_length=int(video_token_len),
    )
    seq = seq.to(device=clip_device, dtype=tgt_dtype)
    return seq.unsqueeze(0)
