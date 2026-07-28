"""在实验运行前检查本地权重目录；缺失时从 Hugging Face 拉取到与 constants 一致的路径。"""

from __future__ import annotations

import glob
import os
from typing import Optional

HF_REPO_X_VARS = "coisini9293/football_penalty_x_vars"
HF_REPO_TOKENIZER = "coisini9293/football_penalty_tokenizer"
HF_REPO_CLIP = "coisini9293/football_penalty_clip"


def _has_llm_weights(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return False
    if glob.glob(os.path.join(model_dir, "*.safetensors")) or glob.glob(
        os.path.join(model_dir, "*.safetensors.index.json")
    ):
        return True
    if glob.glob(os.path.join(model_dir, "*.bin")):
        return True
    return False


def _has_tokenizer_dir(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    markers = (
        "tokenizer.model",
        "tokenizer.json",
        "vocab.txt",
        "spiece.model",
    )
    return any(os.path.isfile(os.path.join(d, m)) for m in markers)


def _has_clip_dir(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    if not os.path.isfile(os.path.join(d, "config.json")):
        return False
    return bool(
        glob.glob(os.path.join(d, "*.safetensors"))
        or glob.glob(os.path.join(d, "*.bin"))
        or glob.glob(os.path.join(d, "pytorch_model.bin"))
    )


def _snapshot(repo_id: str, local_dir: str, token: Optional[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "缺少 huggingface_hub，无法自动下载模型。请执行: pip install huggingface_hub"
        ) from e

    os.makedirs(local_dir, exist_ok=True)
    print(f"[ensure_models] 正在从 Hugging Face 下载 {repo_id} -> {local_dir} ...", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
    )
    print(f"[ensure_models] 完成: {local_dir}", flush=True)


def ensure_experiment_models() -> None:
    """若 BASE_MODEL_DIR / tokenizer / CLIP 目录不完整，则 snapshot 下载对应公开仓库。"""
    from experiment.constants import BASE_MODEL_DIR, LOCAL_CLIP_DIR, LOCAL_TOKENIZER_DIR

    token = os.environ.get("HF_TOKEN")

    if not _has_llm_weights(BASE_MODEL_DIR):
        _snapshot(HF_REPO_X_VARS, BASE_MODEL_DIR, token)
    else:
        print(f"[ensure_models] 已存在主模型目录: {BASE_MODEL_DIR}", flush=True)

    if not _has_tokenizer_dir(LOCAL_TOKENIZER_DIR):
        _snapshot(HF_REPO_TOKENIZER, LOCAL_TOKENIZER_DIR, token)
    else:
        print(f"[ensure_models] 已存在分词器目录: {LOCAL_TOKENIZER_DIR}", flush=True)

    if not _has_clip_dir(LOCAL_CLIP_DIR):
        _snapshot(HF_REPO_CLIP, LOCAL_CLIP_DIR, token)
    else:
        print(f"[ensure_models] 已存在 CLIP 目录: {LOCAL_CLIP_DIR}", flush=True)
