"""Load X-VARS + CLIP without Streamlit cache."""

import os

import torch
from video_chatgpt.eval.model_utils import initialize_model

from experiment.constants import BASE_MODEL_DIR, LOCAL_CLIP_DIR, LOCAL_TOKENIZER_DIR


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu = device.type == "cuda"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    orig_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    model, vision_tower_hf, tokenizer, image_processor, _ = initialize_model(
        model_name=BASE_MODEL_DIR,
        projection_path=None,
        token=os.environ.get("HF_TOKEN"),
        local_tokenizer_dir=LOCAL_TOKENIZER_DIR if os.path.isdir(LOCAL_TOKENIZER_DIR) else None,
        local_clip_dir=LOCAL_CLIP_DIR if os.path.isdir(LOCAL_CLIP_DIR) else None,
        local_files_only=True,
    )
    if orig_cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = orig_cuda
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    dtype = torch.float16 if use_gpu else torch.float32
    model = model.to(device, dtype=dtype)
    vision_tower_hf = vision_tower_hf.to(device, dtype=dtype)
    return model, vision_tower_hf, tokenizer, image_processor, device
