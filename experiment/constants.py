import os

LABELS = ["Offence", "No offence", "Between"]
LABELS_EXT = LABELS + ["Unknown"]

BASE_MODEL_DIR = os.path.join("X-VARS", "X-VARS_weights")
LOCAL_TOKENIZER_DIR = os.path.join("X-VARS", "tokenizer")
LOCAL_CLIP_DIR = os.path.join("visual_encoder", "clip-vit-large-patch14")
