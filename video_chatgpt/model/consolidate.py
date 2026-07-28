"""
Usage:
python3 -m llava.model.consolidate --src ~/model_weights/llava-7b --dst ~/model_weights/llava-7b_consolidate

合并模型权重
将 base model 和 delta/adapter 权重合并为最终部署模型s
"""
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from video_chatgpt.model import *

# 合并模型
def consolidate_ckpt(src_path, dst_path):
    print("Loading model") #加载模型
    src_model = AutoModelForCausalLM.from_pretrained(src_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) #加载模型
    src_tokenizer = AutoTokenizer.from_pretrained(src_path) #加载tokenizer
    src_model.save_pretrained(dst_path) #保存模型
    src_tokenizer.save_pretrained(dst_path) #保存tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser() #解析参数
    parser.add_argument("--src", type=str, required=True) #设置源路径
    parser.add_argument("--dst", type=str, required=True) #设置目标路径

    args = parser.parse_args() #解析参数

    consolidate_ckpt(args.src, args.dst) #合并模型
