"""
Usage:
python

生成模型权重的“增量/差分”文件
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# 计算delta
def make_delta(base_model_path, target_model_path, delta_path, hub_repo_id):
    print("Loading base model") #加载基础模型
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) #加载基础模型

    print("Loading target model") #加载目标模型
    target = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True) #加载目标模型

    print("Calculating delta") #计算delta
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        if name not in base.state_dict():
            assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model' #如果名称不在基础模型中，则抛出异常
            continue
        if param.data.shape == base.state_dict()[name].shape: #如果参数的形状与基础模型的形状相同，则计算delta
            param.data -= base.state_dict()[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}' #如果名称不在基础模型中，则抛出异常
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] -= bparam

    print("Saving delta") #保存delta
    if hub_repo_id:
        kwargs = {"push_to_hub": True, "repo_id": hub_repo_id} #如果hub_repo_id不为空，则设置为True
    else:
        kwargs = {} #如果hub_repo_id为空，则设置为空
    target.save_pretrained(delta_path, **kwargs) #保存delta
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path) #加载目标tokenizer
    target_tokenizer.save_pretrained(delta_path, **kwargs) #保存delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--hub-repo-id", type=str, default=None)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path, args.hub_repo_id)
