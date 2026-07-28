import os
import torch
import torch.nn as nn
from transformers import Trainer
from typing import Optional
from trl import SFTTrainer
from transformers import Seq2SeqTrainer
# 解包模型
def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

# 视频ChatGPT训练器
class VideoChatGPTTrainer(SFTTrainer):
    # 保存
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False): #如果使用mm_mlp_adapter
            # 保存模型
            _state_dict = state_dict
            if _state_dict is None: #如果状态字典为空
                # 只保存模型本身
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {} #权重保存
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in'] #匹配键
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1] #当前文件夹
            parent_folder = os.path.dirname(output_dir) #父文件夹
            if current_folder.startswith('checkpoint-'): #如果当前文件夹以checkpoint-开头
                mm_projector_folder = os.path.join(parent_folder, "mm_projector") #mm_projector文件夹
                os.makedirs(mm_projector_folder, exist_ok=True) #创建文件夹
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin')) #保存权重
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'), ) #保存权重

        # super(VideoChatGPTTrainer, self)._save(output_dir, state_dict) #调用父类方法
