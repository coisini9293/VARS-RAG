'''
clip视觉编码器
'''

import __future__
import os
import torch
import torchvision
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from transformers import AutoTokenizer, CLIPVisionModel
import numpy as np
import torchvision
from torch import dropout, nn
import gc
import logging, time, torch

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s")

# CLIP网络
class CLIPNetwork(torch.nn.Module):
    # 初始化
    def __init__(self, clip_model_id: Optional[str] = None, hf_token: Optional[str] = None):
        """
        初始化CLIPNetwork
        
        参数:
            clip_model_id (Optional[str]): HuggingFace模型ID（如 "coisini9293/football_penalty_clip"），
                                           如果提供则从HuggingFace加载，否则尝试从本地加载
            hf_token (Optional[str]): HuggingFace访问令牌，用于访问私有模型
        """
        super().__init__()

        # 优先使用传入的HuggingFace模型ID，否则从环境变量或默认本地目录读取
        if clip_model_id:
            vision_tower_name = clip_model_id
            use_local_files_only = False  # 从HuggingFace加载
            logger.info(f"[CLIP] 使用HuggingFace模型ID: {vision_tower_name}")
        else:
            vision_tower_name = os.environ.get('LOCAL_CLIP_DIR', os.path.join('visual_encoder', 'clip-vit-large-patch14'))
            use_local_files_only = True  # 从本地加载
            logger.info(f"[CLIP] 使用本地路径: {vision_tower_name}")

        t0 = time.time()
        logger.info(f"[CLIP] 开始加载视觉塔 | src={vision_tower_name}")
        try:
            self.vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower_name,
                token=hf_token,  # HuggingFace token
                low_cpu_mem_usage=True,
                local_files_only=use_local_files_only,  # 根据来源决定是否仅使用本地文件
                torch_dtype=torch.float16,  # 使用float16节省内存
            )
        except Exception as e:
            if clip_model_id:
                logger.error(f"[CLIP] 从HuggingFace加载视觉塔失败: {vision_tower_name}。错误: {e}")
            else:
                logger.error(f"[CLIP] 本地加载视觉塔失败，请将离线文件放置到: {vision_tower_name}。错误: {e}")
            raise
        logger.info(f"[CLIP] 视觉塔加载完成 | elapsed={time.time()-t0:.2f}s")

        feat_dim = 1024 #特征维度

        self.inter = nn.Sequential( #中间层
            nn.LayerNorm(1024),
            nn.Linear(1024, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential( #犯规层
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )


        self.fc_action = nn.Sequential( #动作层
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

    def forward(self, video): #前向传播
        t0 = time.time()
        logger.debug(f"[CLIP] forward start | video_shape={tuple(video.shape)}")
        out = self.vision_tower(video, output_hidden_states=True) #视觉塔输出

        # To create the features which we use as input
        select_hidden_state_layer = -2 #选择隐藏状态层
        select_hidden_state = out.hidden_states[select_hidden_state_layer] #选择隐藏状态
        batch_features = select_hidden_state[:, 1:] #批量特征
        video_features = batch_features.detach().cpu() #视频特征
        
        # 保存pooler_output的形状用于日志（在删除out之前）
        pooled_shape = tuple(out.pooler_output.shape) if hasattr(out, 'pooler_output') and out.pooler_output is not None else None
        
        out = torch.mean(out.pooler_output, dim=0) #池化输出
        out = out.unsqueeze(0) #扩展维度
        out = self.inter(out) #中间层输出
        out_off = self.fc_offence(out) #犯规层输出
        out_act = self.fc_action(out) #动作层输出

        # 清理中间变量
        del out #删除输出
        gc.collect() #垃圾回收
        torch.cuda.empty_cache() #清空缓存

        logger.debug(f"[CLIP] forward 完成 | elapsed={time.time()-t0:.2f}s, pooled={pooled_shape}")
        return out_off.squeeze(), out_act.squeeze(), video_features #返回犯规层输出、动作层输出和视频特征
