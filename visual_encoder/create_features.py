"""
足球视频特征提取模块
============================

本模块是X-VARS（Explainable Video Assistant Referee System）系统的核心组件之一，
负责从足球视频中提取视觉特征，并进行多任务分类预测。

主要功能：
1. 视频帧采样和预处理
2. 使用微调的CLIP模型提取时空特征
3. 预测动作类型和犯规严重程度
4. 生成用于语言模型输入的特征表示

技术架构：
- 基于CLIP视觉编码器进行特征提取
- 采用时空特征融合策略
- 支持多任务分类（动作识别 + 犯规检测）
"""

import os
from clip_model import CLIPNetwork
import torch
import pickle
import math
import argparse
import numpy as np
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
from config.classes import INVERSE_EVENT_DICTIONARY
from decord import VideoReader, cpu
import json
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def load_video(vis_path, num_frm=1000):
    """
    视频加载和预处理函数
    ==================
    
    功能描述：
    从视频文件中加载指定数量的帧，并进行预处理以适配CLIP模型输入要求。
    
    处理流程：
    1. 使用VideoReader加载视频文件
    2. 计算实际需要采样的帧数（不超过视频总帧数）
    3. 通过get_seq_frames函数获取均匀分布的帧索引
    4. 将视频帧调整为CLIP标准输入尺寸（224x224）
    5. 转换为PIL Image格式供后续处理
    
    参数：
    -----
    vis_path : str
        视频文件的完整路径
    num_frm : int, 默认=1000
        期望采样的最大帧数，实际帧数取决于视频长度
    
    返回值：
    ------
    clip_imgs : List[PIL.Image]
        预处理后的图像列表，每个元素为224x224的PIL Image对象
    
    数学公式：
    --------
    实际帧数 = min(视频总帧数, num_frm)
    
    图像变换：
    原始尺寸(H, W) → 标准尺寸(224, 224) 通过双线性插值
    """
    # 初始化视频读取器，使用CPU上下文以节省GPU内存
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)  # 获取视频总帧数
    
    # 确定实际采样帧数，不能超过视频本身的帧数
    total_num_frm = min(total_frame_num, num_frm)
    
    # 获取均匀分布的帧索引序列
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    
    # 批量读取指定索引的帧，返回numpy数组格式 (n_frames, H, W, 3)
    img_array = vr.get_batch(frame_idx).asnumpy()

    # 获取图像数组的维度信息
    a, H, W, _ = img_array.shape
    
    # CLIP模型的标准输入尺寸
    h, w = 224, 224
    
    # 如果图像尺寸不符合CLIP标准，进行尺寸调整
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        # 转换为PyTorch张量并调整维度顺序：(N, H, W, C) → (N, C, H, W)
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        
        # 使用双线性插值调整图像尺寸到224x224
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        
        # 恢复原始维度顺序并转换为uint8格式：(N, C, H, W) → (N, H, W, C)
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    
    # 重塑数组形状以适配后续处理：(n_frames, H, W, C) → (1, n_frames, H, W, C)
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    # 将numpy数组转换为PIL Image对象列表
    clip_imgs = []
    for j in range(total_num_frm):
        # 提取单帧并转换为PIL Image格式
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    """
    均匀帧采样算法
    ==============
    
    功能描述：
    从视频的总帧数中均匀采样出指定数量的帧索引。该算法确保采样的帧在时间轴上均匀分布，
    能够很好地保持视频的时序信息。
    
    算法原理：
    1. 将视频时间轴划分为desired_num_frames个等长区间
    2. 在每个区间的中点位置选择一帧
    3. 通过数学计算确保帧索引的均匀分布
    
    数学公式：
    --------
    区间大小 seg_size = (total_num_frames - 1) / desired_num_frames
    
    对于第i个区间 (i = 0, 1, ..., desired_num_frames-1):
    - 区间起始位置: start = round(seg_size × i)
    - 区间结束位置: end = round(seg_size × (i + 1))
    - 选择的帧索引: frame_idx = (start + end) // 2
    
    参数：
    -----
    total_num_frames : int
        视频的总帧数
    desired_num_frames : int
        期望采样的帧数
    
    返回值：
    ------
    seq : List[int]
        采样得到的帧索引列表，长度为desired_num_frames
    
    示例：
    -----
    >>> get_seq_frames(100, 4)
    [12, 37, 62, 87]  # 在100帧中均匀选择4帧
    """
    # 计算每个采样区间的大小
    # 使用(total_num_frames - 1)确保最后一帧能被正确采样
    seg_size = float(total_num_frames - 1) / desired_num_frames
    
    seq = []  # 存储采样帧索引的列表
    
    # 遍历每个采样区间
    for i in range(desired_num_frames):
        # 计算当前区间的起始位置（四舍五入到最近的整数）
        start = int(np.round(seg_size * i))
        
        # 计算当前区间的结束位置
        end = int(np.round(seg_size * (i + 1)))
        
        # 选择区间中点作为采样帧索引
        # 使用整数除法确保结果为整数
        frame_idx = (start + end) // 2
        seq.append(frame_idx)

    return seq


def get_spatio_temporal_features(features, num_temporal_tokens=44):
    """
    时空特征融合算法
    ===============
    
    功能描述：
    将CLIP提取的视频特征转换为适合语言模型输入的时空融合特征表示。
    该算法分别提取时间维度和空间维度的特征，然后进行拼接融合。
    
    算法原理：
    1. 时间特征提取：对空间维度进行平均池化，保留时间序列信息
    2. 空间特征提取：对时间维度进行平均池化，保留空间布局信息
    3. 时间特征填充：确保时间特征达到指定的token数量
    4. 特征拼接：将时间特征和空间特征在第0维度拼接
    
    数学公式：
    --------
    输入特征维度: (t, s, c) 其中 t=时间, s=空间, c=通道
    
    时间特征提取:
    temporal_tokens = mean(features, axis=1)  # 形状: (t, c)
    
    空间特征提取:
    spatial_tokens = mean(features, axis=0)   # 形状: (s, c)
    
    时间特征填充 (如果 t < num_temporal_tokens):
    temporal_tokens = pad(temporal_tokens, padding_size, 'constant')
    
    最终融合:
    sp_features = concat([temporal_tokens, spatial_tokens], axis=0)
    输出形状: (num_temporal_tokens + s, c)
    
    参数：
    -----
    features : numpy.ndarray
        CLIP提取的原始视频特征，形状为 (t, s, c)
        - t: 时间维度（帧数）
        - s: 空间维度（空间patch数量，通常为256）
        - c: 特征通道数（通常为1024）
    num_temporal_tokens : int, 默认=44
        期望的时间token数量，用于确保时间特征的固定长度
    
    返回值：
    ------
    sp_features : numpy.ndarray
        时空融合特征，形状为 (num_temporal_tokens + s, c)
        前num_temporal_tokens行为时间特征，后s行为空间特征
    
    应用场景：
    --------
    该函数生成的特征将作为语言模型的输入，时间特征捕获动作的时序变化，
    空间特征捕获场景的空间布局，两者结合提供完整的视频理解能力。
    """
    # 解析输入特征的维度信息
    t, s, c = features.shape  # t:时间维度, s:空间维度, c:特征通道数

    # 提取时间特征：对空间维度进行平均池化
    # 这样可以保留每个时间步的全局空间信息
    temporal_tokens = np.mean(features, axis=1)  # 形状变为 (t, c)
    
    # 计算需要填充的时间token数量
    padding_size = num_temporal_tokens - t
    
    # 如果当前时间维度小于期望的token数量，进行零填充
    if padding_size > 0:
        # 在时间维度末尾填充零向量，确保时间特征长度固定
        temporal_tokens = np.pad(temporal_tokens, 
                               ((0, padding_size), (0, 0)), 
                               mode='constant')  # 形状变为 (num_temporal_tokens, c)

    # 提取空间特征：对时间维度进行平均池化
    # 这样可以获得每个空间位置在整个视频序列中的平均表示
    spatial_tokens = np.mean(features, axis=0)  # 形状变为 (s, c)
    
    # 将时间特征和空间特征在第0维度拼接
    # 最终形状: (num_temporal_tokens + s, c)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features

def get_features(video_path):
    """
    视频特征提取和多任务预测主函数
    =============================
    
    功能描述：
    这是整个特征提取流程的核心函数，负责从足球视频中提取特征并进行多任务预测。
    该函数专门针对足球犯规检测场景进行了优化，重点关注犯规动作通常发生的时间段。
    
    处理流程：
    1. 视频帧提取：从完整视频中提取关键时间段的帧
    2. 帧率调整：将帧率调整为标准的17fps
    3. 特征提取：使用微调的CLIP模型提取视频特征
    4. 多任务预测：同时预测动作类型和犯规严重程度
    5. 特征后处理：生成适合语言模型的时空融合特征
    
    关键参数设置：
    -------------
    根据足球视频分析的经验数据设置：
    - 关键帧范围：第63-87帧（约2.5-3.5秒）
    - 目标帧率：17fps（足球视频分析的标准帧率）
    - 原始帧率：25fps（SoccerNet数据集的标准帧率）
    
    数学公式：
    --------
    帧率转换因子:
    factor = (end_frame - start_frame) / (((end_frame - start_frame) / fps_beginning) × fps)
    factor = 24 / ((24 / 25) × 17) ≈ 1.47
    
    帧选择条件:
    选择第j帧当且仅当 j % factor < 1
    
    严重程度映射:
    0 → No offence
    1 → Offence + Severity 1.0 (轻微犯规)
    2 → Offence + Severity 3.0 (黄牌级别)
    3 → Offence + Severity 5.0 (红牌级别)
    
    参数：
    -----
    video_path : str
        输入视频文件的完整路径
    
    返回值：
    ------
    video_clip_features : numpy.ndarray
        时空融合特征，形状为 (num_tokens, 1024)
        用作语言模型的视觉输入
    values : dict
        预测结果字典，包含：
        - "Action class": 动作类型（如"Tackling", "Holding"等）
        - "Offence": 是否犯规（"Offence" 或 "No offence"）
        - "Severity": 犯规严重程度（"1.0", "3.0", "5.0" 或空字符串）
    
    依赖的全局变量：
    -------------
    - image_processor: CLIP图像预处理器
    - model: 微调的CLIPNetwork模型
    - INVERSE_EVENT_DICTIONARY: 类别索引到名称的映射字典
    
    注意事项：
    --------
    1. 该函数假设犯规动作主要发生在视频的3秒左右位置
    2. 使用GPU进行模型推理，需要确保CUDA可用
    3. 返回的特征已转换为float16格式以节省内存
    """
    # ==================== 视频时间段设置 ====================
    # 根据足球犯规检测的经验，犯规动作通常发生在第3秒左右（第75帧附近）
    # 因此选择第63-87帧作为关键分析区间
    start_frame = 63  # 分析起始帧
    end_frame = 87    # 分析结束帧
    
    # ==================== 帧率转换参数 ====================
    fps = 17           # 目标帧率：足球视频分析的标准帧率
    fps_beginning = 25 # 原始帧率：SoccerNet数据集的标准帧率
    
    # 计算帧率转换因子
    # 该因子用于从25fps的原始视频中选择帧以达到17fps的目标帧率
    frame_duration = (end_frame - start_frame) / fps_beginning  # 时间段长度（秒）
    target_frame_count = frame_duration * fps                    # 目标帧数
    factor = (end_frame - start_frame) / target_frame_count      # 采样间隔
    
    # ==================== 视频加载和预处理 ====================
    # 加载完整视频的所有帧
    video_frames = load_video(video_path)
    
    # 只保留关键时间段的帧（第63-87帧）
    # 这样可以减少计算量并聚焦于犯规动作最可能发生的时间段
    video_frames = video_frames[start_frame:end_frame]

    # 使用CLIP预处理器对图像进行标准化处理
    # 包括尺寸调整、归一化等操作，返回PyTorch张量格式
    frames = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']
    
    # ==================== 帧率调整和帧选择 ====================
    final_frames = None  # 存储最终选择的帧
    
    # 按照计算的因子选择帧以达到目标帧率
    # 这个循环实现了从25fps到17fps的转换
    for j in range(len(frames)):
        # 帧选择条件：当 j % factor < 1 时选择该帧
        # 这确保了均匀的帧采样以达到目标帧率
        if j % factor < 1:
            current_frame = frames[j, :, :, :].unsqueeze(0)  # 添加批次维度
            
            if final_frames is None:
                # 初始化最终帧张量
                final_frames = current_frame
            else:
                # 在批次维度上拼接新帧
                final_frames = torch.cat((final_frames, current_frame), 0)

    # 将最终帧移动到GPU进行模型推理
    final_frames = final_frames.cuda()

    # ==================== 模型推理和特征提取 ====================
    # 使用微调的CLIP模型进行前向传播
    # 返回三个输出：犯规预测、动作预测、视频特征
    out_off, out_act, video_features = model(final_frames)

    # ==================== 预测结果解析 ====================
    # 获取犯规严重程度的预测结果（取概率最大的类别）
    preds_sev = torch.argmax(out_off.detach().cpu(), 0)
    
    # 获取动作类型的预测结果
    preds_act = torch.argmax(out_act.detach().cpu(), 0)

    # ==================== 结果格式化 ====================
    values = {}
    
    # 将动作预测索引转换为可读的动作名称
    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
    
    # 根据犯规严重程度索引设置犯规信息
    if preds_sev.item() == 0:
        # 无犯规
        values["Offence"] = "No offence"
        values["Severity"] = ""
    elif preds_sev.item() == 1:
        # 轻微犯规（通常不需要出牌）
        values["Offence"] = "Offence"
        values["Severity"] = "1.0"
    elif preds_sev.item() == 2:
        # 中等犯规（黄牌级别）
        values["Offence"] = "Offence"
        values["Severity"] = "3.0"
    elif preds_sev.item() == 3:
        # 严重犯规（红牌级别）
        values["Offence"] = "Offence"
        values["Severity"] = "5.0"

    # ==================== 特征后处理 ====================
    # 将CLIP特征转换为时空融合特征
    # 转换为float16格式以节省内存，同时保持足够的精度
    video_clip_features = get_spatio_temporal_features(
        video_features.numpy().astype("float16")
    )

    return video_clip_features, values


if __name__ == '__main__':
    """
    批量视频特征提取主程序
    =====================
    
    程序功能：
    对SoccerNet-XFoul数据集进行批量特征提取和预测，为X-VARS系统的训练和评估提供预处理数据。
    
    处理流程：
    1. 解析命令行参数（数据集路径和模型权重路径）
    2. 初始化CLIP图像处理器和微调模型
    3. 遍历Train/Valid/Test三个数据集分割
    4. 对每个视频剪辑进行特征提取和预测
    5. 保存特征文件(.pkl)和预测结果(.json)
    
    输出文件：
    - 特征文件：PRE_CLIP_feature_clip_X.pkl（每个视频剪辑一个）
    - 预测文件：predictionsTrain_clip.json, predictionsValid_clip.json, predictionsTest_clip.json
    
    数据集结构：
    dataset/
    ├── Train/
    │   ├── action_0/
    │   │   ├── clip_1.mp4
    │   │   ├── clip_2.mp4
    │   │   └── clip_3.mp4
    │   └── action_1/
    │       └── ...
    ├── Valid/
    └── Test/
    """

    # ==================== 命令行参数解析 ====================
    parser = ArgumentParser(
        description='X-VARS视频特征提取工具 - 批量处理SoccerNet-XFoul数据集', 
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--path_dataset', 
        required=True, 
        type=str, 
        help='SoccerNet-XFoul数据集的根目录路径，应包含Train/Valid/Test子目录'
    )
    parser.add_argument(
        '--path_weights', 
        required=True, 
        type=str, 
        help='微调CLIP模型权重文件的路径（.pth或.pt格式）'
    )

    args = parser.parse_args()

    # ==================== 初始化配置 ====================
    path_dataset = args.path_dataset  # 数据集根目录
    splits_list = ["Train", "Valid", "Test"]  # 数据集分割列表
    
    # 初始化CLIP图像预处理器
    # 使用与训练时相同的预处理配置，确保特征提取的一致性
    print("正在初始化CLIP图像预处理器...")
    image_processor = CLIPImageProcessor.from_pretrained(
        'openai/clip-vit-large-patch14', 
        torch_dtype=torch.float16
    )

    # ==================== 模型加载和配置 ====================
    print("正在加载微调的CLIP模型...")
    model = CLIPNetwork().cuda()  # 初始化自定义CLIP网络并移至GPU
    
    path_weights = args.path_weights
    print(f"从 {path_weights} 加载模型权重...")
    
    # 加载预训练权重
    checkpoint = torch.load(path_weights, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'])

    # 设置为评估模式，禁用dropout和batch normalization的训练行为
    model.eval()
    print("模型加载完成，进入评估模式")

    # ==================== 批量处理各数据集分割 ====================
    for split in splits_list:
        print(f"\n开始处理 {split} 数据集...")
        
        # 构建当前分割的数据集路径
        path_dataset_split = os.path.join(path_dataset, split)
        
        # 统计当前分割中的动作文件夹数量
        folders = 0
        for _, dirnames, _ in os.walk(path_dataset_split):
            folders += len(dirnames)
            break  # 只计算第一层目录
        
        print(f"发现 {folders} 个动作文件夹")

        # 初始化结果存储结构
        data = {}
        data["Set"] = split  # 记录数据集分割名称
        actions = {}  # 存储所有动作的预测结果
        
        # ==================== 遍历每个动作文件夹 ====================
        for i in range(folders):
            # 构建动作文件夹路径（约定命名：action_0, action_1, ...）
            path_clip = os.path.join(path_dataset_split, f"action_{i}")
            print(f"处理文件夹: {path_clip}")

            # ==================== 处理clip_1.mp4 ====================
            clip1_path = os.path.join(path_clip, "clip_1.mp4")
            if os.path.exists(clip1_path):
                print("  正在处理 clip_1.mp4...")
                
                # 提取视频特征和预测结果
                features, values = get_features(clip1_path)

                # 保存特征到pickle文件
                feature_file = os.path.join(path_clip, "PRE_CLIP_feature_clip_1.pkl")
                with open(feature_file, 'wb') as f:
                    pickle.dump(features, f)

                # 记录预测结果
                actions[feature_file] = values

            # ==================== 处理clip_2.mp4 ====================
            clip2_path = os.path.join(path_clip, "clip_2.mp4")
            if os.path.exists(clip2_path):
                print("  正在处理 clip_2.mp4...")
                
                features, values = get_features(clip2_path)

                feature_file = os.path.join(path_clip, "PRE_CLIP_feature_clip_2.pkl")
                with open(feature_file, 'wb') as f:
                    pickle.dump(features, f)
                
                actions[feature_file] = values

            # ==================== 处理clip_3.mp4 ====================
            clip3_path = os.path.join(path_clip, "clip_3.mp4")
            if os.path.exists(clip3_path):
                print("  正在处理 clip_3.mp4...")
                
                features, values = get_features(clip3_path)

                feature_file = os.path.join(path_clip, "PRE_CLIP_feature_clip_3.pkl")
                with open(feature_file, 'wb') as f:
                    pickle.dump(features, f)
                
                actions[feature_file] = values

        # ==================== 保存当前分割的预测结果 ====================
        data["Actions"] = actions
        
        # 构建输出JSON文件名
        output_file = f"predictions{split}_clip.json"
        print(f"正在保存预测结果到 {output_file}...")
        
        with open(output_file, "w") as outfile:
            json.dump(data, outfile, indent=2, ensure_ascii=False)
        
        print(f"{split} 数据集处理完成，共处理 {len(actions)} 个视频剪辑")

    print("\n所有数据集处理完成！")
    print("输出文件：")
    print("- 特征文件：每个视频剪辑对应一个 .pkl 文件")
    print("- 预测文件：predictionsTrain_clip.json, predictionsValid_clip.json, predictionsTest_clip.json")
