import os
import json
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from video_chatgpt.model import VideoChatGPTLlamaForCausalLM
from video_chatgpt.utils import disable_torch_init
import logging, time

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s")


def _sanitize_tokenizer_dir(tokenizer_dir: str):
    """Ensure local tokenizer.json contains required keys for transformers, notably 'added_tokens'.
    This mutates the local file in-place to avoid runtime KeyError when loading completely offline tokenizers
    that were exported without the optional field.
    """
    try:
        if not tokenizer_dir or not os.path.isdir(tokenizer_dir):
            return
        tokenizer_json_path = os.path.join(tokenizer_dir, 'tokenizer.json')
        if not os.path.isfile(tokenizer_json_path):
            return
        with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        changed = False
        if 'added_tokens' not in data:
            data['added_tokens'] = []
            changed = True
        # Some older exports also miss 'normalizer' or 'pre_tokenizer', but those are model-specific.
        if changed:
            with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            logger.info(f"[init] 修正 tokenizer.json 缺失字段: added_tokens（已写回） | {tokenizer_json_path}")
    except Exception as e:
        logger.warning(f"[init] 清理 tokenizer 目录失败（忽略继续）: {e}")
from video_chatgpt.constants import *
import torch

# 加载视频
def load_video(vis_path, n_clips=1, num_frm=100):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # 使用VideoReader加载视频
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr) #获取视频总帧数

    # 目前，这个函数只支持1个片段
    assert n_clips == 1

    # 计算要提取的总帧数
    total_num_frm = min(total_frame_num, num_frm)
    # 获取要提取的帧索引
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # 提取帧作为numpy数组
    img_array = vr.get_batch(frame_idx).asnumpy()
    # 设置目标图像高度和宽度
    target_h, target_w = 224, 224
    # 如果图像形状与目标不符，则调整大小
    if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

    # 重塑数组以匹配片段和帧的数量
    img_array = img_array.reshape(
        (n_clips, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
    # 将numpy数组转换为PIL Image对象
    clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(total_num_frm)]

    return clip_imgs


# 获取序列帧
def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # 计算每个片段的大小，从中提取帧
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # 计算每个片段的开始和结束索引
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # 将每个片段的中间索引添加到列表中
        seq.append((start + end) // 2)

    return seq


# 初始化模型
def initialize_model(
    model_name,
    projection_path=None,
    token=None,
    local_tokenizer_dir=None,
    local_clip_dir=None,
    local_files_only=False,
):
    """
    初始化模型、tokenizer、图像处理器等组件
    
    Args:
        model_name: 模型名称或路径
        projection_path: 投影层权重路径
        token: Hugging Face 访问令牌
    """
    # 禁用初始torch操作
    t0_all = time.time()
    disable_torch_init()
    logger.info("[init] disable_torch_init 完成")

    # 将模型名称转换为用户路径
    model_name = os.path.expanduser(model_name)

    # 加载tokenizer（优先本地）
    tokenizer = None
    try:
        t0 = time.time()
        tried_paths = []
        prefer_slow = False
        def _maybe_force_slow_tokenizer(dir_path: str) -> bool:
            try:
                if not dir_path:
                    return False
                tj = os.path.join(dir_path, 'tokenizer.json')
                if not os.path.isfile(tj):
                    # 没有 tokenizer.json，通常表示仅提供了 tokenizer.model（LLaMA 类模型常见）
                    # 为避免 Fast tokenizer 强依赖 tokenizer.json，这里强制改用慢速分词器
                    return True
                with open(tj, 'r', encoding='utf-8') as f:
                    json.load(f)
                return False
            except Exception as ex:
                logger.warning(f"[init] 本地 tokenizer.json 无法解析，将使用慢速分词器并忽略该文件: {ex}")
                # 将坏文件改名，避免底层优先选择它
                try:
                    os.replace(tj, tj + '.bak')
                    logger.info(f"[init] 已重命名损坏的 tokenizer.json -> {tj+'.bak'}")
                except Exception:
                    pass
                return True
        if local_tokenizer_dir and os.path.isdir(local_tokenizer_dir):
            tried_paths.append(local_tokenizer_dir)
            logger.info(f"[init] 尝试从本地目录加载 tokenizer | {local_tokenizer_dir}")
            _sanitize_tokenizer_dir(local_tokenizer_dir)
            prefer_slow = _maybe_force_slow_tokenizer(local_tokenizer_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                local_tokenizer_dir,
                trust_remote_code=True,
                local_files_only=True,
                use_fast=not prefer_slow,
            )
        if tokenizer is None:
            # 尝试从模型目录加载（有些权重包含tokenizer）
            model_dir = os.path.expanduser(model_name)
            tried_paths.append(model_dir)
            try:
                _sanitize_tokenizer_dir(model_dir)
                prefer_slow = _maybe_force_slow_tokenizer(model_dir)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    local_files_only=local_files_only,
                    use_fast=not prefer_slow,
                )
            except Exception:
                tokenizer = None
        if tokenizer is None:
            # 作为兜底：需要用户已预先缓存或放置离线资源，否则会失败
            repo_fallback = 'liuhaotian/llava-v1.5-7b'
            tried_paths.append(repo_fallback)
            logger.info(f"[init] 尝试从缓存加载 tokenizer（离线） | {repo_fallback}")
            prefer_slow = _maybe_force_slow_tokenizer(repo_fallback)
            tokenizer = AutoTokenizer.from_pretrained(
                repo_fallback,
                trust_remote_code=True,
                local_files_only=True,
                use_fast=not prefer_slow,
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"[init] tokenizer 加载完成 | elapsed={time.time()-t0:.2f}s")
    except Exception as e:
        logger.error(f"[init] 加载 tokenizer 失败。请设置本地目录 LOCAL_TOKENIZER_DIR 指向包含 tokenizer.json/vocab 的路径。错误: {e}")
        raise

    # 加载模型，传递token
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        device_map = {"": 0}
        torch_dtype = torch.float16
    else:
        device_map = "auto"  # 使用CPU
        torch_dtype = torch.float32  # CPU上使用float32更稳定
    logger.info(f"[init] 设备设置 | cuda={torch.cuda.is_available()} | device_map={device_map} | dtype={torch_dtype}")
    
    t0 = time.time()
    logger.info(f"[init] 开始加载主模型 | model_dir={model_name}")
    try:
        model = VideoChatGPTLlamaForCausalLM.from_pretrained(
            os.path.expanduser(model_name),
            low_cpu_mem_usage=True,
            torch_dtype=torch_dtype,
            use_cache=True,
            device_map=device_map,
            local_files_only=local_files_only,
            token=token if not local_files_only else None,
        )
    except Exception as e:
        logger.error(f"[init] 本地主模型加载失败，请确认路径包含 config.json 与权重: {e}")
        raise
    logger.info(f"[init] 主模型加载完成 | elapsed={time.time()-t0:.2f}s")

    # 加载图像处理器，传递token
    if torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    t0 = time.time()
    # 加载 CLIPImageProcessor（优先本地目录）
    clip_src = local_clip_dir or getattr(model.config, 'mm_vision_tower', None) or 'openai/clip-vit-large-patch14'
    logger.info(f"[init] 开始加载 CLIPImageProcessor | src={clip_src}")
    try:
        image_processor = CLIPImageProcessor.from_pretrained(
            clip_src,
            torch_dtype=torch_dtype,
            local_files_only=True,
        )
    except Exception as e:
        logger.error(f"[init] 加载 CLIPImageProcessor 失败，请将 CLIP 模型离线文件放置到: {clip_src} 或设置 LOCAL_CLIP_DIR。错误: {e}")
        raise
    logger.info(f"[init] CLIPImageProcessor 加载完成 | elapsed={time.time()-t0:.2f}s")

    # 设置使用开始和结束标记用于视频
    mm_use_vid_start_end = True

    # 添加标记到tokenizer
    tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_vid_start_end:
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)

    # 调整模型token嵌入的大小
    t0 = time.time()
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"[init] 调整 token 嵌入大小完成 | new_vocab={len(tokenizer)} | elapsed={time.time()-t0:.2f}s")

    # 从投影路径加载权重，在调整token嵌入大小后
    if projection_path:
        t0 = time.time()
        status = model.load_state_dict(torch.load(projection_path, map_location='cpu'), strict=False)
        try:
            unexpected = getattr(status, 'unexpected_keys', [])
            missing = getattr(status, 'missing_keys', [])
            if unexpected:
                logger.warning(f"[init] 投影层权重存在未期望键: {unexpected}")
            if missing:
                logger.warning(f"[init] 投影层权重缺失键: {missing}")
        except Exception:
            pass
        logger.info(f"[init] 投影层权重加载完成 | path={projection_path} | elapsed={time.time()-t0:.2f}s")

    """print("######### NEXT ###########")
    for name, param in model.state_dict().items():
        if name == "model.mm_projector.weight" or name == "model.mm_projector.base_layer.weight":
            print(param)"""


    # 将模型设置为评估模式并移动到GPU
    # Ensure pad token is valid to avoid generation warnings
    try:
        if getattr(tokenizer, 'pad_token_id', None) in (None, -1):
            tokenizer.pad_token = tokenizer.eos_token
        if getattr(model.config, 'pad_token_id', -1) in (None, -1):
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model, 'generation_config', None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"[init] pad_token_id 设置为 {tokenizer.pad_token_id}")
    except Exception as e:
        logger.warning(f"[init] 设置 pad_token_id 失败（忽略继续）: {e}")

    t0 = time.time()
    model = model.eval()
    logger.info(f"[init] 主模型切换到 eval 模式 | elapsed={time.time()-t0:.2f}s")
    #model = model.cuda()

    vision_tower_name = local_clip_dir or "openai/clip-vit-large-patch14"

    # 加载视觉塔，传递token
    if torch.cuda.is_available():
        torch_dtype = torch.float16
        t0 = time.time()
        logger.info(f"[init] 开始加载 CLIP 视觉塔（GPU） | src={vision_tower_name}")
        try:
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            ).cuda()
        except Exception as e:
            logger.error(f"[init] 加载 CLIP 视觉塔失败（GPU）。请将离线文件放置到: {vision_tower_name}。错误: {e}")
            raise
        logger.info(f"[init] CLIP 视觉塔加载完成（GPU） | elapsed={time.time()-t0:.2f}s")
    else:
        torch_dtype = torch.float32
        t0 = time.time()
        logger.info(f"[init] 开始加载 CLIP 视觉塔（CPU） | src={vision_tower_name}")
        try:
            vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        except Exception as e:
            logger.error(f"[init] 加载 CLIP 视觉塔失败（CPU）。请将离线文件放置到: {vision_tower_name}。错误: {e}")
            raise
        logger.info(f"[init] CLIP 视觉塔加载完成（CPU） | elapsed={time.time()-t0:.2f}s")
    vision_tower = vision_tower.eval()
    logger.info("[init] 视觉塔切换到 eval 模式")

    # 配置视觉模型
    vision_config = model.get_model().vision_config
    vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0]
    vision_config.use_vid_start_end = mm_use_vid_start_end
    if mm_use_vid_start_end:
        vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

    # 设置视频token长度
    video_token_len = 356

    logger.info(f"[init] 全部初始化完成 | total_elapsed={time.time()-t0_all:.2f}s")
    return model, vision_tower, tokenizer, image_processor, video_token_len
