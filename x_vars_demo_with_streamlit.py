"""
X-VARS 智能足球裁判系统 - Streamlit 演示界面
===============================================

本文件是X-VARS（Explainable Video Assistant Referee System）的Streamlit演示界面，
提供基于视频的足球犯规分析和智能对话功能。

主要功能：
1. 视频上传和预处理
2. 使用CLIP视觉编码器提取视频特征
3. 基于X-VARS模型进行足球场景理解和分析
4. 集成RAG（检索增强生成）系统，结合FIFA足球规则知识库
5. 提供简洁的对话式界面进行视频分析

技术特点：
- 强制CPU模式运行，确保最大兼容性
- 支持RAG知识库增强的专业足球规则分析
- 实时系统资源监控和性能优化
- 简洁的回答模式，避免冗余输出

运行方式:
    streamlit run x_vars_demo_with_streamlit.py
"""

import os
import contextlib
# 在导入任何 transformers 相关包前，禁止启用 TensorFlow/Flax 分支，避免触发不兼容的 TF/protobuf 依赖
# 这些环境变量设置确保只使用PyTorch后端，避免TensorFlow相关的依赖冲突
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")  # 禁用TensorFlow后端
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")  # 禁用Flax后端
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # 禁用tokenizer并行化，避免警告

# 优化HuggingFace缓存配置（使用新的环境变量）
# TRANSFORMERS_CACHE已废弃，改用HF_HOME
if "HF_HOME" not in os.environ and "TRANSFORMERS_CACHE" in os.environ:
    # 如果设置了旧的TRANSFORMERS_CACHE，迁移到HF_HOME
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
    print(f"[配置] 已从TRANSFORMERS_CACHE迁移到HF_HOME: {os.environ['HF_HOME']}")

# 禁用CodeCarbon回调（如果存在，避免导入错误）
os.environ.setdefault("DISABLE_CODECARBON", "1")  # 禁用CodeCarbon以减少依赖

from typing import List, Optional, Tuple
import logging

# Streamlit界面框架
import streamlit as st
import pandas as pd
# PyTorch深度学习框架
import torch
import torch.nn.functional as F
# 视频处理库
from decord import VideoReader, cpu
from decord import bridge as decord_bridge

# X-VARS模型相关导入
from video_chatgpt.eval.model_utils import (
    initialize_model,  # 模型初始化工具
)
from video_chatgpt.model.utils import KeywordsStoppingCriteria  # 生成停止条件
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle  # 对话模板
from video_chatgpt.constants import (
    DEFAULT_VIDEO_PATCH_TOKEN,  # 默认视频补丁标记
    DEFAULT_VID_START_TOKEN,    # 默认视频开始标记
    DEFAULT_VID_END_TOKEN,      # 默认视频结束标记
)
from visual_encoder.clip_model import CLIPNetwork  # CLIP视觉编码器

# RAG相关导入 - 简化版本，不依赖FAISS
# RAG（检索增强生成）系统用于结合足球规则知识库提供更专业的分析
try:
    from sentence_transformers import SentenceTransformer  # 句子编码器
    RAG_AVAILABLE = True
    print("RAG依赖加载成功（简化版）")
except ImportError as e:
    # logger还没有定义，所以直接打印
    print(f"RAG依赖未安装: {e}")
    RAG_AVAILABLE = False


# ----------------------------
# 配置 & 日志
# ----------------------------

# 日志配置：设置日志格式，包含时间戳、日志级别、模块名、函数名和消息内容
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("xvars")  # 创建专用的日志记录器

# 模型路径配置
BASE_MODEL_DIR = os.path.join("X-VARS", "X-VARS_weights")  # X-VARS主模型权重目录
LOCAL_TOKENIZER_DIR = os.path.join("X-VARS", "tokenizer")  # 分词器目录
LOCAL_CLIP_DIR = os.path.join("visual_encoder", "clip-vit-large-patch14")  # CLIP视觉编码器目录
CLIP_CLASSIFIER_WEIGHTS = os.path.join("visual_encoder", "14_model.pth.tar")  # CLIP分类头权重文件

# 对话配置
DEFAULT_CONV_MODE = "xvars_football_v1"  # 足球裁判专用（与 custom_dataset / 论文一致）
DEFAULT_VIDEO_TOKEN_LEN = 300  # create_features: 44 temporal + 256 spatial

# RAG 配置 - 简化版本
VECTOR_STORE_DIR = "vector_store"  # 向量存储目录
KNOWLEDGE_BASE_FILE = "vector_store/2023-2024中文版足球規則.txt"  # 足球规则知识库文件
SIMPLE_EMBEDDINGS_FILE = "vector_store/embeddings.pkl"  # 使用简单的pickle存储嵌入向量
DEFAULT_RAG_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 默认优先GPU

# 对话记录自动保存配置
CHAT_HISTORY_DIR = "chat_history"  # 对话记录保存目录
AUTO_SAVE_CHAT = True  # 是否自动保存对话记录

# 详细日志配置
DETAILED_LOG_DIR = "detailed_logs"  # 详细日志保存目录
ENABLE_DETAILED_LOGGING = True  # 是否启用详细日志记录
def _format_bytes(num_bytes: int) -> str:
    """
    将字节数转换为人类可读的格式
    
    参数:
        num_bytes (int): 字节数
        
    返回:
        str: 格式化后的字符串，如 "1.5GB"
    """
    try:
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if num_bytes < 1024.0:
                return f"{num_bytes:3.1f}{unit}"
            num_bytes /= 1024.0
    except Exception:
        return str(num_bytes)

def log_memory(prefix: str = "Mem"):
    """
    记录系统内存使用情况
    
    参数:
        prefix (str): 日志前缀，默认为"Mem"
        
    功能:
        - 记录CPU内存使用情况（RSS - Resident Set Size）
        - 如果GPU可用，记录GPU内存分配和保留情况
        - 使用psutil库获取系统信息
    """
    try:
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss  # 获取进程实际使用的物理内存
        logger.info(f"{prefix} | CPU RSS={_format_bytes(rss)}")
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)  # GPU已分配内存
            reserved = torch.cuda.memory_reserved(0)    # GPU保留内存
            logger.info(f"{prefix} | GPU allocated={_format_bytes(allocated)} reserved={_format_bytes(reserved)}")
    except Exception:
        pass

@contextlib.contextmanager
def step_timer(step_name: str):
    """
    步骤计时器上下文管理器
    
    参数:
        step_name (str): 步骤名称，用于日志记录
        
    功能:
        - 记录步骤开始和结束时间
        - 监控每个步骤前后的内存使用情况
        - 捕获和记录异常信息
        - 提供详细的性能分析日志
        
    使用示例:
        with step_timer("模型加载"):
            model = load_model()
    """
    import time as _time
    logger.info(f"[START] {step_name}")
    log_memory(prefix=f"{step_name} BEFORE")  # 记录步骤开始前的内存使用
    _t0 = _time.time()  # 记录开始时间
    try:
        yield  # 执行被监控的代码块
    except Exception as e:
        _dt = _time.time() - _t0  # 计算执行时间
        logger.error(f"[FAIL ] {step_name} | elapsed={_dt:.2f}s | err={e}")
        log_memory(prefix=f"{step_name} FAIL")  # 记录失败时的内存使用
        raise  # 重新抛出异常
    else:
        _dt = _time.time() - _t0  # 计算执行时间
        logger.info(f"[DONE ] {step_name} | elapsed={_dt:.2f}s")
        log_memory(prefix=f"{step_name} AFTER")  # 记录步骤完成后的内存使用


def _evenly_sample(items: List, max_count: int) -> List:
    """
    均匀采样函数，从列表中均匀选择指定数量的元素
    
    参数:
        items (List): 原始列表
        max_count (int): 最大采样数量
        
    返回:
        List: 采样后的列表
        
    功能:
        - 如果原始列表长度小于等于max_count，直接返回原列表
        - 否则按均匀间隔采样，确保采样结果在时间轴上均匀分布
    """
    if len(items) <= max_count:
        return items
    step = len(items) / max_count  # 计算采样步长
    return [items[int(i * step)] for i in range(max_count)]  # 均匀采样


def _get_device() -> torch.device:
    """
    获取计算设备
    
    返回:
        torch.device: CPU设备
        
    注意:
        本系统强制使用CPU模式，确保最大兼容性和稳定性
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_gpu_memory_management():
    """
    配置GPU内存管理，启用共享内存和内存池
    
    返回:
        bool: 配置是否成功
        
    功能:
        - 设置GPU内存使用比例
        - 启用内存池和缓存优化
        - 配置CUDA计算后端优化
        - 设置内存分配策略
        
    注意:
        本系统主要使用CPU模式，此函数为GPU优化预留
    """
    if not torch.cuda.is_available():
        return False
        
    try:
        # 启用统一内存管理（如果支持）
        if hasattr(torch.cuda, 'set_memory_fraction'):
            # 允许GPU使用更多内存（包括系统内存）
            torch.cuda.set_memory_fraction(0.95)  # 使用95%的可用内存
            
        # 启用内存池和缓存优化
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()  # 清空GPU缓存
            
        # 设置内存分配策略
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'cuda'):
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32精度
            if hasattr(torch.backends.cuda, 'matmul'):
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True  # 允许FP16精度
                
        # 尝试启用CUDA内存池
        try:
            # 设置内存分配器
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 设置最大分割大小为128MB
        except Exception:
            pass
            
        logger.info("GPU内存管理配置完成")
        return True
    except Exception as e:
        logger.warning(f"GPU内存管理配置失败: {e}")
        return False

def get_memory_info():
    """
    获取详细的内存使用信息
    
    返回:
        dict: 包含系统和GPU内存使用信息的字典
        
    功能:
        - 获取系统内存使用情况（总量、可用、已用、使用率）
        - 如果GPU可用，获取GPU内存使用情况
        - 提供完整的内存监控信息用于性能分析
    """
    info = {}
    try:
        import psutil
        # 系统内存信息
        memory = psutil.virtual_memory()
        info['system_total'] = memory.total      # 系统总内存
        info['system_available'] = memory.available  # 可用内存
        info['system_used'] = memory.used        # 已使用内存
        info['system_percent'] = memory.percent  # 内存使用率
        
        # GPU内存信息（如果可用）
        if torch.cuda.is_available():
            info['gpu_allocated'] = torch.cuda.memory_allocated(0)  # GPU已分配内存
            info['gpu_reserved'] = torch.cuda.memory_reserved(0)    # GPU保留内存
            info['gpu_total'] = torch.cuda.get_device_properties(0).total_memory  # GPU总内存
            info['gpu_free'] = info['gpu_total'] - info['gpu_reserved']  # GPU可用内存
            info['gpu_percent'] = (info['gpu_reserved'] / info['gpu_total']) * 100  # GPU内存使用率
            
        return info
    except Exception as e:
        logger.error(f"获取内存信息失败: {e}")
        return {}


@st.cache_resource(show_spinner=False)
def load_rag_encoder(device: str = DEFAULT_RAG_DEVICE):
    """
    缓存的RAG编码器加载函数
    
    参数:
        device (str): 目标设备，默认为CPU
        
    返回:
        SentenceTransformer: 加载的编码器模型，失败时返回None
        
    功能:
        - 使用Streamlit缓存机制，避免重复加载
        - 加载多语言句子编码器用于文本相似度计算
        - 自动处理设备切换和错误恢复
        - 支持CPU和GPU模式（本系统主要使用CPU）
    """
    try:
        logger.info(f"正在加载RAG编码器到设备: {device}")
        # 加载多语言句子编码器，支持中英文文本处理
        encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)
        
        # 确保模型在指定设备上
        if device.lower() == 'cpu':
            encoder = encoder.to('cpu')
            logger.info("RAG编码器已设置为CPU模式")
        elif device.lower().startswith('cuda') and torch.cuda.is_available():
            encoder = encoder.to(device)
            logger.info(f"RAG编码器已设置为GPU模式: {device}")
        else:
            # 如果指定GPU但不可用，回退到CPU
            encoder = encoder.to('cpu')
            logger.warning(f"指定设备{device}不可用，回退到CPU")
        
        logger.info(f"RAG编码器加载完成，当前设备: {encoder.device}")
        return encoder
    except Exception as e:
        logger.error(f"RAG编码器加载失败: {e}")
        return None


class FootballRAGSystem:
    """
    足球规则知识库RAG系统 - 简化版本
    
    这是一个专门为足球规则分析设计的检索增强生成系统，
    结合FIFA官方足球规则知识库，为视频分析提供专业的规则依据。
    
    主要功能：
    1. 加载和索引足球规则文本
    2. 支持向量搜索和文本匹配两种检索模式
    3. 智能关键词扩展和语义相似度计算
    4. 提供足球规则相关的上下文信息
    """
    
    def __init__(self, device: str = DEFAULT_RAG_DEVICE):
        """
        初始化RAG系统
        
        参数:
            device (str): 计算设备，默认为CPU
        """
        self.encoder = None          # 句子编码器
        self.embeddings = None       # 存储所有文本的嵌入向量
        self.texts = []              # 文本内容列表
        self.metadata = []           # 文本元数据列表
        self.initialized = False     # 初始化状态标志
        self.device = device         # RAG系统使用的设备
        
    def _load_encoder(self, device: str = DEFAULT_RAG_DEVICE):
        """
        加载句子编码器
        
        参数:
            device (str): 目标设备
            
        返回:
            SentenceTransformer: 编码器模型
        """
        return load_rag_encoder(device)
    
    def initialize(self):
        """
        初始化RAG系统
        
        返回:
            bool: 初始化是否成功
            
        功能:
            - 检查RAG依赖是否可用
            - 验证知识库文件是否存在
            - 加载或构建向量索引
            - 设置系统为就绪状态
        """
        if not RAG_AVAILABLE:
            logger.warning("RAG依赖未安装，跳过初始化")
            return False
            
        try:
            # 先检查知识库文件是否存在
            logger.info(f"检查知识库文件: {KNOWLEDGE_BASE_FILE}")
            if not os.path.exists(KNOWLEDGE_BASE_FILE):
                logger.error(f"知识库文件不存在: {KNOWLEDGE_BASE_FILE}")
                logger.info(f"当前工作目录: {os.getcwd()}")
                logger.info(f"vector_store目录内容: {os.listdir('vector_store') if os.path.exists('vector_store') else '目录不存在'}")
                return False
            else:
                # 检查文件大小和内容预览
                file_size = os.path.getsize(KNOWLEDGE_BASE_FILE)
                logger.info(f"知识库文件存在，大小: {file_size} 字节")
                
                # 读取文件前几行预览
                try:
                    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                        preview = f.read(200)
                        logger.info(f"文件内容预览: {preview[:100]}...")
                except Exception as e:
                    logger.error(f"文件读取预览失败: {e}")
            
            # 加载编码器
            logger.info(f"正在加载Sentence Transformer编码器到设备: {self.device}")
            self.encoder = self._load_encoder(self.device)
            if self.encoder is None:
                logger.warning("编码器加载失败，将使用纯文本匹配模式")
                # 如果编码器加载失败，我们仍然可以使用文本匹配
                # 不返回False，而是继续
            else:
                logger.info(f"编码器加载成功，设备: {self.device}")
                
            # 尝试加载现有索引
            logger.info("检查是否存在现有嵌入向量...")
            if self._load_existing_index():
                self.initialized = True
                logger.info("RAG系统初始化完成（使用现有向量存储）")
                return True
            
            # 构建新索引
            logger.info("现有嵌入向量不存在，开始构建新的向量存储...")
            if self._build_index():
                self.initialized = True
                logger.info("RAG系统初始化完成（构建新向量存储）")
                return True
            else:
                logger.error("向量存储构建失败")
                
        except Exception as e:
            logger.error(f"RAG系统初始化失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            
        return False
    
    def _align_embeddings_device(self) -> None:
        """Ensure stored embeddings live on the same device as the encoder/query.

        embeddings.pkl is typically saved on CPU. If DEFAULT_RAG_DEVICE is CUDA,
        cos_sim(query_cuda, emb_cpu) raises and search silently falls back to
        keyword matching — looking like \"RAG works\" while vector search does not.
        """
        if self.embeddings is None:
            return
        import torch

        if not isinstance(self.embeddings, torch.Tensor):
            return
        target = torch.device(self.device if self.device else "cpu")
        if self.embeddings.device != target:
            self.embeddings = self.embeddings.to(target)
            logger.info(f"嵌入向量已对齐到设备: {target}")

    def _load_existing_index(self):
        """加载现有的嵌入向量"""
        try:
            if os.path.exists(SIMPLE_EMBEDDINGS_FILE):
                import pickle
                with open(SIMPLE_EMBEDDINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.texts = data['texts']
                    self.metadata = data.get('metadata', [])
                    self.embeddings = data['embeddings']  # 直接存储tensor或list
                self._align_embeddings_device()
                logger.info(f"加载现有嵌入向量: {len(self.texts)}个文本片段")
                return True
        except Exception as e:
            logger.error(f"加载现有嵌入向量失败: {e}")
            
        return False
    
    def _build_index(self):
        """构建新的嵌入向量存储 - 简化版本"""
        try:
            if not os.path.exists(KNOWLEDGE_BASE_FILE):
                logger.error(f"知识库文件不存在: {KNOWLEDGE_BASE_FILE}")
                return False
            
            # 读取足球规则文本
            with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分割文本为段落
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # 进一步分割长段落
            self.texts = []
            self.metadata = []
            
            for i, para in enumerate(paragraphs):
                if len(para) > 500:  # 长段落分割
                    sentences = [s.strip() for s in para.split('。') if s.strip()]
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) > 400:
                            if current_chunk:
                                self.texts.append(current_chunk + "。")
                                self.metadata.append({'source': 'football_rules', 'paragraph': i})
                            current_chunk = sentence
                        else:
                            current_chunk += sentence + "。"
                    if current_chunk:
                        self.texts.append(current_chunk)
                        self.metadata.append({'source': 'football_rules', 'paragraph': i})
                else:
                    self.texts.append(para)
                    self.metadata.append({'source': 'football_rules', 'paragraph': i})
            
            if not self.texts:
                logger.error("没有有效的文本内容")
                return False
            
            # 生成嵌入向量 - 如果编码器可用
            if self.encoder is not None:
                logger.info(f"正在为{len(self.texts)}个文本片段生成嵌入向量... (设备: {self.device})")
                
                try:
                    # 尝试生成嵌入向量，指定输出格式为tensor
                    embeddings = self.encoder.encode(
                        self.texts, 
                        batch_size=32, 
                        show_progress_bar=False,
                        convert_to_numpy=False,  # 强制不转换为numpy
                        convert_to_tensor=True   # 保持tensor格式
                    )
                    logger.info(f"成功生成tensor格式嵌入向量: {type(embeddings)}")
                    # 保存嵌入向量
                    self.embeddings = embeddings
                    
                except Exception as e:
                    logger.warning(f"tensor格式生成失败，尝试备用方法: {e}")
                    
                    # 备用方法：逐个处理文本
                    embeddings_list = []
                    for i, text in enumerate(self.texts):
                        if i % 20 == 0:  # 每20个文档显示一次进度
                            logger.info(f"处理进度: {i+1}/{len(self.texts)}")
                        
                        try:
                            # 单个文本编码
                            single_embedding = self.encoder.encode(
                                [text], 
                                convert_to_numpy=False,
                                convert_to_tensor=True
                            )
                            embeddings_list.append(single_embedding[0])  # 取出单个embedding
                            
                        except Exception as single_e:
                            logger.error(f"文本 {i} 编码失败: {single_e}")
                            # 如果这个也失败，就跳过这个文本
                            continue
                    
                    if embeddings_list:
                        # 手动堆叠tensor
                        import torch
                        embeddings = torch.stack(embeddings_list)
                        logger.info(f"备用方法成功，生成了{len(embeddings_list)}个嵌入向量")
                        # 保存嵌入向量
                        self.embeddings = embeddings
                    else:
                        logger.warning("所有嵌入向量生成方法都失败，将使用纯文本匹配模式")
                        self.embeddings = None  # 标记为文本匹配模式
            else:
                logger.info("编码器不可用，使用纯文本匹配模式")
                self.embeddings = None  # 标记为文本匹配模式
            
            logger.info(f"知识库准备完成: {len(self.texts)}个文档，模式: {'向量搜索' if self.embeddings is not None else '文本匹配'}")
            
            # 保存到文件
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
            
            import pickle
            import torch

            # Persist embeddings on CPU so the pickle is portable across machines/devices
            emb_to_save = self.embeddings
            if isinstance(emb_to_save, torch.Tensor):
                emb_to_save = emb_to_save.detach().to("cpu")
            save_data = {
                'texts': self.texts,
                'metadata': self.metadata,
                # Must use self.embeddings: local `embeddings` is undefined when encoder is None
                'embeddings': emb_to_save,
            }
            # Keep runtime embeddings on the configured device for search
            self._align_embeddings_device()
            
            with open(SIMPLE_EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"嵌入向量存储完成: {len(self.texts)}个文本片段")
            return True
            
        except Exception as e:
            logger.error(f"构建嵌入向量失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> list:
        """搜索相关知识 - 支持向量搜索和文本匹配两种模式"""
        if not self.initialized:
            return []
        
        # 如果没有嵌入向量或编码器，直接使用文本匹配
        if self.embeddings is None or self.encoder is None:
            logger.debug("[RAG] 使用文本匹配模式")
            return self._fallback_search(query, top_k)
            
        try:
            logger.debug(f"[RAG] 检索查询: {query[:50]}... (设备: {self.device})")
            
            # 编码查询 - 使用安全的方法
            try:
                query_embedding = self.encoder.encode(
                    [query],
                    convert_to_numpy=False,
                    convert_to_tensor=True
                )
                logger.debug(f"查询编码成功: {type(query_embedding)}")
            except Exception as e:
                logger.error(f"查询编码失败，使用备用搜索: {e}")
                return self._fallback_search(query, top_k)
            
            # 使用简单的相似度计算
            try:
                from sentence_transformers.util import cos_sim
                import torch

                # Align devices (query may be on CUDA while pickle embeddings are on CPU)
                self._align_embeddings_device()
                if isinstance(query_embedding, torch.Tensor) and isinstance(
                    self.embeddings, torch.Tensor
                ):
                    if query_embedding.device != self.embeddings.device:
                        query_embedding = query_embedding.to(self.embeddings.device)

                # 计算相似度
                scores = cos_sim(query_embedding, self.embeddings)
                
                # 获取最相似的文档
                scores = scores.flatten()  # 展平为1D
                
                # 获取topk结果
                top_results = scores.argsort(descending=True)[:top_k]
                
                results = []
                for idx in top_results:
                    score = float(scores[idx])
                    if score > 0.3:  # 过滤低相关性结果
                        results.append({
                            'text': self.texts[idx],
                            'score': score,
                            'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                        })
                
                logger.debug(f"[RAG] 检索到{len(results)}条相关结果")
                return results
                
            except Exception as e:
                logger.error(f"相似度计算失败: {e}")
                # 如果相似度计算失败，使用备用搜索
                return self._fallback_search(query, top_k)
            
        except Exception as e:
            logger.error(f"知识检索失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            
            # 备用方案：使用更简单的字符串匹配
            try:
                logger.info("使用备用的字符串匹配搜索")
                return self._fallback_search(query, top_k)
            except Exception as fallback_e:
                logger.error(f"备用搜索也失败: {fallback_e}")
                return []
    
    def _fallback_search(self, query: str, top_k: int = 3) -> list:
        """备用的简单字符串匹配搜索 - 智能关键词匹配"""
        query_lower = query.lower().strip()
        results = []
        
        # 足球相关关键词扩展
        football_keywords = {
            '犯规': ['犯规', '违规', '违例', '不当行为'],
            '黄牌': ['黄牌', '警告', '黄卡'],  
            '红牌': ['红牌', '罚下', '红卡', '驱逐'],
            '手球': ['手球', '用手', '手触球'],
            '越位': ['越位', '离位'],
            '点球': ['点球', '十二码', '罚球点'],
            '任意球': ['任意球', '自由球'],
            '角球': ['角球', '角踢'],
            '界外球': ['界外球', '边线球'],
            '冲撞': ['冲撞', '冲击', '撞击'],
            '踢': ['踢', '踢击']
        }
        
        # 扩展查询词
        expanded_query_words = query_lower.split()
        for word in query_lower.split():
            for category, synonyms in football_keywords.items():
                if word in synonyms:
                    expanded_query_words.extend([s for s in synonyms if s not in expanded_query_words])
        
        logger.debug(f"[RAG] 扩展查询词: {expanded_query_words}")
        
        for i, text in enumerate(self.texts):
            text_lower = text.lower()
            score = 0.0
            
            # 1. 精确短语匹配 (最高分)
            if query_lower in text_lower:
                score += 0.9
                
            # 2. 扩展关键词匹配
            matched_expanded = sum(1 for word in expanded_query_words if word in text_lower)
            if len(expanded_query_words) > 0:
                score += 0.6 * (matched_expanded / len(expanded_query_words))
            
            # 3. 原始关键词匹配
            query_words = query_lower.split()
            text_words = text_lower.split()
            matched_original = sum(1 for word in query_words if word in text_words)
            if len(query_words) > 0:
                score += 0.4 * (matched_original / len(query_words))
                
            # 4. 部分匹配加分
            for word in query_words:
                if len(word) > 2:  # 只对长词进行部分匹配
                    if any(word in text_word for text_word in text_words):
                        score += 0.1
            
            # 5. 足球规则特定模式匹配
            if any(keyword in text_lower for keyword in ['第', '条', '章', '规则', '条款']):
                if any(qword in text_lower for qword in query_words):
                    score += 0.2  # 规则条文优先
            
            if score > 0.2:  # 降低阈值，允许更多匹配
                results.append({
                    'text': text,
                    'score': score,
                    'metadata': self.metadata[i] if i < len(self.metadata) else {}
                })
        
        # 按得分排序并返回topk
        results.sort(key=lambda x: x['score'], reverse=True)
        logger.info(f"文本匹配搜索找到{len(results)}条结果，查询词: '{query_lower}'")
        
        # 如果没有找到足够的结果，尝试更宽松的匹配
        if len(results) < top_k and len(query_words) > 1:
            logger.debug("结果不足，尝试单词匹配...")
            for word in query_words:
                if len(word) > 1:  # 跳过过短的词
                    for i, text in enumerate(self.texts):
                        if word in text.lower() and not any(r['text'] == text for r in results):
                            results.append({
                                'text': text,
                                'score': 0.3,
                                'metadata': self.metadata[i] if i < len(self.metadata) else {}
                            })
        
        return results[:top_k]
    
    def get_stats(self):
        """获取RAG系统统计信息"""
        return {
            'initialized': self.initialized,
            'num_documents': len(self.texts) if self.initialized else 0,
            'knowledge_file_exists': os.path.exists(KNOWLEDGE_BASE_FILE),
            'index_file_exists': os.path.exists(SIMPLE_EMBEDDINGS_FILE),
            'has_encoder': self.encoder is not None,
            'has_embeddings': self.embeddings is not None,
            'search_mode': '向量搜索' if (self.embeddings is not None and self.encoder is not None) else '文本匹配'
        }


@st.cache_resource(show_spinner=False)
def load_all_models(
    base_model_dir: str,
    local_tokenizer_dir: Optional[str],
    local_clip_dir: Optional[str],
    clip_classifier_weights: Optional[str],
    load_clip_classifier: bool = False,
):
    """
    离线加载主模型、Tokenizer、CLIP 预处理器，返回 HF 视觉塔与可选自定义分类头
    
    参数:
        base_model_dir (str): X-VARS主模型权重目录
        local_tokenizer_dir (Optional[str]): 本地分词器目录
        local_clip_dir (Optional[str]): 本地CLIP模型目录
        clip_classifier_weights (Optional[str]): CLIP分类头权重文件路径
        
    返回:
        tuple: (model, vision_tower_hf, clip_net, tokenizer, image_processor)
            - model: X-VARS主语言模型
            - vision_tower_hf: CLIP视觉编码器
            - clip_net: CLIP分类头网络
            - tokenizer: 文本分词器
            - image_processor: 图像预处理器
            
    功能:
        - 使用Streamlit缓存机制避免重复加载
        - 直接加载到CPU，避免GPU→CPU的冗余迁移
        - 强制所有模型使用CPU和float32精度
        - 加载预训练的CLIP分类头权重
        - 提供完整的模型推理管道
        - 自动释放GPU内存（如果之前有占用）
    """
    try:
        target_device = _get_device()
        use_gpu = target_device.type == "cuda"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"模型加载目标设备: {target_device}")

        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if not use_gpu:
            # 仅在CPU模式时隐藏GPU，避免initialize_model内部上GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            # HuggingFace token：离线优先，不强制需要
            hf_token = os.environ.get("HF_TOKEN", None)

            with step_timer("initialize_model"):
                model, vision_tower_hf, tokenizer, image_processor, _ = initialize_model(
                    model_name=base_model_dir,  # 主模型路径
                    projection_path=None,  # 投影层路径（可选）
                    token=hf_token,  # HuggingFace token
                    local_tokenizer_dir=local_tokenizer_dir,  # 本地分词器
                    local_clip_dir=local_clip_dir,  # 本地CLIP模型
                    local_files_only=True,  # 仅使用本地文件
                )

            device = target_device
            target_dtype = torch.float16 if use_gpu else torch.float32
            logger.info(f"统一模型设备与精度: device={device}, dtype={target_dtype}")

            with step_timer("set models to target device"):
                model = model.to(device, dtype=target_dtype)
                vision_tower_hf = vision_tower_hf.to(device, dtype=target_dtype)
            logger.info("主模型和视觉编码器已完成设备对齐")

            clip_net = None
            # 可选：自定义 CLIP 分类头（当前主流程并不依赖，默认跳过以节省显存）
            if load_clip_classifier:
                with step_timer("init CLIPNetwork (classifier head)"):
                    clip_net = CLIPNetwork().to(device, dtype=target_dtype).eval()

                if clip_classifier_weights and os.path.isfile(clip_classifier_weights):
                    with step_timer("load CLIP classifier weights"):
                        try:
                            ckpt = torch.load(
                                clip_classifier_weights,
                                map_location=str(device),
                                weights_only=True,
                            )
                        except TypeError:
                            logger.warning("PyTorch版本不支持weights_only，使用传统加载方式")
                            ckpt = torch.load(clip_classifier_weights, map_location=str(device))

                        state = ckpt.get("state_dict", ckpt)
                        clip_net.load_state_dict(state, strict=False)

            return model, vision_tower_hf, clip_net, tokenizer, image_processor
        finally:
            # 恢复原始CUDA设置
            if original_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        import traceback

        logger.error(f"详细错误: {traceback.format_exc()}")
        st.error(f"模型加载失败: {str(e)}")
        st.error("请检查模型文件是否存在，或者尝试重启应用程序")
        st.stop()


def build_prompt(question: str, model, video_token_len: int) -> str:
    """
    构建包含视频信息的对话提示词
    
    参数:
        question (str): 用户问题
        model: X-VARS模型对象
        video_token_len (int): 视频token长度
        
    返回:
        str: 格式化的对话提示词
        
    功能:
        - 根据模型配置决定是否使用视频开始/结束标记
        - 将用户问题与视频token组合
        - 使用对话模板格式化提示词
        - 为模型生成提供完整的上下文
    """
    # 检查模型是否使用视频开始/结束标记
    if getattr(model.get_model().vision_config, "use_vid_start_end", True):
        # 使用完整的视频标记序列
        vid_tokens = (
            DEFAULT_VID_START_TOKEN                    # 视频开始标记
            + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len  # 视频内容标记
            + DEFAULT_VID_END_TOKEN                    # 视频结束标记
        )
    else:
        # 仅使用视频内容标记
        vid_tokens = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # 组合用户问题和视频标记
    qs = question.strip() + "\n" + vid_tokens
    
    # 使用对话模板构建完整的提示词
    conv = conv_templates[DEFAULT_CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)  # 添加用户消息
    conv.append_message(conv.roles[1], None)  # 预留助手回复位置
    return conv.get_prompt()


def make_video_features_from_clip_hidden(features: torch.Tensor, target_length: int = 300) -> torch.Tensor:
    """
    将帧级隐藏状态 (T, S, C) 转为指定长度的时空特征序列
    
    参数:
        features (torch.Tensor): CLIP提取的帧级特征，形状为 (T, S, C)
            - T: 时间维度（帧数）
            - S: 空间维度（空间patch数量）
            - C: 特征通道数
        target_length (int): 目标特征序列长度，默认300
        
    返回:
        torch.Tensor: 时空融合特征，形状为 (target_length, C)
        
    功能:
        - 提取时间特征：对空间维度求平均，保留时序信息
        - 提取空间特征：对时间维度求平均，保留空间布局
        - 合并时空特征：将时间和空间特征拼接
        - 长度调整：通过插值或填充调整到目标长度
    """
    assert features.dim() == 3, "expect features of shape (T, S, C)"
    t, s, c = features.shape

    # 确保输入是float32
    features = features.to(dtype=torch.float32)

    # 时序特征：对空间维度求平均 (T, C)
    # 这样可以保留每个时间步的全局空间信息
    temporal_tokens = features.mean(dim=1)  
    
    # 空间特征：对时间维度求平均 (S, C)  
    # 这样可以获得每个空间位置在整个视频序列中的平均表示
    spatial_tokens = features.mean(dim=0)
    
    # 合并时空特征
    combined_features = torch.cat([temporal_tokens, spatial_tokens], dim=0)  # (T+S, C)
    current_length = combined_features.shape[0]
    
    # 调整到目标长度
    if current_length > target_length:
        # 如果特征太长，进行插值缩减
        indices = torch.linspace(0, current_length - 1, target_length, dtype=torch.long, device=features.device)
        video_features = combined_features[indices]
        logger.debug(f"特征序列缩减: {current_length} → {target_length}")
    elif current_length < target_length:
        # 如果特征太短，进行零填充
        padding_size = target_length - current_length
        padding = torch.zeros(padding_size, c, device=features.device, dtype=torch.float32)  # 明确float32
        video_features = torch.cat([combined_features, padding], dim=0)
        logger.debug(f"特征序列扩展: {current_length} → {target_length} (填充{padding_size})")
    else:
        video_features = combined_features
        logger.debug(f"特征序列长度匹配: {target_length}")
    
    # 确保输出是float32
    video_features = video_features.to(dtype=torch.float32)
    return video_features  # (target_length, C)


def compute_clip_hidden_states(
    vision_model,
    frames_tensor: torch.Tensor,  # (N, 3, 224, 224)
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    分批计算 CLIP 最后第二层隐藏状态并拼接为 (T, S, C)
    
    参数:
        vision_model: CLIP视觉编码器模型
        frames_tensor (torch.Tensor): 视频帧张量，形状为 (N, 3, 224, 224)
        batch_size (int): 批处理大小
        device (torch.device): 计算设备
        
    返回:
        torch.Tensor: CLIP隐藏状态，形状为 (T, S, C)
            - T: 时间维度（帧数）
            - S: 空间维度（空间patch数量）
            - C: 特征通道数
            
    功能:
        - 分批处理视频帧以避免内存溢出
        - 提取CLIP倒数第二层的隐藏状态
        - 移除CLS token，只保留空间patch特征
        - 强制使用CPU模式确保稳定性
    """
    compute_device = device
    use_gpu = compute_device.type == "cuda"
    compute_dtype = torch.float16 if use_gpu else torch.float32
    vision_model = vision_model.to(compute_device, dtype=compute_dtype).eval()

    feats: list[torch.Tensor] = []
    import time as _time
    
    # 分批处理视频帧
    for start in range(0, frames_tensor.shape[0], batch_size):
        end = min(start + batch_size, frames_tensor.shape[0])
        batch = frames_tensor[start:end].to(compute_device, dtype=compute_dtype)
        t0 = _time.time()
        
        with torch.no_grad():
            if use_gpu:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outs = vision_model(batch, output_hidden_states=True)
            else:
                outs = vision_model(batch, output_hidden_states=True)
        
        # 提取倒数第二层隐藏状态，移除CLS token
        hs = outs.hidden_states[-2][:, 1:]  # (B, S, C)
        feats.append(hs)
        
        # 清理中间变量以节省内存
        del outs, hs, batch
        logger.debug(f"[CLIP] chunk {start}:{end} done | elapsed={_time.time()-t0:.3f}s")
    
    # 拼接所有批次的特征
    features = torch.cat(feats, dim=0)  # (T, S, C)
    return features


def _get_seq_frames(total_num_frames: int, desired_num_frames: int) -> List[int]:
    """
    计算均匀采样的帧索引序列
    
    参数:
        total_num_frames (int): 视频总帧数
        desired_num_frames (int): 期望采样的帧数
        
    返回:
        List[int]: 采样帧的索引列表
        
    功能:
        - 将视频时间轴均匀分割为desired_num_frames个区间
        - 每个区间选择中间位置的帧
        - 确保采样帧在时间轴上均匀分布
    """
    seg_size = float(total_num_frames - 1) / desired_num_frames  # 计算每个区间的长度
    seq = []
    for i in range(desired_num_frames):
        start = int(round(seg_size * i))      # 区间开始位置
        end = int(round(seg_size * (i + 1)))  # 区间结束位置
        seq.append((start + end) // 2)        # 选择区间中点
    return seq


def load_torch_frames_without_numpy(path: str, num_frm: int = 96, target_size: Tuple[int, int] = (224, 224), device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    使用 decord 的 Torch 桥加载视频，避免 numpy/PIL，直接返回已预处理并标准化的张量
    
    参数:
        path (str): 视频文件路径
        num_frm (int): 最大采样帧数，默认96
        target_size (Tuple[int, int]): 目标帧尺寸，默认(224, 224)
        device (torch.device): 目标设备，默认CPU
        
    返回:
        torch.Tensor: 预处理后的视频帧张量，形状为 (N, 3, 224, 224)
        
    功能:
        - 使用decord直接加载视频为PyTorch张量
        - 均匀采样指定数量的帧
        - 调整帧尺寸到目标大小
        - 应用CLIP标准化（ImageNet均值和标准差）
        - 强制使用CPU和float32精度
    """
    # 设置decord使用PyTorch后端
    decord_bridge.set_bridge('torch')
    vr = VideoReader(path, ctx=cpu(0))  # 使用CPU上下文
    total = len(vr)  # 获取视频总帧数
    total_num_frm = min(total, num_frm)  # 确定实际采样帧数
    frame_idx = _get_seq_frames(total, total_num_frm)  # 计算均匀采样索引

    # 获取采样帧，形状为 (N, H, W, 3) uint8
    batch = vr.get_batch(frame_idx)
    if batch.dtype != torch.uint8:
        batch = batch.to(torch.uint8)

    # 转为 CHW 格式，并缩放到目标尺寸
    batch = batch.permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
    batch = batch.float() / 255.0  # 归一化到[0,1]
    if batch.shape[-2:] != target_size:
        # 使用双线性插值调整尺寸
        batch = F.interpolate(batch, size=target_size, mode='bilinear', align_corners=False)

    # CLIP 标准化（使用ImageNet均值和标准差）
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=batch.device).view(1, 3, 1, 1)
    batch = (batch - mean) / std

    # 移动到目标设备
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    batch = batch.to(device, dtype=target_dtype)
        
    return batch


def run_generation(
    model,
    tokenizer,
    video_spatio_temporal_features: torch.Tensor,  # (1, N, C), same dtype/device as model
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    debug: bool = False,
) -> str:
    """
    运行模型生成，基于视频特征和文本提示生成回答
    
    参数:
        model: X-VARS语言模型
        tokenizer: 文本分词器
        video_spatio_temporal_features (torch.Tensor): 视频时空特征，形状为 (1, N, C)
        prompt (str): 输入提示词
        temperature (float): 生成温度，控制随机性
        max_new_tokens (int): 最大生成token数
        debug (bool): 是否启用调试模式
        
    返回:
        str: 生成的文本回答
        
    功能:
        - 将文本提示转换为token序列
        - 设置停止条件以控制生成长度
        - 结合视频特征进行多模态生成
        - 解码生成的token序列为文本
        - 清理输出文本中的停止标记
    """
    device = _get_device()
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    inputs = tokenizer([prompt])  # 将提示词转换为token
    input_ids = torch.as_tensor(inputs.input_ids, device=device, dtype=torch.long)

    # 停止标记处理
    conv = conv_templates[DEFAULT_CONV_MODE]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # 确保视频特征在正确的设备和数据类型
    if video_spatio_temporal_features is not None:
        video_spatio_temporal_features = video_spatio_temporal_features.to(device, dtype=target_dtype)

    # 设置生成参数
    gen_kwargs = dict(
        input_ids=input_ids,                                    # 输入token序列
        video_spatio_temporal_features=video_spatio_temporal_features,  # 视频特征
        do_sample=temperature > 0,                             # 是否使用采样
        temperature=float(temperature),                        # 生成温度
        max_new_tokens=int(max_new_tokens),                    # 最大生成长度
        stopping_criteria=[stopping_criteria],                 # 停止条件
        return_dict_in_generate=bool(debug),                   # 是否返回详细信息
        output_scores=bool(debug),                             # 是否输出分数
    )
    
    # 执行生成
    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)

    # 处理生成结果
    if debug and hasattr(outputs, "sequences"):
        logger.debug(f"[GEN] steps={getattr(outputs, 'num_generated_tokens', 'n/a')} | sequences shape={tuple(outputs.sequences.shape)}")
        output_ids = outputs.sequences
    else:
        output_ids = outputs
    
    # 解码生成的token为文本
    outputs_text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    outputs_text = outputs_text.strip().rstrip(stop_str).strip()  # 清理停止标记
    return outputs_text


def model_generate_simple(
    model,
    tokenizer,
    video_spatio_temporal_features: torch.Tensor,
    prompt: str,
    temperature: float,
    max_new_tokens: int,
    do_sample: bool,
    debug: bool = False,
) -> str:
    """简化的生成函数，无超时限制。"""
    import time as _time
    
    device = _get_device()
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    inputs = tokenizer([prompt])
    input_ids = torch.as_tensor(inputs.input_ids, device=device, dtype=torch.long)

    # Llama 位置编码上限通常为 2048；超长 RAG 上下文会导致空输出/索引错误。
    # 视频补丁 token 在序列尾部（靠近 ASSISTANT:），必须保留尾部。
    max_pos = int(getattr(getattr(model, "config", None), "max_position_embeddings", 2048) or 2048)
    if input_ids.shape[1] > max_pos:
        keep_tail = min(320, max_pos // 2)
        keep_head = max_pos - keep_tail
        logger.warning(
            f"[GEN] Prompt 超长 {input_ids.shape[1]}>{max_pos}，截断为 head={keep_head}+tail={keep_tail}"
        )
        input_ids = torch.cat([input_ids[:, :keep_head], input_ids[:, -keep_tail:]], dim=1)
    
    # 计算prompt长度
    prompt_length = input_ids.shape[1]
    logger.info(f"[GEN] Prompt Token化长度: {prompt_length} tokens")
    logger.info(f"[GEN] Input IDs形状: {input_ids.shape}")
    
    # 记录input_ids的前后部分（用于调试）
    if prompt_length > 0:
        logger.info(f"[GEN] Input IDs前20个: {input_ids[0, :20].tolist()}")
        if prompt_length > 20:
            logger.info(f"[GEN] Input IDs后20个: {input_ids[0, -20:].tolist()}")
        
        # 解码前20个token看看内容
        try:
            decoded_prefix = tokenizer.decode(input_ids[0, :min(20, prompt_length)], skip_special_tokens=False)
            logger.info(f"[GEN] Prompt前20个tokens解码: {decoded_prefix[:200]}")
        except Exception as e:
            logger.warning(f"[GEN] 解码前20个tokens失败: {e}")

    # Stop token handling
    conv = conv_templates[DEFAULT_CONV_MODE]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    logger.debug(f"[GEN] 停止词: '{stop_str}'")

    # 确保视频特征在正确的设备和数据类型
    if video_spatio_temporal_features is not None:
        video_spatio_temporal_features = video_spatio_temporal_features.to(device, dtype=target_dtype)
        logger.debug(f"[GEN] 视频特征已转换为: {video_spatio_temporal_features.device}, {video_spatio_temporal_features.dtype}")

    # 确保视频特征在正确的设备和数据类型
    if video_spatio_temporal_features is not None:
        video_spatio_temporal_features = video_spatio_temporal_features.to(device, dtype=target_dtype)
        logger.info(f"[GEN] 视频特征形状: {video_spatio_temporal_features.shape}, 设备: {video_spatio_temporal_features.device}, 类型: {video_spatio_temporal_features.dtype}")
    else:
        logger.error("[GEN] 警告：视频特征为None！这可能导致模型生成错误内容！")

    # 构建生成参数：当do_sample=False时（贪心解码），不传递temperature参数
    # 验证视频特征
    if video_spatio_temporal_features is None:
        logger.error("[GEN] 严重错误：视频特征为None，模型可能无法正确理解视频内容！")
        raise ValueError("视频特征不能为None")
    
    # 验证视频特征形状
    if len(video_spatio_temporal_features.shape) != 3 or video_spatio_temporal_features.shape[0] != 1:
        logger.error(f"[GEN] 严重错误：视频特征形状不正确！期望 (1, N, C)，实际 {video_spatio_temporal_features.shape}")
        raise ValueError(f"视频特征形状不正确: {video_spatio_temporal_features.shape}")
    
    gen_kwargs = dict(
        input_ids=input_ids,
        video_spatio_temporal_features=video_spatio_temporal_features,
        do_sample=bool(do_sample),
        max_new_tokens=int(max_new_tokens),
        stopping_criteria=[stopping_criteria],
        return_dict_in_generate=True,
        output_scores=bool(debug),
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # 只有在使用采样时才添加temperature参数
    if do_sample:
        gen_kwargs['temperature'] = float(temperature)
    
    logger.info(f"[GEN] 生成参数: do_sample={do_sample}, max_new_tokens={max_new_tokens}, video_features_shape={video_spatio_temporal_features.shape}")
    start_time = _time.time()
    logger.debug(f"[GEN] 开始生成: max_tokens={max_new_tokens}, sampling={do_sample}")
    
    with torch.inference_mode():
        outputs = model.generate(**gen_kwargs)
    
    generation_time = _time.time() - start_time
    
    # 分析生成结果
    if hasattr(outputs, "sequences"):
        output_ids = outputs.sequences
        actual_length = output_ids.shape[1] - prompt_length
        
        # 检查生成统计（使用info级别，方便查看）
        logger.info("[GEN] ========== 生成统计 ==========")
        logger.info(f"[GEN] 实际生成tokens: {actual_length}/{max_new_tokens}")
        logger.info(f"[GEN] 生成时间: {generation_time:.2f}s")
        if generation_time > 0:
            logger.info(f"[GEN] 生成速度: {actual_length/generation_time:.1f} tokens/s")
        logger.info(f"[GEN] 完整序列长度: {output_ids.shape[1]} tokens (prompt: {prompt_length} + generated: {actual_length})")
        
        # 记录生成的token序列（前50个和后50个）
        generated_ids = output_ids[0, prompt_length:]
        if len(generated_ids) > 0:
            logger.info(f"[GEN] 生成的Token IDs前50个: {generated_ids[:50].tolist()}")
            if len(generated_ids) > 50:
                logger.info(f"[GEN] 生成的Token IDs后50个: {generated_ids[-50:].tolist()}")
        
        # 分析结束原因
        if actual_length >= max_new_tokens * 0.99:
            logger.info(f"[GEN] 结束原因: 达到max_new_tokens限制 ({max_new_tokens})")
        else:
            logger.info("[GEN] 结束原因: 遇到停止词或EOS token")
            
    else:
        output_ids = outputs
        logger.warning(f"[GEN] 未返回详细生成信息，outputs类型: {type(outputs)}")
        if hasattr(outputs, 'shape'):
            logger.info(f"[GEN] Outputs形状: {outputs.shape}")
    
    # 解码输出
    generated_token_ids = output_ids[:, prompt_length:]
    outputs_text = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
    
    # 记录解码信息
    logger.info("[GEN] ========== 解码结果 ==========")
    logger.info(f"[GEN] 原始解码输出长度: {len(outputs_text)} 字符")
    logger.info(f"[GEN] 原始解码输出内容:\n{outputs_text}")
    
    # 清理输出文本
    outputs_text = outputs_text.strip()
    
    # 移除停止词（如果存在）
    if stop_str and outputs_text.endswith(stop_str):
        outputs_text = outputs_text[:-len(stop_str)].strip()
        logger.info(f"[GEN] 移除了停止词: '{stop_str}'")
    
    # 移除其他常见的停止模式
    common_stops = ["\n\nHuman:", "\n\nAssistant:", "<|im_end|>", "</s>"]
    for stop_pattern in common_stops:
        if stop_pattern in outputs_text:
            outputs_text = outputs_text.split(stop_pattern)[0].strip()
            logger.info(f"[GEN] 移除了停止模式: '{stop_pattern}'")
    
    logger.info(f"[GEN] 清理后输出长度: {len(outputs_text)} 字符")
    logger.info(f"[GEN] 清理后输出内容:\n{outputs_text}")
    logger.info("[GEN] ==============================")
    
    return outputs_text


def extract_frames_to_tensor(temp_video_path: str, image_processor, max_frames: int = 32, device: Optional[torch.device] = None) -> torch.Tensor:
    """读取视频（无 numpy/PIL）→ 直接得到标准化张量 (N, 3, 224, 224)，并按需均匀采样。"""
    target_device = device or _get_device()
    batch = load_torch_frames_without_numpy(temp_video_path, num_frm=96, device=target_device)
    # 均匀子采样到 max_frames
    if batch.shape[0] > max_frames:
        idx = _evenly_sample(list(range(batch.shape[0])), max_frames)
        batch = batch[idx]
    return batch


def initialize_chat_state():
    """
    初始化聊天状态和会话变量
    
    功能:
        - 初始化视频处理相关的会话状态
        - 设置聊天历史记录
        - 初始化RAG系统状态
        - 确保所有必要的会话变量都存在
    """
    # 视频处理相关状态
    if 'video_features' not in st.session_state:
        st.session_state.video_features = None  # 存储提取的视频特征
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None      # 当前处理的视频路径
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False  # 视频是否已处理标志
    if 'processing_info' not in st.session_state:
        st.session_state.processing_info = {}   # 视频处理信息
    if "rag_last_results" not in st.session_state:
        st.session_state.rag_last_results = []
    if "rag_last_query" not in st.session_state:
        st.session_state.rag_last_query = ""
    
    # 聊天相关状态
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []      # 聊天历史记录
    
    # 初始化RAG系统
    if 'rag_system' not in st.session_state:
        rag_device = st.session_state.get('rag_device', DEFAULT_RAG_DEVICE)
        st.session_state.rag_system = FootballRAGSystem(device=rag_device)
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False  # RAG系统初始化状态
    
    # 初始化自动保存配置
    if 'auto_save_chat' not in st.session_state:
        st.session_state.auto_save_chat = AUTO_SAVE_CHAT  # 默认启用自动保存

def add_chat_message(role: str, content: str, metadata: dict = None):
    """
    添加聊天消息到历史记录
    
    参数:
        role (str): 消息角色，'user' 或 'assistant'
        content (str): 消息内容
        metadata (dict): 消息元数据，如处理时间等
        
    功能:
        - 将消息添加到聊天历史
        - 如果是助手消息，自动保存对话记录到本地文件
    """
    import time as _time
    message = {
        'role': role,  # 'user' 或 'assistant'
        'content': content,
        'timestamp': _time.time(),  # 消息时间戳
        'metadata': metadata or {}  # 消息元数据
    }
    st.session_state.chat_history.append(message)
    
    # 如果是助手消息，自动保存对话记录
    if role == 'assistant':
        # 检查是否启用自动保存（优先使用session state中的配置）
        auto_save_enabled = st.session_state.get('auto_save_chat', AUTO_SAVE_CHAT)
        if auto_save_enabled:
            try:
                auto_save_chat_history()
            except Exception as e:
                logger.warning(f"自动保存对话记录时出错: {e}")
                # 静默失败，不干扰用户操作

def display_chat_history():
    """
    显示聊天历史记录
    
    功能:
        - 如果无聊天历史，显示欢迎信息
        - 遍历聊天历史，使用不同的头像显示用户和助手消息
        - 显示处理时间等元数据信息
    """
    if not st.session_state.chat_history:
        st.markdown("💭 **开始您的视频分析对话吧！**")
        st.markdown("上传视频后，您可以就视频内容进行多轮对话。")
        return
    
    # 显示聊天消息
    for i, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            # 用户消息
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(message['content'])
        else:
            # 助手消息
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message['content'])
                # 显示处理时间
                if 'processing_time' in message['metadata']:
                    st.caption(f"⏱️ 处理时间: {message['metadata']['processing_time']:.2f}秒")

def process_video(uploaded_file, example_path):
    """
    处理视频并提取特征（只执行一次）
    
    参数:
        uploaded_file: 用户上传的视频文件
        example_path: 示例视频文件路径
        
    返回:
        bool: 处理是否成功
        
    功能:
        - 将上传的视频文件保存到临时位置
        - 提取视频帧并预处理
        - 使用CLIP编码器提取视频特征
        - 将特征保存到会话状态中
        - 清理临时文件
    """
    import tempfile
    import time as _time
    
    try:
        # 获取设备和参数
        device = _get_device()
        video_token_len = st.session_state.get('video_token_len', 200)  # 视频token长度
        max_frames = st.session_state.get('max_frames', 16)              # 最大采样帧数
        clip_bs = st.session_state.get('clip_bs', 1)  # 使用较小批次
        
        # 将上传文件写到临时位置
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name
        else:
            temp_path = example_path

        progress_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_placeholder.info(f"🔄 开始处理视频（设备: {device}）...")
        progress_bar.progress(10)
        
        with step_timer("完整视频处理"):
            import time as _time
            t0 = _time.time()
            
            # 提取帧
            progress_placeholder.info("📹 正在提取视频帧...")
            frames_tensor = extract_frames_to_tensor(temp_path, None, max_frames=max_frames, device=device)
            progress_bar.progress(45)
            
            # 选择 CLIP 计算设备（侧边栏可切换）
            clip_choice = st.session_state.get("clip_device_choice", "GPU" if torch.cuda.is_available() else "CPU")
            clip_device = torch.device("cuda" if (clip_choice == "GPU" and torch.cuda.is_available()) else "cpu")
            
            # 处理视频特征
            progress_placeholder.info("🧠 正在提取视频特征...")
            video_spatio_temporal = process_clip_features(frames_tensor, clip_device, clip_bs, video_token_len)
            progress_bar.progress(90)
            
            t1 = _time.time()
            processing_time = t1 - t0
            
            # 保存到session state
            st.session_state.video_features = video_spatio_temporal
            st.session_state.video_processed = True
            st.session_state.processing_info = {
                'frames': frames_tensor.shape[0],        # 提取的帧数
                'time': processing_time,                 # 处理时间
                'device': str(clip_device),              # 使用的设备
                'sequence_length': video_spatio_temporal.shape[1]  # 特征序列长度
            }
            
            progress_placeholder.success(f"✅ 视频处理完成！提取了{frames_tensor.shape[0]}帧，耗时{processing_time:.1f}秒")
            progress_bar.progress(100)
            
            # 清理临时文件
            if uploaded_file is not None:
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            
    except Exception as e:
        logger.error(f"视频处理失败: {str(e)}")
        st.error(f"视频处理失败: {str(e)}")
        return False
        
    return True

def process_clip_features(frames_tensor, clip_device, clip_bs, video_token_len):
    """
    提取CLIP特征的核心函数
    
    参数:
        frames_tensor: 视频帧张量
        clip_device: CLIP计算设备
        clip_bs: CLIP批处理大小
        video_token_len: 视频token长度
        
    返回:
        torch.Tensor: 视频时空特征，形状为 (1, N, C)
        
    功能:
        - 加载X-VARS模型组件
        - 分批处理视频帧以节省内存
        - 提取CLIP倒数第二层隐藏状态
        - 将帧级特征转换为时空序列特征
        - 根据设备自动选择精度（GPU: float16, CPU: float32）
    """
    # 获取全局模型
    model, vision_tower_hf, _, _, _ = load_all_models(
        BASE_MODEL_DIR,
        LOCAL_TOKENIZER_DIR if os.path.isdir(LOCAL_TOKENIZER_DIR) else None,
        LOCAL_CLIP_DIR if os.path.isdir(LOCAL_CLIP_DIR) else None,
        CLIP_CLASSIFIER_WEIGHTS if os.path.isfile(CLIP_CLASSIFIER_WEIGHTS) else None,
    )
    
    device = clip_device
    use_gpu = device.type == "cuda"
    tgt_dtype = torch.float16 if use_gpu else torch.float32
    
    # 分批计算隐藏状态
    total_steps = (frames_tensor.shape[0] + int(clip_bs) - 1) // int(clip_bs)
    collected: list[torch.Tensor] = []
    
    logger.info(f"使用{device.type.upper()}模式处理CLIP特征")
    vision_model = vision_tower_hf.to(clip_device, dtype=tgt_dtype).eval()
    # 分批处理视频帧
    for step, start in enumerate(range(0, frames_tensor.shape[0], int(clip_bs))):
        end = min(start + int(clip_bs), frames_tensor.shape[0])
        batch = frames_tensor[start:end].to(clip_device)
        
        batch = batch.to(dtype=tgt_dtype)
        
        with torch.no_grad():
            if use_gpu:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outs = vision_model(batch, output_hidden_states=True)
            else:
                outs = vision_model(batch, output_hidden_states=True)
        
        # 提取倒数第二层隐藏状态，移除CLS token
        hs = outs.hidden_states[-2][:, 1:]
        collected.append(hs.to(dtype=tgt_dtype))
        
        # 清理中间变量
        del outs, hs, batch
        logger.debug(f"处理批次 {step+1}/{total_steps}")
    
    # 拼接特征
    frame_features = torch.cat(collected, dim=0)

    # 转换为时空序列特征（300 tokens 时用论文 create_features 算法）
    if int(video_token_len) == 300:
        from rag_experiment.paper_features import paper_spatio_temporal_features

        video_spatio_temporal = paper_spatio_temporal_features(
            frame_features.to(device=device, dtype=tgt_dtype)
        )
    else:
        video_spatio_temporal = make_video_features_from_clip_hidden(
            frame_features.to(device=device, dtype=tgt_dtype),
            target_length=int(video_token_len),
        )
    
    # 保持与模型一致的数据类型
    video_spatio_temporal = video_spatio_temporal.to(device=device, dtype=tgt_dtype)
    
    logger.info(f"CLIP特征提取完成，形状: {video_spatio_temporal.shape}, 类型: {video_spatio_temporal.dtype}")
    return video_spatio_temporal.unsqueeze(0)  # (1, N, C)

def save_detailed_generation_log(
    question: str,
    enhanced_question: str,
    knowledge_context: str,
    full_prompt: str,
    video_features_info: dict,
    model_params: dict,
    raw_output: str,
    final_output: str,
    processing_time: float,
    use_rag: bool,
    rag_results: list = None
):
    """
    保存详细的生成日志到文件
    
    参数:
        question: 用户原始问题
        enhanced_question: 增强后的问题
        knowledge_context: RAG检索到的知识上下文
        full_prompt: 完整的prompt（包含视频token）
        video_features_info: 视频特征信息
        model_params: 模型参数
        raw_output: 模型原始输出
        final_output: 最终返回的输出
        processing_time: 处理时间
        use_rag: 是否使用RAG
    """
    if not ENABLE_DETAILED_LOGGING:
        return
    
    try:
        import json
        from datetime import datetime
        
        # 确保日志目录存在
        os.makedirs(DETAILED_LOG_DIR, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log_filename = f"generation_log_{timestamp}.json"
        log_filepath = os.path.join(DETAILED_LOG_DIR, log_filename)
        
        # 准备日志数据
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'user_question': question,
            'enhanced_question': enhanced_question,
            'rag_enabled': use_rag,
            'rag_results': rag_results or [],  # RAG检索结果列表
            'knowledge_context': knowledge_context,
            'full_prompt': full_prompt,
            'prompt_length': len(full_prompt),
            'prompt_token_count': len(full_prompt.split()) if full_prompt else 0,  # 粗略估计
            'video_features': video_features_info,
            'model_parameters': model_params,
            'raw_model_output': raw_output,
            'final_output': final_output,
            'output_length': len(final_output),
            'processing_time_seconds': processing_time,
            'video_info': st.session_state.get('processing_info', {}),
            'video_path': st.session_state.get('video_path', 'unknown')
        }
        
        # 保存到JSON文件
        with open(log_filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"详细生成日志已保存: {log_filepath}")
        
        # 同时保存一个可读的文本格式
        text_log_filepath = os.path.join(DETAILED_LOG_DIR, f"generation_log_{timestamp}.txt")
        with open(text_log_filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("X-VARS 详细生成日志\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"时间: {log_data['timestamp']}\n")
            f.write(f"视频路径: {log_data['video_path']}\n")
            f.write(f"处理时间: {processing_time:.2f}秒\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("1. 用户原始问题\n")
            f.write("-" * 80 + "\n")
            f.write(f"{question}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("2. RAG状态和检索结果\n")
            f.write("-" * 80 + "\n")
            f.write(f"RAG启用: {use_rag}\n")
            if rag_results:
                f.write(f"检索到 {len(rag_results)} 条相关规则:\n\n")
                for result in rag_results:
                    f.write(f"  [排名 {result.get('rank', 'N/A')}] 相似度: {result.get('score', 0.0):.4f}\n")
                    f.write(f"  内容: {result.get('text', '')}\n\n")
            if knowledge_context:
                f.write(f"使用的知识上下文（Top1）:\n{knowledge_context}\n\n")
            else:
                f.write("未检索到知识上下文\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("3. 增强后的问题\n")
            f.write("-" * 80 + "\n")
            f.write(f"{enhanced_question}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("4. 完整Prompt（包含视频token）\n")
            f.write("-" * 80 + "\n")
            f.write(f"Prompt长度: {len(full_prompt)} 字符\n")
            f.write(f"Prompt Token数量（估算）: {log_data.get('prompt_token_count', 'N/A')}\n\n")
            f.write("完整Prompt内容:\n")
            f.write("=" * 80 + "\n")
            f.write(f"{full_prompt}\n")
            f.write("=" * 80 + "\n\n")
            
            # 分析Prompt结构
            if DEFAULT_VID_START_TOKEN in full_prompt:
                start_idx = full_prompt.find(DEFAULT_VID_START_TOKEN)
                end_idx = full_prompt.find(DEFAULT_VID_END_TOKEN) + len(DEFAULT_VID_END_TOKEN) if DEFAULT_VID_END_TOKEN in full_prompt else len(full_prompt)
                video_section = full_prompt[start_idx:end_idx]
                f.write(f"视频Token部分长度: {len(video_section)} 字符\n")
                f.write(f"视频Token数量（估算）: {video_section.count(DEFAULT_VIDEO_PATCH_TOKEN)}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("5. 视频特征信息\n")
            f.write("-" * 80 + "\n")
            for key, value in video_features_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("6. 模型参数\n")
            f.write("-" * 80 + "\n")
            for key, value in model_params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("7. 模型原始输出\n")
            f.write("-" * 80 + "\n")
            f.write(f"{raw_output}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("8. 最终返回的输出\n")
            f.write("-" * 80 + "\n")
            f.write(f"{final_output}\n\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"可读格式日志已保存: {text_log_filepath}")
        
    except Exception as e:
        logger.error(f"保存详细日志失败: {e}")
        import traceback
        logger.error(traceback.format_exc())


def generate_answer(question: str) -> tuple[str, float]:
    """
    基于已处理的视频特征和RAG知识生成答案
    
    参数:
        question (str): 用户问题
        
    返回:
        tuple[str, float]: (生成的答案, 处理时间)
        
    功能:
        - 检查视频是否已处理
        - 加载X-VARS模型组件
        - 使用RAG系统检索相关足球规则知识
        - 构建增强的提示词
        - 生成简洁的回答
        - 记录详细的生成日志
        - 返回答案和处理时间
    """
    if not st.session_state.video_processed or st.session_state.video_features is None:
        return "请先处理视频后再开始对话", 0.0
        
    try:
        # 获取全局模型和参数
        model, _, _, tokenizer, _ = load_all_models(
            BASE_MODEL_DIR,
            LOCAL_TOKENIZER_DIR if os.path.isdir(LOCAL_TOKENIZER_DIR) else None,
            LOCAL_CLIP_DIR if os.path.isdir(LOCAL_CLIP_DIR) else None,
            CLIP_CLASSIFIER_WEIGHTS if os.path.isfile(CLIP_CLASSIFIER_WEIGHTS) else None,
        )
        
        video_token_len = st.session_state.get('video_token_len', 200)
        temperature = st.session_state.get('temperature', 0.2)
        max_new_tokens = st.session_state.get('max_new_tokens', 64)
        do_sample = st.session_state.get('do_sample', False)
        use_rag = st.session_state.get('use_rag', True)
        
        import time as _time
        gen_t0 = _time.time()
        
        # 记录用户原始问题
        logger.info("=" * 80)
        logger.info(f"[生成开始] 用户问题: {question}")
        logger.info(f"[参数] video_token_len={video_token_len}, temperature={temperature}, max_new_tokens={max_new_tokens}, do_sample={do_sample}, use_rag={use_rag}")
        
        # 构建简洁的提示词
        enhanced_question = question
        knowledge_context = ""
        rag_results_full = []
        
        # 如果启用RAG且系统可用（简化版本）
        if use_rag and 'rag_system' in st.session_state and st.session_state.rag_system.initialized:
            try:
                # 检索相关足球规则知识
                rag_results = st.session_state.rag_system.search(question, top_k=3)  # 检索top3，但只使用top1
                rag_results_full = rag_results
                st.session_state.rag_last_results = rag_results_full
                st.session_state.rag_last_query = question
                
                if rag_results:
                    # 只使用最相关的一条规则，保持简洁
                    top_result = rag_results[0]
                    knowledge_context = top_result['text'][:200] + "..." if len(top_result['text']) > 200 else top_result['text']
                    
                    # 构建简洁的问题
                    enhanced_question = f"{question}\n\n参考规则：{knowledge_context}\n\n请简洁回答："
                    
                    logger.info(f"[RAG] 检索到 {len(rag_results)} 条相关规则")
                    logger.info(f"[RAG] Top1规则相似度: {top_result.get('score', 'N/A')}")
                    logger.info(f"[RAG] Top1规则内容（前200字符）: {knowledge_context}")
                    if len(rag_results) > 1:
                        logger.info(f"[RAG] Top2规则相似度: {rag_results[1].get('score', 'N/A')}")
                else:
                    logger.warning("[RAG] 未检索到相关规则")
                    
            except Exception as e:
                logger.error(f"[RAG] 检索失败: {e}")
                import traceback
                logger.error(f"[RAG] 详细错误: {traceback.format_exc()}")
                # RAG失败时回退到原始问题
                enhanced_question = f"{question}\n\n请简洁回答："
        else:
            # 没有RAG时也要求简洁回答
            enhanced_question = f"{question}\n\n请简洁回答："
            logger.info("[RAG] RAG未启用或未初始化")
            st.session_state.rag_last_results = []
            st.session_state.rag_last_query = question
        
        logger.info(f"[增强问题] {enhanced_question}")
        
        # 构建提示词
        prompt = build_prompt(enhanced_question, model, int(video_token_len))
        
        # 记录完整prompt信息（详细版本）
        logger.info(f"[Prompt] 完整Prompt长度: {len(prompt)} 字符")
        logger.info(f"[Prompt] Prompt前1000字符:\n{prompt[:1000]}")
        logger.info(f"[Prompt] Prompt后1000字符:\n...{prompt[-1000:]}")
        
        # 记录prompt中的关键部分
        if DEFAULT_VID_START_TOKEN in prompt:
            start_idx = prompt.find(DEFAULT_VID_START_TOKEN)
            end_idx = prompt.find(DEFAULT_VID_END_TOKEN) + len(DEFAULT_VID_END_TOKEN) if DEFAULT_VID_END_TOKEN in prompt else len(prompt)
            video_token_section = prompt[start_idx:end_idx]
            logger.info(f"[Prompt] 视频Token部分长度: {len(video_token_section)} 字符")
            logger.info(f"[Prompt] 视频Token部分预览: {video_token_section[:200]}...{video_token_section[-200:]}")
        
        # 使用tokenizer分析prompt
        try:
            tokenized = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
            prompt_token_count = tokenized['input_ids'].shape[1]
            logger.info(f"[Prompt] Token化后长度: {prompt_token_count} tokens")
            
            # 记录prompt中的特殊token
            special_tokens = []
            if DEFAULT_VID_START_TOKEN in prompt:
                special_tokens.append(f"VID_START: {DEFAULT_VID_START_TOKEN}")
            if DEFAULT_VID_END_TOKEN in prompt:
                special_tokens.append(f"VID_END: {DEFAULT_VID_END_TOKEN}")
            if DEFAULT_VIDEO_PATCH_TOKEN in prompt:
                patch_count = prompt.count(DEFAULT_VIDEO_PATCH_TOKEN)
                special_tokens.append(f"VIDEO_PATCH: {patch_count}个")
            if special_tokens:
                logger.info(f"[Prompt] 特殊Token: {', '.join(special_tokens)}")
        except Exception as e:
            logger.warning(f"[Prompt] Token化分析失败: {e}")
        
        # 确保视频特征是float32类型
        video_features = st.session_state.video_features
        video_features_info = {}
        if video_features is not None:
            # 确保视频特征形状正确 (1, N, C)
            if len(video_features.shape) == 2:
                # 如果是 (N, C)，添加batch维度
                video_features = video_features.unsqueeze(0)
                logger.warning("[视频特征] 自动添加batch维度: (N, C) -> (1, N, C)")
            elif len(video_features.shape) != 3:
                logger.error(f"[视频特征] 错误：视频特征形状不正确: {video_features.shape}")
                raise ValueError(f"视频特征形状必须是 (1, N, C) 或 (N, C)，实际: {video_features.shape}")
            
            video_features = video_features.to(dtype=torch.float32)
            video_features_info = {
                'dtype': str(video_features.dtype),
                'shape': list(video_features.shape),
                'device': str(video_features.device),
                'min': float(video_features.min().item()),
                'max': float(video_features.max().item()),
                'mean': float(video_features.mean().item()),
                'std': float(video_features.std().item())
            }
            logger.info(f"[视频特征] {video_features_info}")
            
            # 验证视频特征是否包含有效数据
            if video_features.numel() == 0:
                logger.error("[视频特征] 严重错误：视频特征为空！")
                raise ValueError("视频特征为空")
            if torch.isnan(video_features).any():
                logger.error("[视频特征] 严重错误：视频特征包含NaN值！")
                raise ValueError("视频特征包含NaN值")
        else:
            logger.error("[视频特征] 严重错误：视频特征为None，无法生成回答！")
            raise ValueError("视频特征不能为None，请先处理视频")
        
        # 生成答案（简化版）
        logger.info(f"[生成] 开始生成答案: max_tokens={max_new_tokens}, temperature={temperature}, do_sample={do_sample}")
        
        output_text = model_generate_simple(
                model=model,
                tokenizer=tokenizer,
            video_spatio_temporal_features=video_features,
                prompt=prompt,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
            debug=True,  # 始终启用调试以监控生成过程
            )
        
        gen_t1 = _time.time()
        processing_time = gen_t1 - gen_t0
        
        # 记录模型原始输出
        raw_output = output_text  # 保存原始输出
        
        # 检查生成结果（简化版）
        if not output_text or output_text.strip() == "":
            output_text = "抱歉，模型没有生成有效的回答。请尝试重新询问。"
            logger.warning("[生成] 模型生成了空回答")
        else:
            logger.info(f"[生成] 生成完成: {len(output_text)}字符, 耗时{processing_time:.2f}s")
            logger.info(f"[生成] 原始输出: {output_text}")
        
        # 如果使用了RAG，在回答后添加简单的知识来源提示
        final_output = output_text
        if use_rag and knowledge_context and 'rag_system' in st.session_state:
            final_output = output_text + "\n\n💡 *结合了足球规则知识*"
            
        # 保存详细日志
        model_params = {
            'video_token_len': video_token_len,
            'temperature': temperature,
            'max_new_tokens': max_new_tokens,
            'do_sample': do_sample,
            'use_rag': use_rag
        }
        
        # 获取完整的知识上下文（用于日志）
        knowledge_context_for_log = ""
        if rag_results_full:
            knowledge_context_for_log = rag_results_full[0].get('text', '')
        
        # 记录RAG检索的完整结果（用于日志）
        rag_results_for_log = []
        if rag_results_full:
            for i, result in enumerate(rag_results_full[:3]):  # 只记录top3
                rag_results_for_log.append({
                    'rank': i + 1,
                    'score': float(result.get('score', 0.0)),
                    'text': result.get('text', '')[:500]  # 限制长度
                })
        
        save_detailed_generation_log(
            question=question,
            enhanced_question=enhanced_question,
            knowledge_context=knowledge_context_for_log,
            full_prompt=prompt,
            video_features_info=video_features_info,
            model_params=model_params,
            raw_output=raw_output,
            final_output=final_output,
            processing_time=processing_time,
            use_rag=use_rag,
            rag_results=rag_results_for_log  # 添加RAG检索结果
        )
        
        logger.info("=" * 80)
            
        return final_output, processing_time
        
    except Exception as e:
        logger.error(f"[错误] 生成答案失败: {str(e)}")
        import traceback
        logger.error(f"[错误] 详细堆栈: {traceback.format_exc()}")
        return f"生成过程出错: {str(e)}", 0.0

def load_football_theme_css():
    """
    加载足球主题CSS样式
    
    功能:
        - 从static目录加载CSS文件
        - 如果文件不存在，使用内联CSS
    """
    css_file = os.path.join("static", "football_theme.css")
    
    if os.path.exists(css_file):
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
    else:
        # 如果文件不存在，使用内联CSS（简化版）
        css_content = """
        <style>
        :root {
            --football-green: #2d5016;
            --football-light-green: #4a7c2a;
            --football-field-green: #3d6b1f;
            --football-white: #ffffff;
            --football-yellow: #ffd700;
        }
        .stApp {
            background: linear-gradient(135deg, #2d5016 0%, #3d6b1f 50%, #2d5016 100%);
        }
        h1 {
            color: var(--football-white) !important;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a3d0a 0%, #2d5016 100%);
            border-right: 3px solid var(--football-yellow);
        }
        .stButton > button {
            background: linear-gradient(135deg, var(--football-yellow) 0%, #ffed4e 100%) !important;
            color: #1a1a1a !important;
            border: 2px solid var(--football-white) !important;
            border-radius: 25px !important;
            transition: all 0.3s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.05) !important;
            box-shadow: 0 8px 20px rgba(255,215,0,0.5) !important;
        }
        </style>
        """
    
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

def load_football_theme_js():
    """
    加载足球主题JavaScript脚本
    
    功能:
        - 从static目录加载JS文件
        - 如果文件不存在，使用内联JS
    """
    js_file = os.path.join("static", "football_theme.js")
    
    if os.path.exists(js_file):
        with open(js_file, 'r', encoding='utf-8') as f:
            js_content = f.read()
    else:
        # 如果文件不存在，使用内联JS（简化版）
        js_content = """
        (function() {
            // 添加消息动画
            const messages = document.querySelectorAll('[data-testid="stChatMessage"]');
            messages.forEach((msg, idx) => {
                msg.style.opacity = '0';
                msg.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    msg.style.transition = 'all 0.5s ease-out';
                    msg.style.opacity = '1';
                    msg.style.transform = 'translateY(0)';
                }, idx * 100);
            });
        })();
        """
    
    st.markdown(f"<script>{js_content}</script>", unsafe_allow_html=True)

def quick_ask(question: str):
    """快捷提问函数"""
    add_chat_message("user", question)
    
    with st.spinner("🤖 正在思考..."):
        answer, processing_time = generate_answer(question)
        add_chat_message("assistant", answer, {"processing_time": processing_time})
    
    st.rerun()

def auto_save_chat_history():
    """
    自动保存对话历史到本地文件
    
    功能:
        - 每次对话后自动保存到本地JSON文件
        - 按视频文件名或时间戳组织文件
        - 包含完整的对话记录、视频信息、RAG信息
        - 静默保存，不干扰用户操作
    """
    # 检查是否启用自动保存（优先使用session state中的配置）
    auto_save_enabled = st.session_state.get('auto_save_chat', AUTO_SAVE_CHAT)
    if not auto_save_enabled:
        return
    
    if not st.session_state.chat_history:
        return
    
    try:
        import json
        from datetime import datetime
        
        # 确保保存目录存在
        os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
        
        # 准备导出数据
        rag_info = {}
        if 'rag_system' in st.session_state:
            rag_stats = st.session_state.rag_system.get_stats()
            rag_info = {
                'rag_enabled': st.session_state.get('use_rag', False),
                'rag_initialized': rag_stats['initialized'],
                'num_knowledge_documents': rag_stats['num_documents']
            }
        
        # 生成会话标识符（基于视频路径或时间戳）
        video_path = st.session_state.get('video_path', 'unknown')
        if video_path and video_path != 'unknown':
            # 使用视频文件名作为会话标识
            video_name = os.path.basename(video_path)
            session_id = os.path.splitext(video_name)[0]
        else:
            # 如果没有视频路径，使用时间戳
            session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 构建保存数据
        save_data = {
            'session_id': session_id,
            'video_path': video_path,
            'save_time': datetime.now().isoformat(),
            'video_info': st.session_state.get('processing_info', {}),
            'rag_info': rag_info,
            'chat_history': []
        }
        
        # 转换对话历史
        for msg in st.session_state.chat_history:
            save_data['chat_history'].append({
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': datetime.fromtimestamp(msg['timestamp']).isoformat(),
                'metadata': msg.get('metadata', {})
            })
        
        # 生成文件名（包含时间戳以确保唯一性）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"chat_{session_id}_{timestamp}.json"
        filepath = os.path.join(CHAT_HISTORY_DIR, filename)
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"对话记录已自动保存: {filepath}")
        
        # 同时保存一个最新的会话文件（覆盖更新）
        latest_filepath = os.path.join(CHAT_HISTORY_DIR, f"chat_{session_id}_latest.json")
        with open(latest_filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"自动保存对话记录失败: {e}")
        # 静默失败，不干扰用户操作


def export_chat_history():
    """
    导出对话历史（手动下载）
    
    功能:
        - 提供手动下载对话记录的功能
        - 生成JSON格式的下载文件
        - 包含完整的对话历史、视频信息、RAG信息
    """
    if not st.session_state.chat_history:
        st.warning("没有对话历史可以导出")
        return
        
    import json
    from datetime import datetime
    
    # 准备导出数据
    rag_info = {}
    if 'rag_system' in st.session_state:
        rag_stats = st.session_state.rag_system.get_stats()
        rag_info = {
            'rag_enabled': st.session_state.get('use_rag', False),
            'rag_initialized': rag_stats['initialized'],
            'num_knowledge_documents': rag_stats['num_documents']
        }
    
    export_data = {
        'export_time': datetime.now().isoformat(),
        'video_info': st.session_state.processing_info,
        'rag_info': rag_info,
        'chat_history': []
    }
    
    for msg in st.session_state.chat_history:
        export_data['chat_history'].append({
            'role': msg['role'],
            'content': msg['content'],
            'timestamp': datetime.fromtimestamp(msg['timestamp']).isoformat(),
            'metadata': msg.get('metadata', {})
        })
    
    # 创建下载链接
    json_data = json.dumps(export_data, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="💾 下载对话记录",
        data=json_data,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def main():
    """
    X-VARS智能足球裁判系统主函数
    
    功能:
        - 设置Streamlit页面配置
        - 初始化聊天状态和RAG系统
        - 创建三列布局：视频管理、视频预览、智能对话
        - 处理视频上传和特征提取
        - 提供对话界面和快捷问题
        - 管理聊天历史和导出功能
    """
    # 设置侧边栏默认隐藏
    st.set_page_config(
        page_title="X-VARS 智能足球裁判系统 (简洁版)", 
        layout="wide",
        initial_sidebar_state="collapsed"  # 默认隐藏侧边栏
    )
    
    # 加载足球主题CSS样式
    load_football_theme_css()
    
    # 加载足球主题JavaScript
    load_football_theme_js()
    
    st.title("⚽ X-VARS 智能足球裁判系统")
    st.caption("📝 简洁模式 | 🧠 视频分析 + 规则知识库 = 快速精准判断 | 直接有效，无冗余回答！")
    
    # 使用说明
    with st.expander("📖 使用指南", expanded=False):
        st.markdown("""
        **🚀 三步开始对话：**
        1. **上传视频** - 在左侧选择足球视频文件或使用示例视频
        2. **处理视频** - 点击"🚀 处理视频"按钮提取特征（仅需一次）
        3. **开始对话** - 在右侧聊天框中输入问题或使用快捷按钮
        
        **💡 使用技巧：**
        - 🎯 **快捷提问**: 首次对话可使用预设问题快速开始
        - 💬 **连续对话**: 无需重复处理视频，可以就同一视频进行多轮对话
        - 🗑️ **清空对话**: 可随时清空聊天历史重新开始
        - 📥 **导出对话**: 可导出完整对话记录为JSON文件
        - ⚙️ **参数调节**: 在左侧调整模型参数优化生成效果
        
        **🔧 优化简洁回答质量：**
        - 📏 **回答太长**: 减少 `Max new tokens` 至32-128
        - 🔄 **回答重复**: 降低 `Temperature` 至0.1-0.2  
        - 📝 **需要更简洁**: 使用左侧 `极简模式` 预设
        - 🧠 **需要专业性**: 启用 `足球规则增强(RAG)` 功能
        
        **🔧 参数说明：**
        - 💻 **默认CPU模式**: CLIP和RAG编码器默认使用CPU，节省显存确保稳定运行
        - ⚡ **性能提升**: 如需更快速度，可手动切换CLIP和RAG到GPU模式
        - 🧠 **RAG知识库**: 结合FIFA足球规则，提供更专业的分析
        - 💾 **显存优化**: RAG编码器默认CPU运行，避免显存冲突
        - 🚀 **一键优化**: 显存不足时点击"一键优化设置"
        - 🛑 **紧急控制**: 需要时使用"强制停止所有处理"
        - 📚 **详细说明**: 查看左侧"技术说明与故障排除"获取更多信息
        """)
    
    # 初始化聊天状态
    initialize_chat_state()

    with st.sidebar:
        st.subheader("运行参数")
        debug_mode = st.checkbox("开启调试模式(终端 DEBUG 日志)", value=False)
        if debug_mode:
            logger.setLevel(logging.DEBUG)
        # 检查是否需要自动优化设置
        if 'auto_optimize' in st.session_state and st.session_state.auto_optimize:
            # 节省显存的优化设置
            default_max_frames = 8
            default_clip_bs = 1
            default_video_token_len = 64  # 安全的较小值
            st.info("🚀 已切换到节省显存模式 - 使用较小的参数以避免序列长度问题")
        else:
            # 默认参数
            default_max_frames = 16
            default_clip_bs = 1  # 批大小1 (默认)
            default_video_token_len = min(DEFAULT_VIDEO_TOKEN_LEN, 200)  # 限制在200以内

        # 参数统一通过form提交，避免边调边触发重绘
        use_rag_for_default = st.session_state.get('use_rag', False)
        default_max_tokens = 128 if use_rag_for_default else 64

        with st.form("runtime_params_form"):
            form_temperature = st.slider(
                "Temperature", 0.0, 1.0, float(st.session_state.get("temperature", 0.2)), 0.1
            )
            form_max_new_tokens = st.slider(
                "Max new tokens", 32, 512, int(st.session_state.get("max_new_tokens", default_max_tokens)), 16
            )
            st.caption("💡 简洁模式推荐值: 32-128。数值越大回答越详细。")
            form_do_sample = st.checkbox(
                "采样生成(取消则贪心)", value=bool(st.session_state.get("do_sample", False))
            )
            form_video_token_len = st.number_input(
                "Video token len",
                min_value=64,
                max_value=512,
                value=int(st.session_state.get("video_token_len", default_video_token_len)),
                step=8,
            )
            st.caption("推荐值：64-300。过大可能导致序列长度不匹配错误。")
            form_max_frames = st.slider("采样帧数", 8, 64, int(st.session_state.get("max_frames", default_max_frames)), 4)
            if torch.cuda.is_available():
                current_clip_device = st.session_state.get("clip_device_choice", "GPU")
                form_clip_device = st.selectbox(
                    "CLIP设备", ["GPU", "CPU"], index=0 if current_clip_device == "GPU" else 1
                )
            else:
                form_clip_device = "CPU"
                st.caption("当前无可用GPU，CLIP自动使用CPU。")
            form_clip_bs = st.slider("CLIP 批大小", 1, 8, int(st.session_state.get("clip_bs", default_clip_bs)), 1)
            apply_params = st.form_submit_button("✅ 应用参数", use_container_width=True)

        if apply_params:
            st.session_state.temperature = form_temperature
            st.session_state.max_new_tokens = form_max_new_tokens
            st.session_state.do_sample = form_do_sample
            st.session_state.video_token_len = int(form_video_token_len)
            st.session_state.max_frames = form_max_frames
            st.session_state.clip_device_choice = form_clip_device
            st.session_state.clip_bs = form_clip_bs
            st.success("参数已应用，将用于后续处理与问答。")

        device = _get_device()
        st.write(f"设备: {device}")
        st.caption("💡 推荐: 4080使用GPU。若出现显存不足再切CPU。")
        
        # 对话记录自动保存配置
        st.markdown("---")
        st.subheader("💾 对话记录管理")
        
        # 自动保存开关
        auto_save_enabled = st.checkbox(
            "自动保存对话记录", 
            value=AUTO_SAVE_CHAT,
            help="每次对话后自动保存到本地文件（chat_history目录）"
        )
        # 更新全局配置（通过session state传递）
        st.session_state.auto_save_chat = auto_save_enabled
        
        if auto_save_enabled:
            st.info(f"✅ 对话记录将自动保存到: `{CHAT_HISTORY_DIR}/` 目录")
            # 显示已保存的文件数量
            if os.path.exists(CHAT_HISTORY_DIR):
                saved_files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith('.json')]
                st.caption(f"📁 已保存 {len(saved_files)} 个对话记录文件")
        else:
            st.warning("⚠️ 自动保存已关闭，对话记录不会自动保存到本地")
        
        # RAG系统控制
        st.markdown("---")
        st.subheader("🧠 RAG知识库")
        
        if not RAG_AVAILABLE:
            st.warning("⚠️ RAG功能不可用")
            st.error("请安装RAG依赖包:")
            st.code("pip install faiss-cpu sentence-transformers", language="bash")
            st.session_state.use_rag = False
        else:
            # RAG设备选择
            if torch.cuda.is_available():
                st.session_state.rag_device = st.selectbox("RAG设备", ["cuda", "cpu"], index=0)
            else:
                st.session_state.rag_device = "cpu"
            st.info(f"🧠 RAG编码器运行在: {st.session_state.rag_device}")
            
            # 如果设备发生变化，重置RAG系统
            if 'rag_system' in st.session_state and st.session_state.rag_system.device != st.session_state.rag_device:
                logger.info(f"RAG设备变更: {st.session_state.rag_system.device} -> {st.session_state.rag_device}")
                st.session_state.rag_system = FootballRAGSystem(device=st.session_state.rag_device)
                st.session_state.rag_initialized = False
                st.info(f"已切换RAG设备到{st.session_state.rag_device}，需要重新初始化")
                
            st.markdown("---")
            # 显示知识库文件状态
            knowledge_exists = os.path.exists(KNOWLEDGE_BASE_FILE)
            if knowledge_exists:
                file_size = os.path.getsize(KNOWLEDGE_BASE_FILE)
                st.info(f"📄 知识库文件: {KNOWLEDGE_BASE_FILE} ({file_size:,} 字节)")
            else:
                st.error(f"❌ 知识库文件不存在: {KNOWLEDGE_BASE_FILE}")
                st.info(f"当前目录: {os.getcwd()}")
                if os.path.exists('vector_store'):
                    files = os.listdir('vector_store')
                    st.info(f"vector_store目录内容: {files}")
                else:
                    st.error("vector_store目录不存在")
            
            # RAG系统初始化控制
            col_init1, col_init2 = st.columns(2)
            
            with col_init1:
                # 初始化RAG系统
                if not st.session_state.rag_initialized and knowledge_exists:
                    if st.button("🔄 初始化知识库", type="primary"):
                        with st.spinner("正在初始化足球规则知识库..."):
                            success = st.session_state.rag_system.initialize()
                            st.session_state.rag_initialized = success
                            if success:
                                st.success("✅ 足球规则知识库加载成功！")
                                st.rerun()
                            else:
                                st.error("❌ 知识库加载失败，请查看终端日志")
                                
            with col_init2:
                # 强制重新构建索引
                if knowledge_exists:
                    if st.button("🔥 强制重建向量存储", type="secondary"):
                        # 删除现有向量存储文件
                        if os.path.exists(SIMPLE_EMBEDDINGS_FILE):
                            os.remove(SIMPLE_EMBEDDINGS_FILE)
                        
                        # 重新初始化
                        st.session_state.rag_initialized = False
                        with st.spinner("正在重建知识库向量存储..."):
                            success = st.session_state.rag_system.initialize()
                            st.session_state.rag_initialized = success
                            if success:
                                st.success("✅ 向量存储重建成功！")
                                st.rerun()
                            else:
                                st.error("❌ 向量存储重建失败")
            
            # RAG系统状态显示
            rag_stats = st.session_state.rag_system.get_stats()
            
            if st.session_state.rag_initialized and rag_stats['initialized']:
                search_mode = rag_stats.get('search_mode', '未知')
                if search_mode == '向量搜索':
                    st.success(f"✅ 知识库已就绪 ({rag_stats['num_documents']}个文档片段) - 🧠 向量搜索模式")
                else:
                    st.warning(f"⚠️ 知识库已就绪 ({rag_stats['num_documents']}个文档片段) - 📝 文本匹配模式")
                    st.caption("💡 向量编码失败，使用文本关键词匹配。功能仍然可用，但搜索精度较低。")
                
                st.session_state.use_rag = st.checkbox("启用足球规则增强", value=True, 
                    help="结合FIFA足球规则知识库提供更专业的分析")
                
                # 显示索引文件状态
                with st.expander("📊 RAG系统详情"):
                    st.write(f"📄 知识库文件: {'✅ 存在' if rag_stats['knowledge_file_exists'] else '❌ 不存在'}")
                    st.write(f"💾 向量存储: {'✅ 存在' if rag_stats['index_file_exists'] else '❌ 不存在'}")
                    st.write(f"📚 文档片段数: {rag_stats['num_documents']}")
                    st.write(f"💻 RAG设备: {st.session_state.rag_device}")
                    st.write(f"🔍 搜索模式: {rag_stats.get('search_mode', '未知')}")
                    
                    # 组件状态
                    st.markdown("**组件状态**:")
                    encoder_status = "✅ 可用" if rag_stats.get('has_encoder', False) else "❌ 不可用"
                    embeddings_status = "✅ 可用" if rag_stats.get('has_embeddings', False) else "❌ 不可用"
                    st.write(f"🤖 编码器: {encoder_status}")
                    st.write(f"🧠 嵌入向量: {embeddings_status}")
                    
                    # 依赖检查
                    st.markdown("**依赖检查**:")
                    import importlib.util
                    if importlib.util.find_spec("sentence_transformers") is not None:
                        st.success("✅ SentenceTransformers 可用")
                    else:
                        st.error("❌ SentenceTransformers 不可用")
                        
                    if rag_stats.get('has_embeddings', False):
                        if importlib.util.find_spec("sentence_transformers.util") is not None:
                            st.success("✅ 相似度计算 可用")
                        else:
                            st.error("❌ 相似度计算 不可用")
                    
                    if rag_stats.get('search_mode') == '文本匹配':
                        st.info("🔧 使用简化文本匹配，不依赖向量计算")
                    else:
                        st.info("🔧 使用向量搜索，基于语义相似度")
                    
            else:
                st.session_state.use_rag = False
                if not knowledge_exists:
                    st.error("请在vector_store目录中放置 '2023-2024中文版足球規則.txt' 文件")
                elif not st.session_state.rag_initialized:
                    st.info("💡 点击'初始化知识库'开始使用RAG功能")
                else:
                    st.error("❌ 知识库初始化失败，请检查终端日志")
        
        # 参数优化建议
        with st.expander("⚙️ 简洁回答参数建议", expanded=False):
            st.markdown("**🎯 如何获得简洁高质量的回答：**")
            
            col_tip1, col_tip2 = st.columns(2)
            with col_tip1:
                st.markdown("""
                **📏 简洁回答长度设置**
                - **极简**: 32-64 tokens
                - **简洁**: 64-128 tokens  
                - **标准**: 128-256 tokens
                - **详细**: 256-512 tokens
                """)
                
            with col_tip2:
                st.markdown("""
                **🎛️ 生成参数调节**
                - **Temperature**: 0.1-0.3 更稳定
                - **采样**: 关闭贪心模式更一致
                - **Token数**: 适中避免冗余
                - **RAG**: 使用精简规则引用
                """)
            
            st.markdown("**🔧 优化建议：**")
            st.markdown("- 📏 **回答太长** → 减少 `Max new tokens`")
            st.markdown("- 🔄 **回答重复** → 降低 `Temperature`")
            st.markdown("- 🤖 **回答太简单** → 适量增加tokens或启用采样")
            st.markdown("- 🧠 **需要专业性** → 启用 `足球规则增强(RAG)`")
            
            # 一键设置按钮
            st.markdown("**⚡ 快速设置：**")
            col_preset1, col_preset2, col_preset3 = st.columns(3)
            
            with col_preset1:
                if st.button("📝 极简模式", help="最简短回答"):
                    st.session_state.max_new_tokens = 64
                    st.session_state.temperature = 0.1
                    st.session_state.do_sample = False
                    st.rerun()
                    
            with col_preset2:
                if st.button("⚖️ 简洁模式", help="简洁专业回答"):
                    st.session_state.max_new_tokens = 128
                    st.session_state.temperature = 0.2
                    st.session_state.do_sample = False
                    st.rerun()
                    
            with col_preset3:
                if st.button("🔬 标准模式", help="标准详细回答"):
                    st.session_state.max_new_tokens = 256
                    st.session_state.temperature = 0.3
                    st.session_state.do_sample = True
                    st.rerun()
        
        # 参数变更提示
        if st.session_state.video_processed:
            st.success("💡 参数已保存，将应用于后续对话")
        else:
            st.info("💡 请先处理视频，然后开始对话")
        
        # 显示系统资源信息 - 仅CPU信息
        show_system_info = st.checkbox("显示系统信息", value=True)
        
        if show_system_info:
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                st.write(f"🖥️ CPU 使用率: {cpu_percent}%")
                st.write(f"💾 内存使用: {memory.percent}% ({_format_bytes(memory.used)}/{_format_bytes(memory.total)})")
                
                # CPU模式提示
                st.info("🎯 系统运行在纯CPU模式，无GPU依赖")
                
                # CPU性能建议
                if cpu_percent > 90:
                    st.warning("⚠️ CPU使用率很高，建议降低批处理大小")
                elif memory.percent > 85:
                    st.warning("⚠️ 内存使用率较高，建议降低采样帧数")
                else:
                    st.success("✅ 系统资源使用正常")
                        
            except ImportError:
                st.write("安装 psutil 以查看详细系统信息")
                
        # CPU优化按钮
        col_opt1, col_opt2 = st.columns([1, 1])
        with col_opt1:
            if st.button("🚀 CPU性能优化"):
                st.session_state.max_frames = 8  # 较少帧数
                st.session_state.clip_bs = 1     # 最小批次
                st.session_state.video_token_len = 128  # 较短序列
                st.success("已优化为CPU高性能模式")
                st.rerun()
        with col_opt2:
            if st.button("🔄 重置为默认设置"):
                st.session_state.max_frames = 16
                st.session_state.clip_bs = 1
                st.session_state.video_token_len = 200
                st.success("已重置为默认设置")
                st.rerun()
                
        # 技术说明 - CPU模式
        with st.expander("🔬 技术说明与CPU优势"):
            st.write("**🖥️ 纯CPU模式优势**：")
            st.write("- ✅ **最大兼容性**: 无需GPU驱动，任何设备都能运行")
            st.write("- ✅ **稳定可靠**: 避免CUDA版本冲突和显存不足问题")
            st.write("- ✅ **精度一致**: 使用Float32单精度，数值计算更精确")
            st.write("- ✅ **内存管理**: 使用系统内存，通常比显存更充足")
            st.write("")
            st.write("**序列长度管理**：")
            st.write("- 📏 **自动调整**: 视频特征序列长度自动匹配Video token len设置")
            st.write("- 🔧 **智能填充**: 序列过短时自动零填充，过长时智能采样")
            st.write("- ⚖️ **长度验证**: 生成前自动验证序列长度匹配")
            st.write("")
            st.write("**CPU优化配置**：")
            st.write("- 🎯 **批处理大小**: 使用较小批次减少内存占用")
            st.write("- 🔢 **序列长度**: 适中的token长度平衡性能和质量")
            st.write("- 📹 **采样帧数**: 合理的帧数确保处理速度")
            st.write("- 💾 **内存效率**: 及时清理中间结果节省内存")
            st.write("")
            st.write("**RAG知识库系统**：")
            st.write("- 🧠 **足球规则库**: 基于FIFA官方足球规则2023-2024版本")
            st.write("- 🔍 **智能检索**: 使用余弦相似度或智能文本匹配")
            st.write("- 🎯 **上下文增强**: 将规则知识与视频内容结合分析")
            st.write("- 📚 **专业判断**: 提供符合官方规则的专业分析")
            st.write("- 🖥️ **CPU优化**: 完全在CPU上运行，无依赖冲突")
            st.write("- 🛠️ **简化实现**: 不依赖FAISS和NumPy，使用纯PyTorch实现")
            st.write("")
            st.write("**性能优化建议**：")
            st.write("- 🚀 **CPU高性能**: 点击'CPU性能优化'自动调整参数")
            st.write("- 💾 **内存管理**: 处理完成后自动清理内存")
            st.write("- ⏱️ **处理时间**: CPU模式处理时间较长但稳定可靠")
            st.write("- 📊 **资源监控**: 实时显示CPU和内存使用情况")
                
        # 紧急状况控制
        st.markdown("---")
        st.subheader("⚠️ 紧急控制")
        if st.button("🛑 强制停止所有处理", type="secondary"):
            st.session_state.emergency_stop = True
            st.error("已发送紧急停止信号！请刷新页面重新开始。")
            
        if st.button("🧹 清理系统内存", help="清理Python内存缓存"):
            import gc
            gc.collect()
            st.success("系统内存清理完成！")

    # 初始化模型
    model_loading_status = st.empty()
    model_loading_status.info("🔄 正在初始化模型，首次加载可能需要几分钟...")
    
    with st.spinner("正在加载本地模型与视觉编码器，请稍候……"):
        try:
            with step_timer("load_all_models"):
                model, vision_tower_hf, _clip_net, tokenizer, image_processor = load_all_models(
                    BASE_MODEL_DIR,
                    LOCAL_TOKENIZER_DIR if os.path.isdir(LOCAL_TOKENIZER_DIR) else None,
                    LOCAL_CLIP_DIR if os.path.isdir(LOCAL_CLIP_DIR) else None,
                    CLIP_CLASSIFIER_WEIGHTS if os.path.isfile(CLIP_CLASSIFIER_WEIGHTS) else None,
                )
            model_loading_status.success("✅ 模型加载完成！")
        except Exception as e:
            model_loading_status.error(f"❌ 模型加载失败: {str(e)}")
            st.stop()
            
    # 清理加载状态提示
    import time
    time.sleep(1)
    model_loading_status.empty()

    # 第一行：视频上传和视频预览（并排显示，视频更大）
    col_upload, col_video = st.columns([1, 2])

    with col_upload:
        st.subheader("📹 视频管理")
        
        # 视频状态显示
        if st.session_state.video_processed:
            st.success("✅ 视频已处理完成")
            if st.session_state.processing_info:
                info = st.session_state.processing_info
                st.info(f"📊 处理信息：{info.get('frames', 0)}帧，耗时{info.get('time', 0):.1f}秒")
        else:
            st.info("🔄 请上传视频开始对话")
        
        uploaded = st.file_uploader("选择一个视频文件", type=["mp4", "mov", "avi", "mkv"]) 
        example_path = os.path.join("data", "videos", "clip_1.mp4")
        use_example = st.checkbox("使用示例视频", value=os.path.isfile(example_path))
        
        # 处理视频上传
        if uploaded is not None or use_example:
            new_video_path = example_path if use_example else uploaded.name if uploaded else None
            
            # 检查是否是新视频
            if new_video_path != st.session_state.video_path:
                st.session_state.video_path = new_video_path
                st.session_state.video_processed = False
                st.session_state.video_features = None
                st.session_state.chat_history = []  # 新视频清空聊天记录
                st.rerun()
        
        # 视频处理按钮
        if (uploaded is not None or use_example) and not st.session_state.video_processed:
            if st.button("🚀 处理视频", type="primary"):
                with st.spinner("正在处理视频..."):
                    success = process_video(uploaded, example_path if use_example else None)
                    if success:
                        st.rerun()
        
        # 重新处理按钮
        if st.session_state.video_processed:
            if st.button("🔄 重新处理视频"):
                st.session_state.video_processed = False
                st.session_state.video_features = None
                st.session_state.chat_history = []
                st.rerun()

    with col_video:
        st.subheader("📺 视频预览")
        
        video_placeholder = st.empty()
        
        # 显示当前视频
        if st.session_state.video_path:
            try:
                if use_example:
                    video_placeholder.video(example_path)
                elif uploaded is not None:
                    video_placeholder.video(uploaded)
            except Exception as e:
                st.error(f"视频显示错误: {e}")
        else:
            video_placeholder.info("📹 上传视频后将在此显示预览")

    # 第二行：分标签展示（对话 / RAG检索 / 系统状态）
    st.markdown("---")
    tab_chat, tab_rag, tab_system = st.tabs(["💬 智能对话", "📚 RAG检索结果", "📊 系统状态"])

    with tab_chat:
        chat_container = st.container()
        with chat_container:
            display_chat_history()

        # 快捷问题按钮（放在聊天历史之后，输入框之前）
        if st.session_state.video_processed and not st.session_state.chat_history:
            st.markdown('<div class="quick-questions-section">', unsafe_allow_html=True)
            st.markdown("**🎯 快捷提问：**")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("⚽ 场面分析", key="quick1", use_container_width=True):
                    question = "请描述这段视频中发生的情况，并分析是否存在违规行为。"
                    quick_ask(question)
            with col2:
                if st.button("🟨 判罚决定", key="quick2", use_container_width=True):
                    question = "根据足球规则，这种情况是否需要出牌？应该给出什么样的处罚？"
                    quick_ask(question)
            with col3:
                if st.button("👥 球员行为", key="quick3", use_container_width=True):
                    question = "分析涉及球员的具体行为，是否符合足球规则要求？"
                    quick_ask(question)
            with col4:
                if st.button("📋 规则解释", key="quick4", use_container_width=True):
                    question = "请根据FIFA足球规则解释这个场面，裁判的判罚是否正确？"
                    quick_ask(question)

            # 额外的RAG特色问题
            if st.session_state.get("use_rag", False) and st.session_state.rag_initialized:
                col5, col6 = st.columns(2)
                with col5:
                    if st.button("⚖️ 规则依据", key="quick5", use_container_width=True):
                        question = "这个判罚的具体规则依据是什么？请引用相关的足球规则条款。"
                        quick_ask(question)
                with col6:
                    if st.button("🔍 细节分析", key="quick6", use_container_width=True):
                        question = "从足球规则的角度，分析这个场面中的关键细节和判罚要点。"
                        quick_ask(question)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        # 聊天输入
        if st.session_state.video_processed:
            question = st.chat_input("输入您的问题...")
            if question:
                add_chat_message("user", question)
                with st.spinner("🤖 正在思考..."):
                    answer, processing_time = generate_answer(question)
                    add_chat_message("assistant", answer, {"processing_time": processing_time})
                st.rerun()
        else:
            st.info("请先处理视频")

        # 聊天管理按钮
        if st.session_state.chat_history:
            col_clear1, col_clear2 = st.columns(2)
            with col_clear1:
                if st.button("🗑️ 清空对话"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col_clear2:
                if st.button("📥 导出对话"):
                    export_chat_history()

    with tab_rag:
        st.subheader("最近一次RAG检索")
        last_query = st.session_state.get("rag_last_query", "")
        if last_query:
            st.caption(f"查询：{last_query}")
        rag_last_results = st.session_state.get("rag_last_results", [])
        if rag_last_results:
            rag_df = pd.DataFrame(
                [
                    {
                        "rank": i + 1,
                        "score": round(float(item.get("score", 0.0)), 4),
                        "snippet": str(item.get("text", "")).replace("\n", " ")[:180],
                    }
                    for i, item in enumerate(rag_last_results)
                ]
            )
            st.dataframe(rag_df, use_container_width=True, hide_index=True)
        else:
            st.info("暂无RAG检索记录。先处理视频并提问后，这里会展示检索Top结果。")

    with tab_system:
        st.subheader("运行指标")
        info = st.session_state.get("processing_info", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("视频帧数", int(info.get("frames", 0)))
        m2.metric("处理耗时(s)", f"{float(info.get('time', 0.0)):.2f}")
        m3.metric("特征长度", int(info.get("sequence_length", 0)))
        m4.metric("CLIP设备", str(info.get("device", "N/A")))


if __name__ == "__main__":
    """
    程序入口点
    
    当直接运行此脚本时，启动X-VARS智能足球裁判系统的Streamlit界面
    """
    main()

