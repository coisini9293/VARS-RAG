import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from video_chatgpt.train.llava_trainer import VideoChatGPTTrainer
from video_chatgpt import video_conversation as conversation_lib
from video_chatgpt.model import *
import torch.distributed as dist
from video_chatgpt.constants import *
import pickle
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer


IGNORE_INDEX = -100 #忽略索引
DEFAULT_PAD_TOKEN = "[PAD]" #默认填充标记
DEFAULT_EOS_TOKEN = "</s>" #默认结束标记
DEFAULT_BOS_TOKEN = "</s>" #默认开始标记
DEFAULT_UNK_TOKEN = "<unk>" #默认未知标记


@dataclass
class ModelArguments: #模型参数
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m") #模型名称或路径
    version: Optional[str] = field(default="v0") #版本
    freeze_backbone: bool = field(default=False) #冻结骨干网络
    tune_mm_mlp_adapter: bool = field(default=False) #tune_mm_mlp_adapter
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None) #预训练mm_mlp_adapter
    mm_use_vid_start_end: bool = field(default=False) #mm_use_vid_start_end


@dataclass
class DataArguments: #数据参数
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."}) #数据路径
    lazy_preprocess: bool = False #懒惰预处理
    is_multimodal: bool = False #是否多模态
    sep_video_conv_front: bool = False #分隔视频对话前
    video_token_len: int = 0 #视频token长度
    video_folder: Optional[str] = field(default=None) #视频文件夹
    frame_aspect_ratio: str = 'square' #帧宽高比


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None) #缓存目录
    optim: str = field(default="adamw_torch") #优化器
    remove_unused_columns: bool = field(default=False) #移除未使用的列
    freeze_mm_mlp_adapter: bool = field(default=False) #冻结mm_mlp_adapter
    force_fsdp: bool = field(default=False) #强制fsdp
    model_max_length: int = field(
        default=512, #模型最大长度
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str): #安全保存模型
    """Collects the state dict and dump to disk."""
    dist.barrier() #屏障
    state_dict = trainer.model.state_dict() #状态字典
    trainer._save(output_dir, state_dict=state_dict)  # noqa #保存模型


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
): #智能调整tokenizer和embedding
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) #添加特殊标记
    model.resize_token_embeddings(len(tokenizer)) #调整tokenizer

    if num_new_tokens > 0: #如果新的标记大于0
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean( #输入嵌入平均
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean( #输出嵌入平均
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg #输入嵌入
        output_embeddings[-num_new_tokens:] = output_embeddings_avg #输出嵌入


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict: #tokenize
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers): #mask目标
    # cur_idx = 0 #当前索引
    cur_idx = tokenized_lens[0] #当前索引
    tokenized_lens = tokenized_lens[1:] #tokenized_lens
    target[:cur_idx] = IGNORE_INDEX #忽略索引
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human": #如果说话者是人类
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX #忽略索引
        cur_idx += tokenized_len #当前索引


def _add_speaker_and_signal(header, source, get_conversation=True): #添加说话者和开始/结束信号
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### " #开始信号
    END_SIGNAL = "\n" #结束信号
    conversation = header #对话
    for sentence in source:
        from_str = sentence["from"] #说话者
        if from_str.lower() == "human": #如果说话者是人类
            from_str = conversation_lib.default_conversation.roles[0] #人类角色
        elif from_str.lower() == "gpt": #如果说话者是gpt
            from_str = conversation_lib.default_conversation.roles[1] #gpt角色
        else:
            from_str = 'unknown' #未知角色
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " + #开始信号 + 说话者 + 冒号 + 句子
                             sentence["value"] + END_SIGNAL) #结束信号
        if get_conversation: #如果需要获取对话
            conversation += sentence["value"] #添加句子
    conversation += BEGIN_SIGNAL #添加开始信号
    return conversation


def preprocess_multimodal(
        sources: Sequence[str],
        multimodal_cfg: dict,
        cur_token_len: int,
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal'] #是否多模态
    video_token_len = cur_token_len #视频token长度
    if not is_multimodal: #如果不需要多模态
        return sources

    for source in sources:
        if multimodal_cfg['sep_video_conv_front']: #如果需要分隔视频对话前
            assert DEFAULT_VIDEO_TOKEN in source[0]['value'] #如果视频token在句子中
            source[0]['value'] = source[0]['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip() #替换视频token
            source[0]['value'] = DEFAULT_VIDEO_TOKEN + conversation_lib.default_conversation.sep + \
            conversation_lib.default_conversation.roles[0] + ": " + source[0]['value'] #添加说话者
        for sentence in source:
            replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len #替换视频token
            if multimodal_cfg['use_vid_start_end']: #如果需要使用视频开始/结束标记
                replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN #替换视频token
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, replace_token) #替换视频token

    return sources


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy() #复制对话


    roles = {"human": conv.roles[0], "gpt": conv.roles[1]} #角色

    # 应用提示模板
    conversations = [] #对话
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]: #如果说话者不是人类
            # 如果说话者不是人类，跳过第一个
            source = source[1:]

        conv.messages = [] #消息
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]] #角色
            assert role == conv.roles[j % 2], f"{i}" #如果角色不匹配
            conv.append_message(role, sentence["value"]) #添加消息
        conversations.append(conv.get_prompt()) #添加对话


    # 标记化对话
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids #标记化

    torch.set_printoptions(threshold=10_000) #设置打印选项



    targets = input_ids.clone() #目标



    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO #如果分隔符样式不是两个

    # 掩码目标
    # 我认为这里我们掩码分隔符sep后的所有内容

    sep = conv.sep + conv.roles[1] + ": " #分隔符

    for conversation, target in zip(conversations, targets):

        total_len = int(target.ne(tokenizer.pad_token_id).sum()) #目标长度
        rounds = conversation.split(conv.sep2) #分割对话
        cur_len = 1 #当前长度
        target[:cur_len] = IGNORE_INDEX #忽略索引
        for i, rou in enumerate(rounds):
            if rou == "": #如果回合为空
                break
            parts = rou.split(sep) #分割回合
            if len(parts) != 2: #如果部分长度不为2
                break
            parts[0] += sep #添加分隔符

            # PARTS[0] = "Questions + Assistant: ""
            round_len = len(tokenizer(rou).input_ids) #回合长度
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2 #指令长度

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX #忽略索引

            cur_len += round_len #当前长度
        target[cur_len:] = IGNORE_INDEX #忽略索引

        if cur_len < tokenizer.model_max_length: #如果当前长度小于模型最大长度
            if cur_len != total_len: #如果当前长度不等于目标长度
                target[:] = IGNORE_INDEX #忽略索引
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." #打印警告
                    f" (ignored)" #忽略
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy() #复制对话
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]} #角色

    # 应用提示模板
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]: #如果说话者不是人类
            # 如果说话者不是人类，跳过第一个
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]] #角色
            assert role == conv.roles[j % 2], f"{i}" #如果角色不匹配
            conv.append_message(role, sentence["value"]) #添加消息
        conversations.append(conv.get_prompt()) #添加对话

    # 标记化对话
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids #标记化
    targets = input_ids.clone() #目标
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT #如果分隔符样式不是多分隔符

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": " #分隔符
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) #目标长度

        rounds = conversation.split(conv.sep) #分割对话
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt # user + gpt
        cur_len = 0 #当前长度
        target[:cur_len] = IGNORE_INDEX #忽略索引
        for i, rou in enumerate(re_rounds):
            if rou == "": #如果回合为空
                break

            parts = rou.split(sep) #分割回合
            if len(parts) != 2: #如果部分长度不为2
                break
            parts[0] += sep #添加分隔符
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids) #回合长度
            instruction_len = len(tokenizer(parts[0]).input_ids) #指令长度
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX #忽略索引

            cur_len += round_len #当前长度
        target[cur_len:] = IGNORE_INDEX #忽略索引

        if cur_len < tokenizer.model_max_length: #如果当前长度小于模型最大长度
            if cur_len != total_len:
                target[:] = IGNORE_INDEX #忽略索引
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." #打印警告
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

# 预处理
def preprocess(
        sources: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1": #如果版本是v1
        print("go to preprocess v1") #打印
        return preprocess_v1(sources, tokenizer) #预处理
    if conversation_lib.default_conversation.version == "mpt": #如果版本是mpt
        return preprocess_mpt(sources, tokenizer)

    # 添加结束信号并连接在一起
    conversations = [] #对话
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n" #头
        conversation = _add_speaker_and_signal(header, source) #添加说话者和开始/结束信号
        conversations.append(conversation) #添加对话
    # 标记化对话
    conversations_tokenized = _tokenize_fn(conversations, tokenizer) #标记化对话
    input_ids = conversations_tokenized["input_ids"] #输入ID
    targets = copy.deepcopy(input_ids) #目标
    for target, source in zip(targets, sources): #遍历目标和源
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], #标记化
                                      tokenizer)["input_ids_lens"] #标记化
        speakers = [sentence["from"] for sentence in source] #说话者
        _mask_targets(target, tokenized_lens, speakers) #掩码目标

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset): #监督数据集
    """Dataset for supervised fine-tuning.""" #监督数据集

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...") #加载数据
        list_data_dict = json.load(open(data_path, "r")) #加载数据

        logging.warning("Formatting inputs...") #格式化输入
        sources = [example["conversations"] for example in list_data_dict] #源
        data_dict = preprocess(sources, tokenizer) #预处理

        self.input_ids = data_dict["input_ids"] #输入ID
        self.labels = data_dict["labels"] #标签

    def __len__(self): #获取长度
        return len(self.input_ids) #返回输入ID的长度

    def __getitem__(self, i) -> Dict[str, torch.Tensor]: #获取项目
        return dict(input_ids=self.input_ids[i], labels=self.labels[i]) #返回输入ID和标签


class LazySupervisedDataset(Dataset): #懒惰监督数据集
    """Dataset for supervised fine-tuning.""" #监督数据集

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        print("INITIALIZING LAZY SUPERVISED DATASET") #初始化懒惰监督数据集
        logging.warning("Loading data...") #加载数据
        list_data_dict = json.load(open(data_path, "r")) #加载数据

        logging.warning("Formatting inputs...Skip in lazy mode") #格式化输入
        self.tokenizer = tokenizer #标记器
        self.list_data_dict = list_data_dict #列表数据字典
        self.multimodal_cfg = multimodal_cfg #多模态配置

    def __len__(self): #获取长度
        return len(self.list_data_dict) #返回列表数据字典的长度

    def __getitem__(self, i) -> Dict[str, torch.Tensor]: #获取项目

        torch.set_printoptions(threshold=10_000)

        print("GET ITEM FROM LAZY") #获取项目
        sources = self.list_data_dict[i] #源



        if isinstance(i, int): #如果i是整数
            sources = [sources] #源
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME # FIXME

        """# LOAD FEATURES IF THEY ARE THERE
        if 'video' in sources[0]:
            video_file = self.list_data_dict[i]['video']
            video_folder = self.multimodal_cfg['video_folder']
            with open(f"{video_folder}/{video_file}", "rb") as f:
                features = pickle.load(f)

            cur_token_len = 356  # 100 temporal + 256 spatial, TODO: Hard Coding is not good
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.multimodal_cfg, cur_token_len)"""

        data_dict = preprocess(
            sources,
            self.tokenizer)


        if isinstance(i, int): #如果i是整数
            data_dict = dict(input_ids=data_dict["input_ids"][0], #输入ID
                             labels=data_dict["labels"][0]) #标签

        # video exist in the data
        if 'video' in self.list_data_dict[i]: #如果视频在数据中
            data_dict["video"] = features #视频

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object): #监督数据集的整理器
    """Collate examples for supervised fine-tuning.""" #监督数据集的整理器

    tokenizer: transformers.PreTrainedTokenizer #标记器

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]: #调用

        print("START DataCollatorForSupervisedDataset") #开始监督数据集的整理器

        input_ids, labels = tuple([instance[key] for instance in instances] #输入ID和标签
                                  for key in ("input_ids", "labels"))

        torch.set_printoptions(threshold=10_000) #设置打印选项


        input_ids = torch.nn.utils.rnn.pad_sequence( #填充序列
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id) #填充序列
        labels = torch.nn.utils.rnn.pad_sequence(labels, #填充序列
                                                 batch_first=True, #批量第一
                                                 padding_value=IGNORE_INDEX) #忽略索引

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'video' in instances[0]: #如果视频在数据中
            features = [torch.tensor(instance['video']) for instance in instances] #视频
            
            if all(x is not None and x.shape == features[0].shape for x in features): #如果所有特征不为空且形状相同
                print(torch.stack(features).shape) #打印特征形状
                batch['video_spatio_temporal_features'] = torch.stack(features) #堆叠特征
            else:
                print(features.shape) #打印特征形状
                batch['video_spatio_temporal_features'] = features #视频空间时间特征


        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, #标记器
                                data_args) -> Dict: #数据参数
    """Make dataset and collator for supervised fine-tuning.""" #制作监督数据集和整理器
    dataset_cls = (LazySupervisedDataset #懒惰监督数据集
                   if data_args.lazy_preprocess else SupervisedDataset) #监督数据集


    # 创建数据集
    train_dataset = dataset_cls(tokenizer=tokenizer, #标记器
                                data_path=data_args.data_path, #数据路径
                                multimodal_cfg=dict( #多模态配置
                                    is_multimodal=data_args.is_multimodal,
                                    sep_video_conv_front=data_args.sep_video_conv_front, #分隔视频对话前
                                    video_token_len=data_args.video_token_len, #视频token长度
                                    video_folder=data_args.video_folder, #视频文件夹
                                    frame_aspect_ratio=data_args.frame_aspect_ratio,
                                    use_vid_start_end=getattr(data_args, 'mm_use_vid_start_end', False)))


    train_loader = torch.utils.data.DataLoader(train_dataset, #训练数据集
        batch_size=1, shuffle=True, #批量大小和是否打乱
        num_workers=1, pin_memory=True) #工作线程数和是否使用内存

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer) #监督数据集的整理器


    return dict(train_dataset=train_dataset, #训练数据集
                eval_dataset=None, #评估数据集
                data_collator=data_collator) #监督数据集的整理器



def train(): #训练
    parser = transformers.HfArgumentParser( #参数解析器
        (ModelArguments, DataArguments, TrainingArguments)) #模型参数、数据参数和训练参数
    model_args, data_args, training_args = parser.parse_args_into_dataclasses() #解析参数

    print(model_args) #打印模型参数

    print(training_args) #打印训练参数

    model = VideoChatGPTLlamaForCausalLM.from_pretrained( #从预训练模型加载
        model_args.model_name_or_path, #模型名称或路径
        cache_dir=training_args.cache_dir, #缓存目录
        # torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float, #如果bf16为True，则使用bfloat16，否则使用float
    )
    model.config.use_cache = False #使用缓存

    if model_args.freeze_backbone: #如果冻结骨干网络
        model.model.requires_grad_(False) #冻结骨干网络

    tokenizer = transformers.AutoTokenizer.from_pretrained( #从预训练模型加载标记器
        model_args.model_name_or_path, #模型名称或路径
        cache_dir=training_args.cache_dir, #缓存目录
        model_max_length=training_args.model_max_length, #模型最大长度
        padding_side="right", #填充侧
        use_fast=False, #使用快速
    )

    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"] #默认对话

    model_vision_dict = model.get_model().initialize_vision_modules( #初始化视觉模块
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter #预训练mm_mlp_adapter
    )
    vision_config = model_vision_dict['vision_config'] #视觉配置


    data_args.video_token_len = model_vision_dict['video_token_len'] #视频token长度
    data_args.is_multimodal = False #是否多模态

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter #tune_mm_mlp_adapter

    if model_args.tune_mm_mlp_adapter: #如果tune_mm_mlp_adapter
        model.requires_grad_(False) #冻结模型
        for p in model.get_model().mm_projector.parameters(): #遍历mm_projector的参数
            p.requires_grad = True #需要梯度

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter #冻结mm_mlp_adapter

    if training_args.freeze_mm_mlp_adapter: #如果冻结mm_mlp_adapter
        for p in model.get_model().mm_projector.parameters(): #遍历mm_projector的参数
            p.requires_grad = False #不需要梯度

    model.config.mm_use_vid_start_end = data_args.mm_use_vid_start_end = model_args.mm_use_vid_start_end #mm_use_vid_start_end
    vision_config.use_vid_start_end = training_args.use_vid_start_end = model_args.mm_use_vid_start_end #use_vid_start_end


    model.config.sep_video_conv_front = data_args.sep_video_conv_front #分隔视频对话前


    model.initialize_vision_tokenizer(mm_use_vid_start_end=model_args.mm_use_vid_start_end, #mm_use_vid_start_end
                                      tokenizer=tokenizer, #标记器
                                      device=training_args.device, #设备
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter, #tune_mm_mlp_adapter
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter) #预训练mm_mlp_adapter


    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad] #不需要梯度的参数
    if len(params_no_grad) > 0: #如果不需要梯度的参数大于0
        if training_args.fsdp is not None and len(training_args.fsdp) > 0: #如果fsdp不为空且长度大于0
            if len(params_no_grad) < 10: #如果不需要梯度的参数小于10
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format( #打印警告
                    len(params_no_grad), params_no_grad)) #打印不需要梯度的参数
            else: #如果不需要梯度的参数大于10
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format( #打印警告
                        len(params_no_grad), ', '.join(params_no_grad[:10]))) #打印不需要梯度的参数
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.") #打印警告
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func): #补丁FSDP使用原始参数
                def wrap_func(*args, **kwargs): #包装函数
                    use_orig_params = kwargs.pop('use_orig_params', True) #使用原始参数
                    return func(*args, **kwargs, use_orig_params=use_orig_params) #返回函数

                return wrap_func #返回包装函数

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__) #补丁FSDP使用原始参数


    ######## HERE #########

    data_module = make_supervised_data_module(tokenizer=tokenizer, #标记器
                                              data_args=data_args) #数据参数

    dataset = data_module["train_dataset"] #训练数据集

    train_loader = torch.utils.data.DataLoader(dataset, #训练数据集
        batch_size=1, shuffle=True, #批量大小和是否打乱
        num_workers=0, pin_memory=True) #工作线程数和是否使用内存



    ######################

    training_args.report_to = [] #报告到

    print(training_args) #打印训练参数

    # training_args.max_steps = 10
    trainer = VideoChatGPTTrainer(model=model, #模型
                                  tokenizer=tokenizer, #标记器
                                  args=training_args, #训练参数
                                  **data_module) #数据模块

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")): #如果输出目录存在检查点
        trainer.train(resume_from_checkpoint=True) #从检查点恢复训练
    else: #如果输出目录不存在检查点
        trainer.train() #训练
    trainer.save_state() #保存状态
    safe_save_model_for_hf_trainer(trainer=trainer, #训练器
                                   output_dir=training_args.output_dir) #输出目录



if __name__ == "__main__":
    train()
