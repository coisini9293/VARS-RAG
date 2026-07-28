'''
模型相关的工具函数
权重加载、参数处理、数据格式转换、日志等工具
'''


import torch
from video_chatgpt.model import *
from transformers import StoppingCriteria

# 关键词停止准则
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords #关键词
        self.keyword_ids = [tokenizer(keyword).input_ids for keyword in keywords] #关键词ID
        self.keyword_ids = [keyword_id[0] for keyword_id in self.keyword_ids if type(keyword_id) is list and len(keyword_id) == 1] #关键词ID
        self.tokenizer = tokenizer #tokenizer
        self.start_len = None #开始长度
        self.input_ids = input_ids #输入ID

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None: #如果开始长度为空，则设置为输入ID的长度
            self.start_len = self.input_ids.shape[1] #设置开始长度
        else: #如果开始长度不为空，则检查关键词ID
            for keyword_id in self.keyword_ids: #检查关键词ID
                if output_ids[0, -1] == keyword_id: #如果输出ID的最后一个元素等于关键词ID，则返回True
                    return True
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0] #解码输出ID
            for keyword in self.keywords: #检查关键词
                if keyword in outputs: #如果关键词在输出中，则返回True
                    return True
        return False #如果关键词不在输出中，则返回False
