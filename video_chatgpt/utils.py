'''
工具函数
'''

import logging
import logging.handlers
import os
import sys

import requests

from video_chatgpt.constants import LOGDIR
# 定义常量
server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None #处理器

# 构建日志记录器
def build_logger(logger_name, logger_filename):
    global handler #全局处理器

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置根处理程序的格式
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # 只重定向stdout，不重定向stderr以避免递归错误
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    # 注释掉stderr重定向，避免递归错误
    # stderr_logger = logging.getLogger("stderr")
    # stderr_logger.setLevel(logging.ERROR)
    # sl = StreamToLogger(stderr_logger, logging.ERROR)
    # sys.stderr = sl

    # 获取日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 为所有日志记录器添加文件处理程序
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


# 流到日志记录器
class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO): #初始化
        self.terminal = sys.stdout #终端
        self.logger = logger #日志记录器
        self.log_level = log_level #日志级别
        self.linebuf = '' #行缓冲区

    def __getattr__(self, attr): #获取属性
        return getattr(self.terminal, attr) #获取属性

    def write(self, buf): #写入
        temp_linebuf = self.linebuf + buf #临时行缓冲区
        self.linebuf = '' #行缓冲区
        for line in temp_linebuf.splitlines(True): #分割行
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self): #刷新
        if self.linebuf != '': #如果行缓冲区不为空
            self.logger.log(self.log_level, self.linebuf.rstrip()) #记录日志
        self.linebuf = '' #行缓冲区


def disable_torch_init(): #禁用torch初始化
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None) #设置线性层的参数
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None) #设置层归一化的参数


def violates_moderation(text): #违反监管
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations" #URL
    headers = {"Content-Type": "application/json", #头部
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]} #授权
    text = text.replace("\n", "") #替换换行符
    data = "{" + '"input": ' + f'"{text}"' + "}" #数据
    data = data.encode("utf-8") #编码
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5) #请求
        flagged = ret.json()["results"][0]["flagged"] #标记
    except requests.exceptions.RequestException as e: #请求异常
        flagged = False #标记
    except KeyError as e: #键错误
        flagged = False #标记

    return flagged #返回标记


def pretty_print_semaphore(semaphore): #漂亮的打印信号量
    if semaphore is None: #如果信号量为空
        return "None" #返回None
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})" #返回信号量
