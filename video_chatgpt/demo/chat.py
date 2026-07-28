import time
import torch
import gradio as gr
from video_chatgpt.utils import (build_logger) #用于记录日志
from conversation_discussion import conv_templates, SeparatorStyle #用于构建对话模板和分隔符样式
from conversation_discussion import load_video #用于加载视频
from video_chatgpt.model.utils import KeywordsStoppingCriteria #用于定义关键词停止条件
import logging
from video_chatgpt.constants import * #用于定义常量，包括视频分隔符、默认视频标记等
import numpy as np
from visual_encoder.config.classes import INVERSE_EVENT_DICTIONARY #用于定义事件字典，将事件编号转换为事件名称
import pickle
import json


logging.basicConfig(level=logging.WARNING) #设置日志级别，只显示警告和错误信息

logger = build_logger("gradio_web_server", "gradio_web_server.log") #创建日志记录器

headers = {"User-Agent": "Video-ChatGPT"} #设置请求头，用于标识请求来源

no_change_btn = gr.Button.update() #更新按钮状态，不进行任何操作
enable_btn = gr.Button.update(interactive=True) #更新按钮状态，设置为可交互
disable_btn = gr.Button.update(interactive=False) #更新按钮状态，设置为不可交互


def post_process_code(code): #用于处理代码，将代码中的换行符替换为空格
    sep = "\n```" #代码块分隔符
    if sep in code: #如果代码块分隔符在代码中
        blocks = code.split(sep) #将代码按分隔符分割
        if len(blocks) % 2 == 1: #如果代码块数量为奇数
            for i in range(1, len(blocks), 2): #遍历奇数块
                blocks[i] = blocks[i].replace("\\_", "_") #将代码中的下划线替换为空格
        code = sep.join(blocks) #将代码块重新拼接
    return code #返回处理后的代码

#定义Chat类，用于处理对话
class Chat:
    def __init__(self, model_name, conv_mode, tokenizer, image_processor, vision_tower, model, replace_token):  #初始化Chat类，设置模型参数
        self.model_name = model_name #模型名称
        self.conv_mode = conv_mode #对话模式
        self.tokenizer = tokenizer #分词器
        self.image_processor = image_processor #图像处理器
        self.vision_tower = vision_tower #视觉塔
        self.model = model #模型
        self.replace_token = replace_token #替换标记

    def upload_video(self, video, img_list): #上传视频，将视频转换为图像列表
        if isinstance(video, str):  #如果视频是字符串，则认为是一个路径
            frames = load_video(video) #加载视频

            start_frame = 63 #开始帧
            end_frame = 87 #结束帧
            fps = 17 #帧率
            fps_beginning = 25 #开始帧率
            factor = (end_frame - start_frame) / (((end_frame - start_frame) / fps_beginning) * fps) #计算因子

            frames = frames[start_frame:end_frame] #截取视频
            frames = self.image_processor.preprocess(frames, return_tensors='pt')['pixel_values'] #预处理图像

            final_frames = None #初始化最终帧

            # 我们提取尽可能多的帧，以每秒17帧的速度
            for j in range(len(frames)): #遍历帧
                if j%factor<1: #如果帧索引与因子相除的余数小于1
                    if final_frames == None: #如果最终帧为空
                        final_frames = frames[j,:,:,:].unsqueeze(0) #将帧转换为张量
                    else:
                        final_frames = torch.cat((final_frames, frames[j,:,:,:].unsqueeze(0)), 0) #将帧拼接

            img_list.append(final_frames.cuda()) #将最终帧添加到图像列表
        else:
            raise NotImplementedError #如果视频不是字符串，则抛出错误
        msg = "Received." #返回消息
        return msg #返回消息
    

    def get_spatio_temporal_features_torch(self, features, num_temporal_tokens=44): #获取时空特征，计算时空特征的平均值
        t, s, c = features.shape #获取特征的形状

        temporal_tokens = np.mean(features, axis=1) #计算时空特征的平均值
        padding_size = num_temporal_tokens - t #计算填充大小
        if padding_size > 0: #如果填充大小大于0
            temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant') #填充时空特征

        spatial_tokens = np.mean(features, axis=0) #计算空间特征的平均值
        sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0) #将时空特征和空间特征拼接

        return sp_features #返回时空特征

    def answer(self, state, img_list, temperature, max_new_tokens, first_run): #回答问题，生成对话
        if state.skip_next: #如果跳过下一个输入
            # 由于无效输入，跳过此生成调用
            yield (state, state.to_gradio_chatbot(), img_list, first_run) + (no_change_btn,) * 5 #跳过下一个输入
            return
        
        image_tensor = img_list[0] #获取图像张量
        # Generate video spatio-temporal features
        image_tensor = image_tensor.cuda() #将图像张量转换为cuda张量


        with torch.no_grad(): #禁用梯度计算，提高计算效率
            out_off, out_act, frame_features = self.vision_tower(image_tensor) #获取输出，包括犯规、动作和帧特征

            preds_sev = torch.argmax(out_off.detach().cpu(), 0) #获取输出，犯规类别
            preds_act = torch.argmax(out_act.detach().cpu(), 0) #获取输出，动作类别

            values = {} #初始化值
            action_class = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()] #获取动作类
            if preds_sev.item() == 0:
                offence_class = "No offence" #犯规类别为无犯规
                severity_class = "" #严重程度为空
            elif preds_sev.item() == 1:
                offence_class = "Offence" #犯规类别为犯规
                severity_class = "1.0" #严重程度为1.0
            elif preds_sev.item() == 2:
                offence_class = "Offence" #犯规类别为犯规
                severity_class = "3.0" #严重程度为3.0
            elif preds_sev.item() == 3:
                offence_class = "Offence" #犯规类别为犯规
                severity_class = "5.0" #严重程度为5.0

            if offence_class == "Offence": #如果犯规类别为犯规
                offence_class = ", foul" #犯规类别为犯规，添加犯规

            if offence_class == "No offence": #如果犯规类别为无犯规
                offence_class = " and no foul " #犯规类别为无犯规，添加无犯规

            if severity_class == "1.0": #如果严重程度为1.0
                offence_class += " and no card" #严重程度为1.0，添加无黄牌
            
            if severity_class == "3.0": #如果严重程度为3.0
                offence_class += " and a yellow card" #严重程度为3.0，添加黄牌

            if severity_class == "5.0": #如果严重程度为5.0
                offence_class += " and a red card" #严重程度为5.0，添加红牌

            if action_class == "Tackling": #如果动作类别为铲球
                action_class = "a tackle" #动作类别为铲球，添加铲球
            if action_class == "Standing tackling": #如果动作类别为站立铲球
                action_class = "a foot duel" #动作类别为站立铲球，添加脚踢
            if action_class == "Elbowing": #如果动作类别为肘击
                action_class = "using his elbows or arms" #动作类别为肘击，添加肘击
            if action_class == "Holding": #如果动作类别为抱摔
                action_class = "holding" #动作类别为抱摔，添加抱摔
            if action_class == "High leg": #如果动作类别为高腿
                action_class = "a high leg" #动作类别为高腿，添加高腿
            if action_class == "Pushing": #如果动作类别为推搡
                action_class = "pushing" #动作类别为推搡，添加推搡
            if action_class == "Challenge": #如果动作类别为挑战
                action_class = "a shoulder challenge" #动作类别为挑战，添加肩部挑战
            if action_class == "Dive": #如果动作类别为假摔
                action_class = "a simulation" #动作类别为假摔，添加假摔
            
        if first_run: #如果第一次运行
            conv_mode = self.conv_mode #获取对话模式
            new_state = conv_templates[conv_mode].copy() #复制对话模板
            new_state.append_message(new_state.roles[0], state.messages[-2][1]) #添加消息
            new_state.append_message(new_state.roles[1], None) #添加消息
            state = new_state #更新状态
            first_run = False #更新第一次运行


        f = open("annotations/predictionsTest_clip.json") #打开注释文件
        data = json.load(f) #加载注释文件
        pred = data["Actions"] #获取动作

        path = "/gpfs/scratch/acad/telim/VARS/dataset/Test/action_115/PRE_CLIP_feature_clip_3.pkl" #获取路径

        pred_action = pred[path]["Action class"] #获取动作类别
        pred_off = pred[path]["Offence"] #获取犯规
        pred_card = pred[path]["Severity"] #获取严重程度

        if pred_off == "Offence": #如果犯规类别为犯规
            pred_off = ", foul" #犯规类别为犯规，添加犯规

        if pred_off == "No offence": #如果犯规类别为无犯规
            pred_off = " and no foul " #犯规类别为无犯规，添加无犯规

        if pred_card == "1.0": #如果严重程度为1.0
            pred_off += " and no card" #严重程度为1.0，添加无黄牌
        
        if pred_card == "3.0": #如果严重程度为3.0
            pred_off += " and a yellow card" #严重程度为3.0，添加黄牌

        if pred_card == "5.0": #如果严重程度为5.0
            pred_off += " and a red card" #严重程度为5.0，添加红牌

        if pred_action == "Tackling": #如果动作类别为铲球
            pred_action = "a tackle" #动作类别为铲球，添加铲球
        if pred_action == "Standing tackling": #如果动作类别为站立铲球
            pred_action = "a foot duel" #动作类别为站立铲球，添加脚踢
        if pred_action == "Elbowing": #如果动作类别为肘击
            pred_action = "using his elbows or arms" #动作类别为肘击，添加肘击
        if pred_action == "Holding": #如果动作类别为抱摔
            pred_action = "holding" #动作类别为抱摔，添加抱摔
        if pred_action == "High leg": #如果动作类别为高腿
            pred_action = "a high leg" #动作类别为高腿，添加高腿
        if pred_action == "Pushing": #如果动作类别为推搡
            pred_action = "pushing" #动作类别为推搡，添加推搡
        if pred_action == "Challenge": #如果动作类别为挑战
            pred_action = "a shoulder challenge" #动作类别为挑战，添加肩部挑战
        if pred_action == "Dive": #如果动作类别为假摔
            pred_action = "a simulation" #动作类别为假摔，添加假摔

        state.set_predictions(pred_action, pred_off) #设置预测
        # Construct prompt
        prompt = state.get_prompt() #获取提示

        prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, self.replace_token, 1) #替换视频标记

        inputs = self.tokenizer([prompt]) #分词

        input_ids = torch.as_tensor(inputs.input_ids).cuda() #将输入转换为cuda张量

        stop_str = state.sep if state.sep_style != SeparatorStyle.TWO else state.sep2 #获取分隔符
        keywords = [stop_str] #获取关键词
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids) #获取停止条件

        # 取消注释此行以进行调试
        # pload = { #请求参数
        #     "model": self.model_name, #模型名称
        #     "prompt": prompt, #提示
        #     "temperature": float(temperature), #温度
        #     "max_new_tokens": min(int(max_new_tokens), 1536), #最大新标记数
        #     "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2, #停止条件  
        # }
        # logger.info(f"==== request ====\n{pload}") #记录请求参数

        state.messages[-1][-1] = "" #清空消息
        yield (state, state.to_gradio_chatbot(), img_list, first_run) + (disable_btn,) * 5 #禁用按钮

        video_spatio_temporal_features = self.get_spatio_temporal_features_torch(frame_features.numpy().astype("float16")) #获取时空特征

        video_spatio_temporal_features = torch.from_numpy(video_spatio_temporal_features).unsqueeze(0) #将时空特征转换为张量
        video_spatio_temporal_features = video_spatio_temporal_features.cuda() #将时空特征转换为cuda张量

        with open(path, 'rb') as f: #打开路径
            video_spatio_temporal_features = torch.from_numpy(pickle.load(f)).cuda() #将时空特征转换为张量
            video_spatio_temporal_features = video_spatio_temporal_features.unsqueeze(0) #将时空特征转换为张量


        with torch.inference_mode(): #禁用梯度计算，提高计算效率
            output_ids = self.model.generate( #生成输出
                input_ids, #输入
                video_spatio_temporal_features=video_spatio_temporal_features, #时空特征
                do_sample=False, #是否采样
                temperature=float(temperature), #温度
                max_new_tokens=min(int(max_new_tokens), 1536), #最大新标记数
                stopping_criteria=[stopping_criteria]) #停止条件

        input_token_len = input_ids.shape[1] #获取输入标记长度
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item() #计算输入和输出之间的差异
        if n_diff_input_output > 0: #如果输入和输出之间的差异大于0
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids') #打印警告
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0] #解码输出
        outputs = outputs.strip() #去除输出两端的空白字符
        if outputs.endswith(stop_str): #如果输出以停止字符串结尾
            outputs = outputs[:-len(stop_str)] #去除停止字符串
        outputs = outputs.strip() #去除输出两端的空白字符
        output = post_process_code(outputs) #处理输出
        for character in output: #遍历输出
            state.messages[-1][-1] += character #添加字符
            time.sleep(0.01) #等待0.01秒
            yield (state, state.to_gradio_chatbot(), img_list, first_run) + (enable_btn,) * 5 #更新状态
        # state.messages[-1][-1] = state.messages[-1][-1][:-1] #去除最后一个字符
        logger.info(f"{output}") #记录输出
        yield (state, state.to_gradio_chatbot(), img_list, first_run) + (enable_btn,) * 5 #更新状态
