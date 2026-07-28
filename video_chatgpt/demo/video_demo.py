import os
import argparse
import datetime
import json
import time
import gradio as gr
from video_chatgpt.video_conversation import (default_conversation)
from video_chatgpt.utils import (build_logger, violates_moderation, moderation_msg)
from video_chatgpt.demo.gradio_patch import Chatbot as grChatbot
from video_chatgpt.utils import disable_torch_init
from video_chatgpt.demo.chat import Chat
from video_chatgpt.demo.template import tos_markdown, css, title, disclaimer, Seafoam
from video_chatgpt.eval.model_utils import initialize_model
from video_chatgpt.constants import *

import logging

logging.basicConfig(level=logging.WARNING) #设置日志级别

logger = build_logger("gradio_web_server", "gradio_web_server.log") #构建日志记录器

headers = {"User-Agent": "Video-ChatGPT"} #设置请求头

no_change_btn = gr.Button.update() #更新按钮，不改变按钮状态
enable_btn = gr.Button.update(interactive=True) #更新按钮，改变按钮状态为可交互
disable_btn = gr.Button.update(interactive=False) #更新按钮，改变按钮状态为不可交互


def get_conv_log_filename(): #获取日志文件名
    t = datetime.datetime.now() #获取当前时间
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json") #拼接日志文件名
    return name

# 获取窗口URL参数
get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""

# 加载demo
def load_demo(url_params): #加载demo
    logger.info(f"load_demo.. params: {url_params}") #记录日志

    state = default_conversation.copy() #复制对话状态
    return (state, #返回状态
            gr.Chatbot.update(visible=True), #更新聊天记录，设置为可见
            gr.Textbox.update(visible=True), #更新输入框，设置为可见
            gr.Button.update(visible=True), #更新按钮，设置为可见
            gr.Button.update(visible=True), #更新按钮，设置为可见
            gr.Row.update(visible=True), #更新行，设置为可见
            gr.Accordion.update(visible=True)) #更新折叠面板，设置为可见

# 投票
def vote_last_response(state, vote_type):
    with open(get_conv_log_filename(), "a") as fout: #打开日志文件
        data = {
            "tstamp": round(time.time(), 4), #时间戳
            "type": vote_type, #投票类型
            "state": state.dict() #状态
        }
        logger.info(f"{get_conv_log_filename}.") #记录日志
        fout.write(json.dumps(data) + "\n") #写入日志

# 点赞
def upvote_last_response(state, imagebox):
    logger.info(f"upvote.") #记录日志
    vote_last_response(state, "upvote") #投票
    return ("", imagebox) + (disable_btn,) * 3 #返回状态

# 踩
def downvote_last_response(state, imagebox):
    logger.info(f"downvote.") #记录日志
    vote_last_response(state, "downvote") #投票
    return ("", imagebox) + (disable_btn,) * 3 #返回状态

# 标记
def flag_last_response(state, imagebox):
    logger.info(f"flag.") #记录日志
    vote_last_response(state, "flag") #投票
    return ("", imagebox) + (disable_btn,) * 3 #返回状态

# 重新生成
def regenerate(state):
    logger.info(f"regenerate.") #记录日志
    state.messages[-1][-1] = None #设置消息为空
    state.skip_next = False #设置跳过下一个
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5 #返回状态

# 清除历史记录
def clear_history(img_list):
    logger.info(f"clear_history.") #记录日志
    state = default_conversation.copy() #复制对话状态
    if img_list is not None: #如果图片列表不为空
        img_list = [] #设置图片列表为空
    return (state, state.to_gradio_chatbot(), "", gr.update(value=None, interactive=True), #更新输入框
            gr.update(value="Upload Video", interactive=True), img_list #更新按钮
            ) + (disable_btn,) * 5 #返回状态

# 添加文本
def add_text(state, text, image, first_run):
    logger.info(f"add_text. ip:. len: {len(text)}") #记录日志
    if len(text) <= 0 and image is None: #如果文本为空且图片为空
        state.skip_next = True #设置跳过下一个
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5 #返回状态
    if args.moderate:
        flagged = violates_moderation(text) #检查文本是否违规
        if flagged: #如果违规
            state.skip_next = True #设置跳过下一个
            return (state, state.to_gradio_chatbot(), moderation_msg, None) + (
                no_change_btn,) * 5

    text = text[:1536]  # 截断文本
    if first_run: #如果第一次运行
        text = text[:1200]  # 截断文本
        if '<video>' not in text:
            text = text + '\n<video>' #添加视频
        text = (text, image) #设置文本和图片
        state = default_conversation.copy() #复制对话状态
    state.append_message(state.roles[0], text) #添加消息
    state.append_message(state.roles[1], None) #添加消息
    state.skip_next = False #设置跳过下一个
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5 #返回状态

# 上传图片
def upload_image(image, state):
    if image is None: #如果图片为空
        return None, gr.update(interactive=True), state, None #返回状态
    state = default_conversation.copy() #复制对话状态
    img_list = [] #设置图片列表为空
    first_run = True #设置第一次运行
    llm_message = chat.upload_video(image, img_list) #上传视频
    return gr.update(interactive=False), gr.update(value="Start Chatting",
                                                   interactive=False), state, img_list, first_run #返回状态

# 构建主题
seafoam = Seafoam()

# 构建demo
def build_demo(embed_mode):
    textbox = gr.Textbox(show_label=False, #不显示标签
                         placeholder="Please upload your video first by clicking 'Upload Video'. Enter text and press ENTER", #设置占位符
                         visible=True).style(container=False) #设置文本框
    with gr.Blocks(title="Oryx Video-ChatGPT", theme=seafoam, css=css) as demo:
        state = gr.State() #状态
        img_list = gr.State() #图片列表
        first_run = gr.State() #第一次运行

        if not embed_mode:
            gr.Markdown(title) #显示标题
            # gr.Markdown(description) #显示描述

        with gr.Row():
            with gr.Column(scale=3):
                imagebox = gr.Video() #显示视频
                upload_button = gr.Button(value="Upload Video", interactive=True, variant="primary") #显示上传按钮

                # Add a text note beneath the button
                gr.Markdown(
                    "NOTE: Please make sure you **<span style='color:red'>press the 'Upload Video' button</span>**"
                    " and wait for it to display 'Start Chatting' "
                    "before submitting question to Video-ChatGPT.",
                    style="color:gray") #显示提示
                cur_dir = os.path.dirname(os.path.abspath(__file__))
                gr.Examples(examples=[
                    [f"{cur_dir}/demo_sample_videos/sample_2.mp4", "Why is this video strange?"], #显示示例
                    [f"{cur_dir}/demo_sample_videos/sample_6.mp4",
                     "Can you write a short poem inspired from the video."], #显示示例
                    [f"{cur_dir}/demo_sample_videos/sample_8.mp4",
                     "Where is this video taken? What place/landmark is shown in the video?"], #显示示例
                    [f"{cur_dir}/demo_sample_videos/sample_15.mp4",
                     "What is the main challenge faced by the people on the boat."], #显示示例
                ], inputs=[imagebox, textbox])

                with gr.Accordion("Parameters", open=False, visible=False) as parameter_row: #显示参数
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True, #显示温度
                                            label="Temperature", )
                    max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, #显示最大输出token
                                                  label="Max output tokens", )

            with gr.Column(scale=6):
                chatbot = grChatbot(elem_id="chatbot", label="VideoChat-GPT Chatbot", visible=True).style(height=600) #显示聊天记录
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render() #显示输入框
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit", visible=True) #显示提交按钮
                with gr.Row(visible=True) as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False) #显示点赞按钮
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False) #显示踩按钮
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False) #显示标记按钮
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False) #显示重新生成按钮
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False) #显示清除历史记录按钮

        gr.Markdown(disclaimer) #显示免责声明

        if not embed_mode:
            gr.Markdown(tos_markdown) #显示服务条款

        url_params = gr.JSON(visible=False) #显示URL参数
        # 注册监听器
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn] #显示按钮列表
        upvote_btn.click(upvote_last_response, #点击点赞按钮
                         [state, imagebox], [textbox, imagebox, upvote_btn, downvote_btn, flag_btn])
        downvote_btn.click(downvote_last_response, #点击踩按钮
                           [state, imagebox], [textbox, imagebox, upvote_btn, downvote_btn, flag_btn])
        flag_btn.click(flag_last_response, #点击标记按钮
                       [state, imagebox], [textbox, imagebox, upvote_btn, downvote_btn, flag_btn])
        regenerate_btn.click(regenerate, [state], #点击重新生成按钮
                             [state, chatbot, textbox, imagebox] + btn_list).then(
            chat.answer, [state, img_list, temperature, max_output_tokens, first_run], #点击重新生成按钮
            [state, chatbot, img_list, first_run] + btn_list)
        clear_btn.click(clear_history, [img_list], #点击清除历史记录按钮
                        [state, chatbot, textbox, imagebox, upload_button, img_list] + btn_list)

        upload_button.click(upload_image, [imagebox, state], #点击上传按钮
                            [imagebox, upload_button, state, img_list, first_run])

        textbox.submit(add_text, [state, textbox, imagebox, first_run], [state, chatbot, textbox] + btn_list #点击提交按钮
                       ).then(chat.answer, [state, img_list, temperature, max_output_tokens, first_run], #点击提交按钮
                              [state, chatbot, img_list, first_run] + btn_list)
        submit_btn.click(add_text, [state, textbox, imagebox, first_run], [state, chatbot, textbox] + btn_list #点击提交按钮
                         ).then(chat.answer, [state, img_list, temperature, max_output_tokens, first_run],
                                [state, chatbot, img_list, first_run] + btn_list)

        demo.load(load_demo, [url_params], #加载demo
                  [state, chatbot, textbox, upload_button, submit_btn, button_row, parameter_row],
                  _js=get_window_url_params)

    return demo

# 解析参数
def parse_args():
    parser = argparse.ArgumentParser(description="Demo") #创建参数解析器

    parser.add_argument("--host", type=str, default="0.0.0.0") #设置主机
    parser.add_argument("--port", type=int) #设置端口
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001") #设置控制器URL
    parser.add_argument("--concurrency-count", type=int, default=8) #设置并发数
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"]) #设置模型列表模式
    parser.add_argument("--share", action="store_true") #设置分享
    parser.add_argument("--moderate", action="store_true") #设置审核
    parser.add_argument("--embed", action="store_true") #设置嵌入
    parser.add_argument("--model-name", type=str, required=True) #设置模型名称
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14") #设置视觉塔名称
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt_v1") #设置对话模式
    parser.add_argument("--projection_path", type=str, required=False, default="") #设置投影路径

    args = parser.parse_args() #解析参数

    return args #返回参数


if __name__ == "__main__":
    args = parse_args() #解析参数
    logger.info(f"args: {args}") #记录日志
    logger.info(args) #记录日志
    disable_torch_init() #禁用torch初始化

    model, vision_tower, tokenizer, image_processor, video_token_len = \
        initialize_model(args.model_name, args.projection_path) #初始化模型

    # 创建替换token, 这将替换提示中的<video>
    replace_token = DEFAULT_VIDEO_PATCH_TOKEN * video_token_len #创建替换token
    replace_token = DEFAULT_VID_START_TOKEN + replace_token + DEFAULT_VID_END_TOKEN #创建替换token

    # 创建聊天
    chat = Chat(args.model_name, args.conv_mode, tokenizer, image_processor, vision_tower, model, replace_token) #创建聊天
    print('Initialization Finished') #记录日志

    demo = build_demo(args.embed) #创建demo
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10) #设置并发数
    demo.launch(share=True, enable_queue=True) #启动demo
