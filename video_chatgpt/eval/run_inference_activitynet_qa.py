'''
ActivityNet-QA 视频问答预测
评估模型在标准视频问答基准（ActivityNet-QA）上的表现
'''

import os
import argparse
import json
from tqdm import tqdm
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer

# 解析参数
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # 定义命令行参数
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True) #设置视频目录
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True) #设置问题文件
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True) #设置答案文件
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True) #设置输出目录
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True) #设置输出名称
    parser.add_argument("--model-name", type=str, required=True) #设置模型名称
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1') #设置对话模式
    parser.add_argument("--projection_path", type=str, required=True) #设置投影路径

    return parser.parse_args() #返回参数


# 运行推理
def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # 初始化模型
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # 加载包含问题和答案的地面真值文件
    with open(args.gt_file_question) as file:
        gt_questions = json.load(file)
    with open(args.gt_file_answers) as file:
        gt_answers = json.load(file)

    # 如果输出目录不存在，则创建它
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # 列表来存储输出结果
    conv_mode = args.conv_mode #对话模式    

    video_formats = ['.mp4', '.avi', '.mov', '.mkv'] #视频格式

    # 遍历地面真值文件中的每个样本
    index = 0 #索引
    for sample in tqdm(gt_questions):
        video_name = sample['video_name'] #视频名称
        question = sample['question'] #问题
        id = sample['question_id'] #问题ID
        answer = gt_answers[index]['answer'] #答案
        index += 1 #索引加1

        sample_set = {'id': id, 'question': question, 'answer': answer} #样本集

        # 加载视频文件
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}") #临时路径
            if os.path.exists(temp_path):
                video_path = temp_path #视频路径
                break

        # 检查视频是否存在
        if os.path.exists(video_path):
            video_frames = load_video(video_path) #加载视频帧

        try:
            # 在视频上运行推理并添加输出到列表
            output = video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred'] = output #样本集的预测
            output_list.append(sample_set) #输出列表
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # 将输出列表保存到JSON文件
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file) #将输出列表保存到JSON文件


if __name__ == "__main__":
    args = parse_args() #解析参数
    run_inference(args) #运行推理
