'''
一致性基准测试
给定一组相关问题，模型的回答是否前后一致、逻辑自洽
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
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--projection_path", type=str, required=True)

    return parser.parse_args() #返回参数

# 运行推理
def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # 初始化模型
    model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(args.model_name,
                                                                                        args.projection_path)
    # 加载地面真值文件
    with open(args.gt_file) as file:
        gt_contents = json.load(file)

    # 如果输出目录不存在，则创建它
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # 列表来存储输出结果
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # 遍历地面真值文件中的每个样本
    for sample in tqdm(gt_contents):
        video_name = sample['video_name']
        sample_set = sample
        question_1 = sample['Q1']
        question_2 = sample['Q2']

        # 加载视频文件
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # 检查视频是否存在
        if video_path is not None:  # Modified this line
            video_frames = load_video(video_path)

        try:
            # 在视频上运行推理并添加输出到列表
            output_1 = video_chatgpt_infer(video_frames, question_1, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred1'] = output_1

            # 在视频上运行推理并添加输出到列表
            output_2 = video_chatgpt_infer(video_frames, question_2, conv_mode, model, vision_tower,
                                             tokenizer, image_processor, video_token_len)
            sample_set['pred2'] = output_2

            output_list.append(sample_set)
        except Exception as e:
            print(f"Error processing video file '{video_name}': {e}")

    # 将输出列表保存到JSON文件
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args() #解析参数
    run_inference(args) #运行推理
