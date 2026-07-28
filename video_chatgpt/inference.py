'''
推理
'''

from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import torch

# 定义常量
DEFAULT_VIDEO_TOKEN = "<video>" #默认视频标记
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>" #默认视频补丁标记
DEFAULT_VID_START_TOKEN = "<vid_start>" #默认视频开始标记
DEFAULT_VID_END_TOKEN = "<vid_end>" #默认视频结束标记


def get_spatio_temporal_features_torch(features): #计算空间时间特征
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # 提取特征的维度
    t, s, c = features.shape

    # 计算时间轴上的平均值
    temporal_tokens = torch.mean(features, dim=1)

    # 计算填充大小
    padding_size = 100 - t

    # 如果需要，填充时间轴上的tokens
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # 计算空间轴上的平均值
    spatial_tokens = torch.mean(features, dim=0)

    # 连接时间轴和空间轴上的tokens并转换为半精度
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens

# 视频ChatGPT推理
def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # 准备问题字符串
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # 准备对话提示
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 标记化提示
    inputs = tokenizer([prompt])

    # 预处理视频帧并获取图像张量
    image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

    # 将图像张量移动到GPU并降低精度到半精度
    image_tensor = image_tensor.half().cuda()

    # 生成视频空间时间特征
    with torch.no_grad():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
    video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

    # 将输入移动到GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # 定义生成停止条件
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # 运行模型推理
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    # 检查输出是否与输入相同
    n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # 解码输出tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # 清理输出字符串
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs