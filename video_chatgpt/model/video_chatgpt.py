'''
核心模型定义与推理逻辑
包含 Video-ChatGPT 模型的结构定义、加载、推理、输入输出处理等核心逻辑
'''

from typing import List, Optional, Tuple, Union
import logging, time
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

DEFAULT_VIDEO_TOKEN = "<video>" #视频标记
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>" #视频补丁标记
DEFAULT_VID_START_TOKEN = "<vid_start>" #视频开始标记
DEFAULT_VID_END_TOKEN = "<vid_end>" #视频结束标记


class VisionConfig: #视觉配置
    def __init__(self):
        self.frame_size = 224 #帧大小
        self.patch_size = 14 #补丁大小
        self.hidden_size = 1024 #隐藏大小
        self.use_vid_start_end = None #是否使用视频开始结束标记
        self.vid_start_token = None #视频开始标记
        self.vid_end_token = None #视频结束标记
        self.vid_patch_token = None #视频补丁标记

# 视频ChatGPT配置
class VideoChatGPTConfig(LlamaConfig):
    model_type = "VideoChatGPT" #模型类型

# 视频ChatGPT模型
class VideoChatGPTLlamaModel(LlamaModel):
    config_class = VideoChatGPTConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):  # TODO: Remove unused params
        super(VideoChatGPTLlamaModel, self).__init__(config)

        self.path_to_projection_layer = "video_chatgpt-7B.bin" #投影层路径

        if hasattr(config, "mm_vision_tower"): #如果配置中包含mm_vision_tower，则设置视觉配置
            self.vision_config = VisionConfig() #设置视觉配置

        # LOADING THE PRETRAINED PROJECTION LAYER
        if hasattr(config, "use_mm_proj"):
            print("LOAD PROJECTION LAYER") #加载投影层
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size) #设置投影层
        self._logger = logging.getLogger(__name__)

# 初始化视觉模块
    def initialize_vision_modules(self, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):

        vision_config = self.vision_config #获取视觉配置
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2 #计算补丁数量

        self.config.use_mm_proj = True #设置使用mm_proj
        self.config.mm_hidden_size = vision_config.hidden_size #设置隐藏大小

        return dict(
            video_token_len=num_patches, #设置视频补丁数量
            vision_config=vision_config #设置视觉配置
        )

# 前向传播
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, 'orig_embeds_params', None) #获取原始嵌入参数
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None: #如果输入嵌入为空，则设置为输入ID的嵌入
            t0 = time.time()
            self._logger.debug("[vcgpt] embedding tokens")
            inputs_embeds = self.embed_tokens(input_ids) #设置输入嵌入
            self._logger.debug(f"[vcgpt] embedding done | elapsed={time.time()-t0:.4f}s, shape={tuple(inputs_embeds.shape)}")

        if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None: #如果输入ID的形状不为1或训练，则设置视频空间时间特征
            

            #video_features = self.mm_projector_test(video_spatio_temporal_features) #测试投影层

            t1 = time.time()
            self._logger.debug("[vcgpt] mm_projector forward (video features)")
            video_features = self.mm_projector(video_spatio_temporal_features) #设置视频空间时间特征
            self._logger.debug(f"[vcgpt] projector done | elapsed={time.time()-t1:.4f}s, out_shape={tuple(video_features.shape)}")

            # 创建只有零的tensor
            dummy_video_features = torch.zeros(video_features.shape[1], 1024, device=inputs_embeds.device,
                                               dtype=inputs_embeds.dtype)
            # 将其推过与视频空间时间特征相同的线性层
            dummy_video_features = self.mm_projector(dummy_video_features)

            new_input_embeds = [] #新的输入嵌入
            cur_video_idx = 0 #当前视频索引
            t2 = time.time()
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): #遍历输入ID和输入嵌入
                
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0: #如果当前输入ID的补丁标记为0，则设置输入嵌入
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_video_features).sum() #设置输入嵌入
                    new_input_embeds.append(cur_input_embeds)
                    cur_video_idx += 1
                    continue
                if self.vision_config.use_vid_start_end: #如果使用视频开始结束标记，则设置输入嵌入
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != (
                            cur_input_ids == self.vision_config.vid_end_token).sum():
                        raise ValueError("The number of video start tokens and video end tokens should be the same.")
                    video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0] #获取视频开始标记

                    for video_start_token_pos in video_start_tokens: #遍历视频开始标记
                        cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device) #设置视频特征
                        num_patches = cur_video_features.shape[0] #获取补丁数量

                        if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token: #如果当前输入ID的补丁标记为0，则设置输入嵌入
                            raise ValueError("The video end token should follow the video start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos].detach(),
                                                              cur_input_embeds[
                                                              video_start_token_pos:video_start_token_pos + 1],
                                                              cur_video_features, cur_input_embeds[
                                                                                  video_start_token_pos + num_patches
                                                                                  + 1:video_start_token_pos
                                                                                  + num_patches + 2],
                                                              cur_input_embeds[
                                                              video_start_token_pos + num_patches + 2:].detach()),
                                                             dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos + 1],
                                                              cur_video_features,
                                                              cur_input_embeds[video_start_token_pos
                                                                               + num_patches + 1:]), dim=0)
                        cur_video_idx += 1 #当前视频索引加1
                    new_input_embeds.append(cur_new_input_embeds) #新的输入嵌入
                else:
                    cur_video_features = video_features[cur_video_idx] #设置视频特征
                    num_patches = cur_video_features.shape[0] #获取补丁数量
                    if (cur_input_ids == self.vision_config.vid_patch_token).sum() != num_patches: #如果当前输入ID的补丁标记为0，则设置输入嵌入
                        raise ValueError(
                            "The number of video patch tokens should be the same as the number of video patches.")
                    masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0] #获取补丁标记
                    mask_index_start = masked_indices[0] #获取补丁标记开始索引
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches,
                                                       device=masked_indices.device, dtype=masked_indices.dtype)).any(): #如果补丁标记不连续，则抛出异常
                        raise ValueError("The video patch tokens should be consecutive.")
                    if orig_embeds_params is not None: #如果原始嵌入参数不为空，则设置输入嵌入
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(),
                                                          cur_video_features,
                                                          cur_input_embeds[mask_index_start + num_patches:].detach()),
                                                         dim=0)
                    else: #如果原始嵌入参数为空，则设置输入嵌入
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_video_features,
                                                          cur_input_embeds[mask_index_start + num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds) #新的输入嵌入
                    cur_video_idx += 1 #当前视频索引加1
            inputs_embeds = torch.stack(new_input_embeds, dim=0) #新的输入嵌入
            self._logger.debug(f"[vcgpt] stitch video embeds done | elapsed={time.time()-t2:.4f}s, shape={tuple(inputs_embeds.shape)}")
        

        return super(VideoChatGPTLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


# 视频ChatGPTLlamaForCausalLM
class VideoChatGPTLlamaForCausalLM(LlamaForCausalLM):
    config_class = VideoChatGPTConfig #配置类

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config) #初始化

        self.model = VideoChatGPTLlamaModel(config) #设置模型

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) #设置线性层

        # Initialize weights and apply final processing
        self.post_init() #初始化

    def get_model(self): #获取模型
        return self.model #返回模型

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions #设置输出注意力
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states #设置输出隐藏状态
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict #设置返回字典

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model( #模型输出
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features
        )

        hidden_states = outputs[0] #隐藏状态

        logits = self.lm_head(hidden_states) #线性层输出
        logits_test = torch.argmax(logits, dim=2) #线性层输出


        loss = None #损失
        if labels is not None:
            #移位，使tokens < n预测n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            #展平tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            #启用模型/流水线并行
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:] #输出
            return (loss,) + output if loss is not None else output #返回损失和输出

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values: #如果past_key_values不为空，则设置输入ID
            input_ids = input_ids[:, -1:] #设置输入ID

        #如果`inputs_embeds`被传递，我们只想在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds} #设置输入嵌入
        else:
            model_inputs = {"input_ids": input_ids} #设置输入ID

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_vid_start_end, tokenizer, device, #初始化视觉tokenizer 
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_config #获取视觉配置
        vision_config.use_vid_start_end = mm_use_vid_start_end #设置使用视频开始结束标记
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True) #添加视频补丁标记
        self.resize_token_embeddings(len(tokenizer)) #调整token嵌入大小

        if mm_use_vid_start_end: #如果使用视频开始结束标记，则设置输入嵌入
            num_new_tokens = tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN])

            if num_new_tokens > 0: #如果新的token数量大于0，则设置输入嵌入
                input_embeddings = self.get_input_embeddings().weight.data #获取输入嵌入
                output_embeddings = self.get_output_embeddings().weight.data #获取输出嵌入

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean( #计算输入嵌入平均值
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean( #计算输出嵌入平均值
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg #设置输入嵌入
                output_embeddings[-num_new_tokens:] = output_embeddings_avg #设置输出嵌入

            if tune_mm_mlp_adapter: #如果tune_mm_mlp_adapter不为空，则设置输入嵌入
                self.get_model().orig_embeds_params = [ #设置原始嵌入参数
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters(): #遍历输入嵌入参数
                    p.requires_grad = True #设置输入嵌入参数可训练
                for p in self.get_output_embeddings().parameters(): #遍历输出嵌入参数
                    p.requires_grad = False #设置输出嵌入参数不可训练

            if pretrain_mm_mlp_adapter: #如果预训练mm_mlp_adapter不为空，则设置输入嵌入
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu') #加载预训练mm_mlp_adapter
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight'] #获取嵌入权重
                assert num_new_tokens == 2 #如果新的token数量不等于2，则抛出异常
                if input_embeddings.shape == embed_tokens_weight.shape: #如果输入嵌入的形状与嵌入权重相同，则设置输入嵌入
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:] #设置输入嵌入
                elif embed_tokens_weight.shape[0] == num_new_tokens: #如果嵌入权重的形状与新的token数量相同，则设置输入嵌入
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight #设置输入嵌入
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_VIDEO_PATCH_TOKEN])[0] #设置视频补丁标记


AutoConfig.register("VideoChatGPT", VideoChatGPTConfig) #注册视频ChatGPT配置
AutoModelForCausalLM.register(VideoChatGPTConfig, VideoChatGPTLlamaForCausalLM) #注册视频ChatGPTLlamaForCausalLM