# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import List, Dict, Optional, Union, Any

from PIL import Image
import torch

from data.data_utils import pil_img2rgb
from modeling.bagel.qwen2_navit import NaiveCache


# VLM思考系统提示词 - 用于视觉语言模型的推理过程
VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
The reasoning process is enclosed within </think> </think> tags, i.e. </think> reasoning process here </think> answer here'''

# 图像生成思考系统提示词 - 用于让模型在生成图像前先进行思考规划
GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
The planning process is enclosed within </think> </think> tags, i.e. </think> planning process here </think> image here'''


class InterleaveInferencer:
    
    def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
        
        # 初始化交错推理器所需的各个组件
        self.model = model  # Bagel主模型
        self.vae_model = vae_model  # 变分自编码器模型
        self.tokenizer = tokenizer  # 文本分词器
        self.vae_transform = vae_transform  # VAE输入图像变换
        self.vit_transform = vit_transform  # ViT输入图像变换
        self.new_token_ids = new_token_ids  # 特殊token ID字典
        
    def init_gen_context(self): 
        
        # 初始化生成上下文
        gen_context = {
            'kv_lens': [0],  # 键值对缓存长度
            'ropes': [0],  # 位置编码参数
            'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),  # 过去计算的键值对缓存
        }
        return gen_context

    @torch.no_grad()
    def update_context_text(self, text, gen_context):
        
        # 用于交错数据，当前仅支持1个数据推理
        # 更新文本上下文
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        
        # 准备文本提示
        generation_input, kv_lens, ropes = self.model.prepare_prompts(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            prompts=[text],
            tokenizer=self.tokenizer, 
            new_token_ids=self.new_token_ids,
        )

        # 更新文本缓存
        past_key_values = self.model.forward_cache_update_text(past_key_values, **generation_input)
        
        # 更新上下文信息
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def update_context_image(self, image, gen_context, vae=True, vit=True):
        
        # 用于交错数据，当前仅支持1个数据推理
        # 更新图像上下文
        assert vae or vit  # 至少需要更新VAE或ViT中的一个
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        if vae:
            # 更新VAE部分
            generation_input, kv_lens, ropes = self.model.prepare_vae_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vae_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vae(self.vae_model, past_key_values, **generation_input)
        
        if vit:
            # 更新ViT部分
            generation_input, kv_lens, ropes = self.model.prepare_vit_images(
                curr_kvlens=kv_lens,
                curr_rope=ropes, 
                images=[image],
                transforms=self.vit_transform, 
                new_token_ids=self.new_token_ids,
            )
            past_key_values = self.model.forward_cache_update_vit(past_key_values, **generation_input)

        # 更新上下文信息
        gen_context['kv_lens'] = kv_lens
        gen_context['ropes'] = ropes
        gen_context['past_key_values'] = past_key_values
        
        return gen_context

    @torch.no_grad()
    def gen_image(
        self, 
        image_shape, 
        gen_context, 
        cfg_text_scale=4.0,  # 文本CFG尺度参数
        cfg_img_scale=1.5,   # 图像CFG尺度参数

        cfg_text_precontext=None, 
        cfg_img_precontext=None, 
        cfg_interval=(0.4, 1.0),  # CFG生效区间
        cfg_renorm_min=0.0,       # CFG重归一化最小值
        cfg_renorm_type="global",  # CFG重归一化类型
        
        num_timesteps=50,  # 扩散步骤数
        timestep_shift=3.0,
        enable_taylorseer=False,
    ):
        
        # 生成图像的核心方法
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']
        
        # 准备VAE潜变量
        generation_input = self.model.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=ropes, 
            image_sizes=[image_shape], 
            new_token_ids=self.new_token_ids,
        ) 
        
        # 准备文本CFG
        cfg_text_past_key_values = cfg_text_precontext['past_key_values']
        kv_lens_cfg = cfg_text_precontext['kv_lens']
        ropes_cfg = cfg_text_precontext['ropes']
        generation_input_cfg_text = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # 准备图像CFG
        cfg_img_past_key_values = cfg_img_precontext['past_key_values']
        kv_lens_cfg = cfg_img_precontext['kv_lens']
        ropes_cfg = cfg_img_precontext['ropes']
        generation_input_cfg_img = self.model.prepare_vae_latent_cfg(
            curr_kvlens=kv_lens_cfg,
            curr_rope=ropes_cfg, 
            image_sizes=[image_shape], 
        )

        # 调用模型的generate_image方法进行图像生成
        unpacked_latent = self.model.generate_image(
            past_key_values=past_key_values,
            cfg_text_past_key_values=cfg_text_past_key_values,
            cfg_img_past_key_values=cfg_img_past_key_values,
            num_timesteps=num_timesteps,
            cfg_text_scale=cfg_text_scale,
            cfg_img_scale=cfg_img_scale,
            cfg_interval=cfg_interval,
            cfg_renorm_min=cfg_renorm_min,
            cfg_renorm_type=cfg_renorm_type,
            timestep_shift=timestep_shift,
            **generation_input,
            cfg_text_packed_position_ids=generation_input_cfg_text['cfg_packed_position_ids'],
            cfg_text_packed_query_indexes=generation_input_cfg_text['cfg_packed_query_indexes'],
            cfg_text_key_values_lens=generation_input_cfg_text['cfg_key_values_lens'],
            cfg_text_packed_key_value_indexes=generation_input_cfg_text['cfg_packed_key_value_indexes'],
            cfg_img_packed_position_ids=generation_input_cfg_img['cfg_packed_position_ids'],
            cfg_img_packed_query_indexes=generation_input_cfg_img['cfg_packed_query_indexes'],
            cfg_img_key_values_lens=generation_input_cfg_img['cfg_key_values_lens'],
            cfg_img_packed_key_value_indexes=generation_input_cfg_img['cfg_packed_key_value_indexes'],
            enable_taylorseer=enable_taylorseer,
        )

        # 将生成的潜变量解码为实际图像
        image = self.decode_image(unpacked_latent[0], image_shape)
        return image

        
    def decode_image(self, latent, image_shape):
        
        # 将VAE潜变量解码为图像
        H, W = image_shape
        h, w = H // self.model.latent_downsample, W // self.model.latent_downsample  # 计算潜变量的高度和宽度

        # 重塑潜变量并应用爱因斯坦求和
        latent = latent.reshape(1, h, w, self.model.latent_patch_size, self.model.latent_patch_size, self.model.latent_channel)
        latent = latent.permute(0, 5, 1, 3, 2, 4).contiguous()
        latent = latent.reshape(1, self.model.latent_channel, h * self.model.latent_patch_size, w * self.model.latent_patch_size)
        
        # 调用VAE解码器解码潜变量
        image = self.vae_model.decode(latent)
        
        # 将图像数据转换为PIL格式
        image = (image * 0.5 + 0.5).clamp(0, 1)[0].permute(1, 2, 0) * 255
        image = Image.fromarray((image).to(torch.uint8).cpu().numpy())

        return image

    @torch.no_grad()
    def gen_text(self, gen_context, max_length: int = 500, do_sample: bool = True, temperature: float = 1.0):
        
        # 生成文本
        gen_context = deepcopy(gen_context)
        past_key_values = gen_context['past_key_values']
        kv_lens = gen_context['kv_lens']
        ropes = gen_context['ropes']

        # 准备开始标记
        generation_input = self.model.prepare_start_tokens(kv_lens, ropes, self.new_token_ids)
        
        # 调用模型生成文本
        unpacked_latent = self.model.generate_text(
            past_key_values=past_key_values,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            end_token_id=self.new_token_ids['eos_token_id'],
            **generation_input,
        )
        
        # 解码文本并处理特殊标记
        output = self.tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]
        return output

    @torch.no_grad()
    def interleave_inference(
        self,
        input_lists: List[Union[str, Image.Image]],
        think=False,
        understanding_output=False,

        max_think_token_n=1000,
        do_sample=False,
        text_temperature=0.3,
        cfg_text_scale=3.0,
        cfg_img_scale=1.5,
        cfg_interval=[0.4, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        image_shapes=(1024, 1024),
        enable_taylorseer=False,
    ) -> List[Union[str, Image.Image]]:

        # 交错推理 - 处理多模态输入并生成相应的输出
        output_list = []
        gen_context = self.init_gen_context()
        cfg_text_context = deepcopy(gen_context)
        cfg_img_context = deepcopy(gen_context)

        # 使用自动混合精度计算加速
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # 如果需要思考过程，添加思考系统提示词
            if think:
                if understanding_output:
                    system_prompt = VLM_THINK_SYSTEM_PROMPT 
                else:
                    system_prompt = GEN_THINK_SYSTEM_PROMPT
                gen_context = self.update_context_text(system_prompt, gen_context)
                cfg_img_context = self.update_context_text(system_prompt, cfg_img_context)

            # 处理输入列表中的每个元素
            for input_term in input_lists:
                if isinstance(input_term, str):
                    # 处理文本输入
                    cfg_text_context = deepcopy(gen_context)
                    gen_context = self.update_context_text(input_term, gen_context)
                    cfg_img_context = self.update_context_text(input_term, cfg_img_context)

                elif isinstance(input_term, Image.Image):
                    # 处理图像输入
                    input_term = self.vae_transform.resize_transform(pil_img2rgb(input_term))
                    gen_context = self.update_context_image(input_term, gen_context, vae=not understanding_output)

                    image_shapes = input_term.size[::-1]
                    cfg_text_context = deepcopy(gen_context)

                else:
                    raise ValueError(f"不支持的输入类型: {type(input_term)}")

            # 根据是否需要理解输出决定生成文本还是图像
            if understanding_output:
                gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                output_list.append(gen_text)

            else:
                # 如果需要思考过程，先生成思考文本
                if think:
                    gen_text = self.gen_text(gen_context, do_sample=do_sample, temperature=text_temperature, max_length=max_think_token_n)
                    gen_context = self.update_context_text(gen_text, gen_context)
                    output_list.append(gen_text)

                # 生成图像
                img = self.gen_image(
                    image_shapes, 
                    gen_context, 
                    cfg_text_precontext=cfg_text_context, 
                    cfg_img_precontext=cfg_img_context,

                    cfg_text_scale=cfg_text_scale, 
                    cfg_img_scale=cfg_img_scale, 
                    cfg_interval=cfg_interval, 
                    timestep_shift=timestep_shift, 
                    num_timesteps=num_timesteps,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                    enable_taylorseer=enable_taylorseer,
                )

                output_list.append(img)

        return output_list
    
    def __call__(
        self, 
        image: Optional[Image.Image] = None, 
        text: Optional[str] = None, 
        **kargs
    ) -> Dict[str, Any]:
        
        # 简化接口，处理单张图像和/或单个文本提示
        output_dict = {'image': None, 'text': None}

        # 检查输入是否有效
        if image is None and text is None:
            print('请至少提供一个输入：图像或文本。')
            return output_dict

        # 构建输入列表
        input_list = []
        if image is not None:
            input_list.append(image)
        if text is not None:
            input_list.append(text)

        # 执行交错推理
        output_list = self.interleave_inference(input_list, **kargs)

        # 处理输出结果
        for i in output_list:
            if isinstance(i, Image.Image):
                output_dict['image'] = i
            elif isinstance(i, str):
                output_dict['text'] = i
        return output_dict
