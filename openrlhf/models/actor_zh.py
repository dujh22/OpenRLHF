from typing import Optional, Tuple, Union  # 导入类型提示模块

import deepspeed  # 导入DeepSpeed库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数接口
from peft import LoraConfig, TaskType, get_peft_model  # 导入PEFT库中的相关模块
from peft.tuners.lora import LoraLayer  # 导入LoRA层
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel  # 导入transformers库中的相关模块
from transformers.deepspeed import HfDeepSpeedConfig  # 导入DeepSpeed配置模块

from .utils import log_probs_from_logits  # 从本地模块中导入log_probs_from_logits函数

class Actor(nn.Module):
    """
    Actor模型基类。

    参数:
        model (nn.Module): Actor模型。
        lora_rank (int): LoRA的秩。
        lora_train_bias (str): LoRA的偏置训练模式。
    """

    def __init__(
        self,
        pretrain_or_model,  # 预训练模型的路径或直接传入模型
        use_flash_attention_2=False,  # 是否使用flash_attention_2
        bf16=True,  # 是否使用bf16数据类型
        load_in_4bit=False,  # 是否以4bit精度加载模型
        lora_rank=0,  # LoRA的秩
        lora_alpha=16,  # LoRA的alpha参数
        lora_dropout=0,  # LoRA的dropout率
        target_modules=None,  # 目标模块
        ds_config=None,  # DeepSpeed配置
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):  # 如果传入的是字符串，则认为是预训练模型的路径
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"  # 设置注意力机制的实现方式

            # 注：为了避免全局效果，dschf在函数作用域内定义
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:  # 如果DeepSpeed配置不为空且零优化级别为3
                dschf = HfDeepSpeedConfig(ds_config)  # 初始化HfDeepSpeedConfig
            else:
                dschf = None

            if load_in_4bit:  # 如果以4bit精度加载模型
                assert bf16, "我们只支持 bnb_4bit_compute_dtype = bf16"  # 确保使用bf16数据类型
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,  # 信任远程代码
                attn_implementation=attn_implementation,  # 使用指定的注意力机制实现方式
                quantization_config=nf4_config,  # 量化配置
                torch_dtype=torch.bfloat16 if bf16 else "auto",  # 使用指定的数据类型
            )

            # LoRA
            if lora_rank > 0:  # 如果LoRA秩大于0
                # 启用输入梯度需求
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,  # 任务类型为因果语言模型
                    r=lora_rank,  # LoRA的秩
                    lora_alpha=lora_alpha,  # LoRA的alpha参数
                    target_modules=target_modules,  # 目标模块
                    lora_dropout=lora_dropout,  # LoRA的dropout率
                    bias="none",  # 无偏置
                )
                self.model = get_peft_model(self.model, lora_config)  # 获取PEFT模型

                if load_in_4bit:  # 如果以4bit精度加载模型
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):  # 如果模块是LoraLayer
                            module = module.to(torch.bfloat16)
                        if "norm" in name:  # 如果模块名称包含norm
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:  # 如果模块名称包含lm_head或embed_tokens
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE - 平衡损失
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:  # 如果模型配置包含output_router_logits
                print("[MoE] 设置 output_router_logits 为 True")
                self.model.config.output_router_logits = True  # 设置output_router_logits为True

        else:
            self.model = pretrain_or_model  # 直接使用传入的模型
    
    @torch.no_grad()  # 表示此方法在不需要计算梯度时运行
    def generate(self, input_ids: torch.Tensor, **kwargs) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],  # 返回类型可能是一个包含两个LongTensor的元组
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],  # 也可能是包含两个LongTensor和一个BoolTensor的元组
    ]:
        generate_args = {  # 设置生成参数
            "input_ids": input_ids,  # 输入的ID
            "top_k": kwargs.get("top_k", None),  # 从top_k个最高概率中选择下一个token
            "top_p": kwargs.get("top_p", None),  # 从累积概率大于top_p的token池中选择下一个token
            "do_sample": kwargs.get("do_sample", True),  # 是否进行采样
            "early_stopping": True,  # 是否提前停止
            "temperature": kwargs.get("temperature", 1),  # 控制生成的多样性
            "use_cache": True,  # 是否使用缓存
            "num_beams": kwargs.get("num_beams", 1),  # 使用beam search的束数
            "attention_mask": kwargs.get("attention_mask"),  # 注意力掩码
            "eos_token_id": kwargs.get("eos_token_id"),  # 句子结束的token ID
            "pad_token_id": kwargs.get("pad_token_id"),  # 填充的token ID
            "min_new_tokens": kwargs.get("min_new_tokens ", 1),  # 生成的新token的最小数量
        }

        if kwargs.get("max_new_tokens", None):  # 如果提供了max_new_tokens参数
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):  # 如果提供了max_length参数
            generate_args["max_length"] = kwargs.get("max_length")

        # 调用generate方法生成序列
        sequences = self.model.generate(**generate_args)

        # 准备mask张量
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        return self.process_sequences(sequences, input_ids.size(1), eos_token_id, pad_token_id)  # 返回处理后的序列

    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        # 构建注意力掩码，标记序列中不等于结束标记(eos_token_id)和填充标记(pad_token_id)的位置
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)  
        
        # 获取序列的长度（假设每个序列长度相同）
        seq_length = attention_mask.size(1)  
        
        # 以下代码等效于:
        #
        # for i in range(attention_mask.size(0)):  # 遍历批次中的每个序列
        #     for t in reversed(range(seq_length)):  # 反向遍历序列中的每个时间步
        #         if attention_mask[i][t] > 0.5:  # 如果当前时间步不为填充或结束标记
        #             attention_mask[i][min(t + 1, seq_length - 1)] = True  # 标记下一个时间步位置
        #             sequences[i][min(t + 1, seq_length - 1)] = eos_token_id  # 在这个位置插入结束标记
        #             break  # 退出内层循环，继续处理下一个序列
        #
        
        # 计算每个序列中第一个出现结束标记或填充标记后的索引
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)  
        
        # 根据计算的索引，更新注意力掩码，将这些位置标记为 1
        attention_mask.scatter_(dim=1, index=eos_indices, value=1)  
        
        # 根据计算的索引，在这些位置插入结束标记
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)  

        # 在强化学习中, state_i（当前token）+ action_i（下一个token） -> state_i+1（下一个token）
        state_seq = sequences[:, input_len - 1 : -1]  # 提取从 input_len 开始到最后的子序列（过滤掉最后一个标记）
        
        # 我们只计算 state_i 既不是结束标记也不是填充标记的位置的损失
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)  
        
        # 返回处理后的序列，注意力掩码和行动掩码
        return sequences, attention_mask, action_mask  

    def forward(
        self,
        sequences: torch.LongTensor,  # 输入序列（长整数类型的张量）
        num_actions: int = None,  # 动作数量，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为 None
        return_output=False,  # 是否返回完整的模型输出
    ) -> torch.Tensor:
        """返回动作的 log 概率"""
        
        # 生成 position_ids，作为位置编码，用于表示每个 token 的位置
        position_ids = attention_mask.long().cumsum(-1) - 1

        # 将注意力掩码中为0的位置对应的 position_ids 设为1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # 使用模型进行前向传播，传入生成的位置编码
        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

        # 从模型输出的 logits 中计算 log 概率，范围为所有但不包括最后一个 token（即：output["logits"][:, :-1, :]）
        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        # 如果 return_output 为 True，返回（log_probs, 以及 output），否则只返回 log_probs
        if return_output:
            return output if num_actions is None else (log_probs[:, -num_actions:], output)
        else:
            return log_probs[:, -num_actions:]  # 仅返回最后 num_actions 个 log 概率

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        # 启用模型的梯度检查点功能，以减少显存使用
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        # 禁用模型的梯度检查点功能
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        # 打印模型中可训练的参数
        self.model.print_trainable_parameters()
