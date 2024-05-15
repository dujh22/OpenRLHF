from typing import Optional  # 导入可选类型

import deepspeed  # 导入 DeepSpeed
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from peft import LoraConfig, get_peft_model  # 导入 LoRA 配置和获取 PEFT 模型的函数
from peft.tuners.lora import LoraLayer  # 导入 LoRA 层
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig  # 导入 AutoConfig, AutoModel 和 BitsAndBytesConfig
from transformers.deepspeed import HfDeepSpeedConfig  # 导入 HuggingFace 的 DeepSpeed 配置
from transformers.dynamic_module_utils import get_class_from_dynamic_module  # 导入动态模块工具
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  # 导入 Mixtral 模型的稀疏 MoE 模块

from openrlhf.utils.logging import init_logger  # 导入初始化日志记录器的函数

logger = init_logger(__name__)  # 初始化日志记录器


# 构建带有序列分类头的变压器模型
# 参考：https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1310
def get_llm_for_sequence_regression(
    model_name_or_path: str,  # 模型名称或路径
    model_type: str,  # 模型类型，可以是 "reward" 或 "critic"
    *,
    bf16=True,  # 是否启用 bfloat16，默认值为 True
    load_in_4bit=False,  # 是否以 4-bit 加载，默认值为 False
    lora_rank=0,  # LoRA 的秩，默认值为 0
    lora_alpha=16,  # LoRA 的 alpha 参数，默认值为 16
    target_modules=None,  # 目标模块，默认值为 None
    lora_dropout=0,  # LoRA 的 dropout 率，默认值为 0
    normalize_reward=False,  # 是否标准化奖励，默认值为 False
    use_flash_attention_2=False,  # 是否使用 Flash Attention 2.0，默认值为 False
    ds_config: dict = None,  # DeepSpeed 配置，默认值为 None
    init_value_head: bool = False,  # 是否初始化 value_head，默认值为 False
    **kwargs,  # 其他参数
) -> nn.Module:
    """获取带有序列分类头的变压器模型。

    参数:
        model_name_or_path (str): 预训练模型的路径。
        model_type (str): 模型类型，可以是 "reward" 或 "critic"。
        bf16 (bool, optional): 是否启用 bfloat16。默认为 True。
        normalize_reward (bool, optional): 是否标准化奖励。默认为 False。
        use_flash_attention_2 (bool, optional): 是否使用 Flash Attention 2.0。默认为 False。
        ds_config (dict, optional): DeepSpeed 配置，用于在启用 ZeRO-3 时自动将模型拆分到多个 GPU 上。默认为 None。

    返回:
        nn.Module: 预训练的变压器模型。
    """
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."  # 确保模型类型是 "critic" 或 "reward"

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)  # 从预训练模型加载配置
    config.normalize_reward = normalize_reward  # 设置是否标准化奖励
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"  # 设置注意力实现方式

    try:
        base_class = AutoModel._model_mapping[type(config)]  # 获取基础模型类
        base_pretrained_class = base_class.__base__  # 获取基础预训练模型类
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)  # 获取奖励模型类
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)  # 获取批判模型类
    except Exception as e:
        print("Failed to load from AutoModel, construct from modelling file.")  # 打印加载失败信息
        module_file, causal_model_name = config.auto_map["AutoModelForCausalLM"].split(".")

        # 特殊情况
        if causal_model_name == "QWenLMHeadModel":
            auto_model_name = "QWenModel"
            pretrained_model_name = "QWenPreTrainedModel"
        elif causal_model_name == "InternLMForCausalLM":
            auto_model_name = "InternLMModel"
            pretrained_model_name = "InternLMPreTrainedModel"
        else:
            if "AutoModel" not in config.auto_map:
                auto_model_name = causal_model_name.split("For")[0] + "Model"
            else:
                auto_model_name = config.auto_map["AutoModel"].split(".")[1]
            pretrained_model_name = causal_model_name.split("For")[0] + "PreTrainedModel"

        logger.info(f"BASE_MODEL_CLASS: {auto_model_name}, PRETRAINED_MODEL_CLASS: {pretrained_model_name}")

        base_pretrained_class = get_class_from_dynamic_module(
            f"{module_file}.{pretrained_model_name}", model_name_or_path
        )  # 获取动态模块中的基础预训练类
        base_class = get_class_from_dynamic_module(f"{module_file}.{auto_model_name}", model_name_or_path)  # 获取动态模块中的基础类
        if model_type == "reward":
            cls_class = _get_reward_model(base_pretrained_class, base_class)  # 获取奖励模型类
        else:
            cls_class = _get_critic_model(base_pretrained_class, base_class)  # 获取批判模型类

    # 注意：dschf 在函数作用域中定义以避免全局影响
    # 参考：https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)  # 初始化 DeepSpeed 配置
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"  # 确保 bf16 为 True
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )  # 初始化 BitsAndBytes 配置
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        **kwargs,
    )  # 从预训练模型加载模型

    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()  # 启用输入需要梯度
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )  # 初始化 LoRA 配置
        model = get_peft_model(model, lora_config)  # 获取 PEFT 模型

        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)  # 将 LoRA 层转换为 bfloat16
                if "norm" in name:
                    module = module.to(torch.float32)  # 将 norm 层转换为 float32
                if "value_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)  # 将 value_head 和 embed_tokens 层转换为 bfloat16

    # MoE - 平衡损失
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")  # 打印 MoE 输出路由日志
        model.config.output_router_logits = True  # 设置输出路由日志为 True

    # 注意：仅用于奖励模型训练，手动初始化 value_head
    # 因为 deepspeed.zero.Init() 不会初始化它们
    # TODO: 找到更好的方法来明确奖励模型训练
    if init_value_head:
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model  # 返回模型


# 获取奖励模型的辅助函数
def _get_reward_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        supports_gradient_checkpointing = True  # 支持梯度检查点

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)  # 添加线性层作为 value_head

            # 标准化奖励
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)  # 注册均值缓冲区
            self.register_buffer("std", torch.ones(1), persistent=False)  # 注册标准差缓冲区

            # 从配置文件中加载均值和标准差
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            # 参考：https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = self.value_head(last_hidden_states).squeeze(-1)

            # 在训练模式下进行左填充
            if self.training:
                reward = values[:, -1]
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

                # 在评估模式下标准化奖励
                if self.normalize_reward:
                    reward = (reward - self.mean) / self.std
            if return_output:
                return reward, outputs
            else:
                return reward

    return LLMForSequenceRegression  # 返回奖励模型类


# 获取批判模型的辅助函数
def _get_critic_model(base_pretrained_model, base_llm_model):
    class LLMForSequenceRegression(base_pretrained_model):
        supports_gradient_checkpointing = True  # 支持梯度检查点

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head = nn.Linear(config.hidden_size, 1, bias=False)  # 添加线性层作为 value_head

            # 标准化奖励
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)  # 注册均值缓冲区
            self.register_buffer("std", torch.ones(1), persistent=False)  # 注册标准差缓冲区

            # 从配置文件中加载均值和标准差
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            action_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            # 参考：https://github.com/OpenLLMAI/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = self.value_head(last_hidden_states).squeeze(-1)[:, :-1]
            num_actions = action_mask.size(1)

            # 标准化奖励
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            if return_output:
                return outputs if num_actions is None else (values[:, -num_actions:], outputs)
            else:
                return values[:, -num_actions:]

    return LLMForSequenceRegression  # 返回批判模型类
