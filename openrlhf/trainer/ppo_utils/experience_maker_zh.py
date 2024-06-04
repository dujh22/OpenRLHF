import logging  # 导入日志模块
import time  # 导入时间模块
from abc import ABC  # 从abc模块导入ABC类，用于定义抽象基类
from copy import deepcopy  # 从copy模块导入深度复制函数
from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
from typing import List, Optional, Tuple, Union  # 从typing模块导入常用类型提示

import ray  # 导入Ray库，用于分布式计算
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from tqdm import tqdm  # 从tqdm导入进度条库

from openrlhf.models.actor import Actor  # 从openrlhf.models.actor导入Actor类
from openrlhf.models.utils import compute_reward, masked_mean  # 从openrlhf.models.utils导入所需的函数
from openrlhf.utils.logging import init_logger  # 从openrlhf.utils.logging导入初始化日志函数

logger = init_logger(__name__)  # 初始化日志记录器

@dataclass
class Experience:  # 定义Experience类，用于存储一批数据
    """Experience is a batch of data.
    These data should have the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)

    "A" is the number of actions.
    
    经验是一批数据。
    这些数据应具有序列长度和操作次数。
    序列采用左填充。

    每个张量的形状：
    序列： (B, S)
    action_log_probs: (B, A)
    值：（B, A）
    返回值: (B, A)
    优势：（B, A）
    attention_mask: (B, S)
    action_mask: (B, A)

    A是行动的数量。
    """

    sequences: torch.Tensor  # 序列张量
    action_log_probs: torch.Tensor  # 动作对数概率张量
    values: torch.Tensor  # 值张量
    returns: torch.Tensor  # 回报张量
    advantages: torch.Tensor  # 优势张量
    attention_mask: Optional[torch.LongTensor]  # 注意力掩码
    action_mask: Optional[torch.BoolTensor]  # 动作掩码
    info: Optional[dict]  # 额外信息字典

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:  # 定义将张量移动到设备的函数
        self.sequences = self.sequences.to(device)  # 将序列张量移动到设备
        self.action_log_probs = self.action_log_probs.to(device)  # 将动作对数概率张量移动到设备
        self.values = self.values.to(device)  # 将值张量移动到设备
        self.returns = self.returns.to(device)  # 将回报张量移动到设备
        self.advantages = self.advantages.to(device)  # 将优势张量移动到设备
        if self.attention_mask is not None:  # 如果注意力掩码不为空
            self.attention_mask = self.attention_mask.to(device)  # 将注意力掩码移动到设备
        if self.action_mask is not None:  # 如果动作掩码不为空
            self.action_mask = self.action_mask.to(device)  # 将动作掩码移动到设备

    def pin_memory(self):  # 定义将张量固定到内存的函数
        self.sequences = self.sequences.pin_memory()  # 将序列张量固定到内存
        self.action_log_probs = self.action_log_probs.pin_memory()  # 将动作对数概率张量固定到内存
        self.values = self.values.pin_memory()  # 将值张量固定到内存
        self.returns = self.returns.pin_memory()  # 将回报张量固定到内存
        self.advantages = self.advantages.pin_memory()  # 将优势张量固定到内存
        if self.attention_mask is not None:  # 如果注意力掩码不为空
            self.attention_mask = self.attention_mask.pin_memory()  # 将注意力掩码固定到内存
        if self.action_mask is not None:  # 如果动作掩码不为空
            self.action_mask = self.action_mask.pin_memory()  # 将动作掩码固定到内存
        return self  # 返回自身

class NaiveExperienceMaker(ABC):  # 定义NaiveExperienceMaker类，继承自抽象基类ABC
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,  # actor模型
        critic: nn.Module,  # critic模型
        reward_model: nn.Module,  # 奖励模型
        initial_model: Actor,  # 初始模型
        tokenizer,  # 标记器
        prompt_max_len: int,  # 提示的最大长度
        kl_controller,  # KL控制器
        strategy=None,  # 策略
        reward_fn=None,  # 奖励函数
    ) -> None:  # 返回类型为空
        super().__init__()
        self.actor = actor  # 设置actor模型
        self.critic = critic  # 设置critic模型
        self.reward_model = reward_model  # 设置奖励模型
        self.initial_model = initial_model  # 设置初始模型
        self.tokenizer = tokenizer  # 设置标记器
        self.prompt_max_len = prompt_max_len  # 设置提示的最大长度
        self.kl_ctl = kl_controller  # 设置KL控制器
        self.strategy = strategy  # 设置策略
        self.reward_fn = reward_fn  # 设置奖励函数

    # tokenizer 函数，用于标记化输入文本
    def tokenize_fn(self, texts, max_length, device):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",  # 返回PyTorch张量
            max_length=max_length,  # 设置最大长度
            padding=True,  # 启用填充
            truncation=True,  # 启用截断
        )
        return {k: v.to(device) for k, v in batch.items()}  # 将所有张量移动到指定设备

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:  # 定义生成经验函数
        self.actor.eval()  # 设置actor模型为评估模式
        self.critic.eval()  # 设置critic模型为评估模式
        self.initial_model.eval()  # 设置初始模型为评估模式
        self.reward_model.eval()  # 设置奖励模型为评估模式

        # 生成序列
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")  # 标记化输入
        sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)  # 生成序列、注意力掩码和动作掩码
        num_actions = action_mask.size(1)  # 获取动作数量

        # 计算动作对数概率
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # 计算初始动作对数概率
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # 计算值
        value = self.critic(sequences, action_mask, attention_mask)

        # 计算奖励
        r = self.reward_model(sequences, attention_mask)

        reward, kl = compute_reward(
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "reward": r,
            "return": reward.sum(dim=-1),
            "response_length": action_mask.float().sum(dim=-1),
            "total_length": attention_mask.float().sum(dim=-1),
        }
        # 重置模型状态
        self.actor.train()
        self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,  # 值张量
        rewards: torch.Tensor,  # 奖励张量
        action_mask: torch.Tensor,  # 动作掩码
        gamma: float,  # 折扣因子
        lambd: float,  # GAE参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 输出类型为张量的元组
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)

        根据奖励和价值计算优势和回报的函数。
        计算方法与 PPO 原文相同： https://arxiv.org/abs/1707.06347
        请注意，奖励可能包括 KL 发散损失项。

        优势如下
        Adv1 = R1 + γ * λ * R2 + γ^2 * λ^2 * R3 + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        返回值如下
        Ret1 = R1 + γ * λ * R2 + γ^2 * λ^2 * R3 + ...
                   + γ * (1 -λ) V2 + γ^2 * λ * (1 -λ) V3 + ...

        输入
        - 值： 张量形状（批量大小、响应大小）
        - 回报： 形状张量（批量大小、响应大小）

        输出：
        - 优势： 形状张量（批量大小、响应大小）
        - 回报： 形状张量（batch_size、response_size）
        """
        lastgaelam = 0  # 上一个GAE lambda
        advantages_reversed = []  # 优势的反向列表
        response_length = rewards.size(1)  # 响应的长度

        # 屏蔽无效的响应
        values = action_mask * values
        rewards = action_mask * rewards

        for t in reversed(range(response_length)):  # 反向遍历响应长度
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0  # 获取下一个值
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]  # 计算delta
            lastgaelam = delta + gamma * lambd * lastgaelam  # 计算GAE lambda
            advantages_reversed.append(lastgaelam)  # 将GAE lambda添加到反向列表中
        advantages = torch.stack(advantages_reversed[::-1], dim=1)  # 将优势反转回正向
        returns = advantages + values  # 计算回报
        return advantages.detach(), returns  # 返回优势和回报


class RemoteExperienceMaker(NaiveExperienceMaker):  # 定义RemoteExperienceMaker类，继承自NaiveExperienceMaker
    def __init__(self, *args, vllm_engines: List = None, **kwargs):  # 构造函数，接收可变参数和关键字参数
        super().__init__(*args, **kwargs)  # 调用父类的构造函数
        self.vllm_engines = vllm_engines  # 设置vLLM引擎

    @torch.no_grad()
    def make_experience(self, prompts: Union[str, List[str]], **generate_kwargs) -> Experience:  # 定义生成经验的函数
        self.actor.eval()  # 设置actor模型为评估模式
        device = torch.cuda.current_device()  # 获取当前CUDA设备

        # 生成序列
        start = time.time()  # 记录开始时间
        sequences, attention_mask, action_mask = (
            self._generate_local(prompts, **generate_kwargs)  # 本地生成序列
            if self.vllm_engines is None  # 如果没有vLLM引擎
            else self._generate_vllm(prompts, **generate_kwargs)  # 使用vLLM引擎生成序列
        )
        generate_time = time.time() - start  # 计算生成序列所用时间

        num_actions = action_mask.size(1)  # 获取动作数量
        sequences_cpu, attention_mask_cpu, action_mask_cpu = (  # 将序列、注意力掩码和动作掩码移到CPU
            sequences.to("cpu"),
            attention_mask.to("cpu"),
            action_mask.to("cpu"),
        )

        # 初始动作对数概率
        base_action_log_probs_ref = self.initial_model.forward.remote(sequences_cpu, num_actions, attention_mask_cpu)

        # 计算值
        value_ref = self.critic.forward.remote(sequences_cpu, action_mask_cpu, attention_mask_cpu)

        # 避免CUDA内存不足，当模型联合定位时
        if self.strategy.args.colocate_critic_reward:
            ray.get([value_ref])  # 获取值引用

        if self.strategy.args.colocate_actor_ref:
            ray.get([base_action_log_probs_ref])  # 获取初始动作对数概率引用

        # 计算奖励
        r_refs = []
        for rm in self.reward_model:
            r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu))  # 远程调用奖励模型

        # 计算动作对数概率
        start = time.time()  # 记录开始时间
        action_log_probs = self.actor(sequences, num_actions, attention_mask)  # 计算动作对数概率
        actor_time = time.time() - start  # 计算所用时间

        # 等待初始模型、评论模型和奖励模型完成
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)  # 获取所有引用值
        wait_time = time.time() - start  # 计算等待时间

        base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]  # 解包引用值
        base_action_log_probs, value = base_action_log_probs.to(device), value.to(device)  # 移动到设备
        rewards = [r.to(device) for r in rewards]  # 移动所有奖励到设备
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]  # 计算奖励函数

        reward, kl = compute_reward(  # 计算奖励和KL散度
            r,
            self.kl_ctl.value,
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
        )
        advantage, returns = self.get_advantages_and_returns(  # 计算优势和回报
            value,
            reward,
            action_mask,
            generate_kwargs["gamma"],
            generate_kwargs["lambd"],
        )

        info = {  # 收集信息
            "kl": masked_mean(kl, action_mask, dim=-1),  # 计算KL散度的掩码平均值
            "reward": r,  # 奖励
            "return": reward.sum(dim=-1),  # 回报之和
            "response_length": action_mask.float().sum(dim=-1),  # 响应长度
            "total_length": attention_mask.float().sum(dim=-1),  # 总长度
        }

        if self.strategy.args.perf:  # 如果启用了性能分析
            batch_size = 1 if isinstance(prompts, str) else len(prompts)  # 获取批处理大小
            info["generate_time"] = torch.full((batch_size,), generate_time, device=device)  # 记录生成时间
            info["actor_time"] = torch.full((batch_size,), actor_time, device=device)  # 记录actor时间
            info["wait_time"] = torch.full((batch_size,), wait_time, device=device)  # 记录等待时间

        experience = Experience(  # 创建经验对象
            sequences,
            action_log_probs,
            value,
            returns,
            advantage,
            attention_mask,
            action_mask,
            info,
        )

        # 发送经验到评论模型
        experience_cpu = deepcopy(experience)
        experience_cpu.to_device("cpu")  # 移动到CPU
        self._ref = self.critic.append.remote(experience_cpu)  # 远程调用append方法

        self.actor.train()  # 重置模型状态为训练模式
        return experience  # 返回经验对象

    def _generate_local(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 定义本地生成函数
        inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")  # 标记化输入
        return self.actor.generate(**inputs, **kwargs)  # 生成序列

    def _generate_vllm(self, prompts: List[str], **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # 定义vLLM生成函数
        from vllm import SamplingParams  # 仅在此函数内导入vLLM的SamplingParams

        # 轮询负载均衡
        rank = torch.distributed.get_rank()  # 获取当前进程的rank
        llm = self.vllm_engines[rank % len(self.vllm_engines)]  # 选择当前rank对应的vLLM引擎

        sampling_params = SamplingParams(  # 创建采样参数
            temperature=kwargs.get("temperature", 1.0),  # 温度参数
            top_p=kwargs.get("top_p", 1.0),  # top-p截断参数
            top_k=kwargs.get("top_k", -1),  # top-k截断参数
            max_tokens=kwargs.get("max_new_tokens", 16),  # 最大生成token数
        )

        # TODO: 由于vLLM的标记化器不支持输入截断，暂时无法传递`max_length`参数
        input_ids = self.tokenize_fn(prompts, self.prompt_max_len, device="cpu")["input_ids"]
        assert self.tokenizer.padding_side == "left", f"tokenizer padding_size should be left"
        pad_indices = (input_ids != self.tokenizer.pad_token_id).to(dtype=torch.int).argmax(dim=-1)  # 获取填充索引
        prompt_token_ids = []  # 创建提示token ID列表
        for i, pad_index in enumerate(pad_indices.numpy()):  # 遍历填充索引
            prompt_token_ids.append(input_ids[i][pad_index:].tolist())  # 添加未填充部分的token ID
        outputs = ray.get(llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids))  # 远程调用LLM生成

        # 将所有输出连接成如下格式：
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0  # 初始化最大输入长度和最大输出长度
        for output in outputs:  # 遍历输出
            output_token_ids = output.outputs[0].token_ids  # 获取输出token ID
            if output_token_ids[0] == self.tokenizer.eos_token_id:  # 如果第一个token为EOS
                logger.warning(f"Only EOS output for prompt: {output.prompt}")  # 记录警告
                output.outputs[0].token_ids = [self.tokenizer.unk_token_id, self.tokenizer.eos_token_id]  # 用未知token和EOS替换
            max_input_len = max(max_input_len, len(output.prompt_token_ids))  # 更新最大输入长度
            max_output_len = max(max_output_len, len(output_token_ids))  # 更新最大输出长度

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id  # 获取填充和EOS token ID
        sequences = []  # 初始化序列列表
        for output in outputs:  # 遍历输出
            input_len = len(output.prompt_token_ids)  # 获取输入长度
            input_ids = [pad_token_id] * (max_input_len - input_len) + output.prompt_token_ids  # 左填充输入

            output_len = len(output.outputs[0].token_ids)  # 获取输出长度
            output_ids = output.outputs[0].token_ids + [pad_token_id] * (max_output_len - output_len)  # 右填充输出
            if output_ids[output_len - 1] != eos_token_id:  # 如果输出最后一个token不是EOS
                assert output_len == max_output_len  # 确保输出长度等于最大输出长度
                output_ids[-1] = eos_token_id  # 将最后一个token替换为EOS

            sequences.append(input_ids + output_ids)  # 连接输入和输出，并添加到序列列表中

        sequences = torch.tensor(sequences)  # 转换为张量
        sequences, attention_mask, action_mask = self.actor.process_sequences(  # 处理序列
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences.to("cuda"), attention_mask.to("cuda"), action_mask.to("cuda")  # 返回处理后的序列和掩码

    def flush(self):
        "Ensure all experience has been sent to critic"
        ray.get(self._ref)  # 获取引用，确保所有经验已发送到评论模型
        self._ref = None  # 重置引用