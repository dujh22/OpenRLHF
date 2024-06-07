from typing import Optional, Tuple, Union  # 从typing模块导入常用类型提示

import bitsandbytes as bnb  # 导入bitsandbytes库，取别名为bnb
import deepspeed  # 导入deepspeed库
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.nn.functional as F  # 从PyTorch导入功能函数

def compute_approx_kl(
    log_probs: torch.Tensor,  # 新分布的对数概率张量
    log_probs_base: torch.Tensor,  # 基准分布的对数概率张量
    action_mask: Optional[torch.Tensor] = None,  # 动作掩码
) -> torch.Tensor:  # 返回一个张量
    """
    计算两个分布之间的近似KL散度。
    参考Schulman的博客: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: 新分布的对数概率张量。
        log_probs_base: 基准分布的对数概率张量。
        action_mask: 动作掩码。
    """

    log_ratio = log_probs - log_probs_base  # 计算对数比值
    return log_ratio * action_mask  # 返回经过掩码处理的对数比值

def compute_reward(
    r: Union[torch.Tensor, float],  # 奖励值，可以是张量或浮点数
    kl_coef: float,  # KL系数
    log_probs: torch.Tensor,  # 新分布的对数概率张量
    log_probs_base: torch.Tensor,  # 基准分布的对数概率张量
    action_mask: Optional[torch.Tensor] = None,  # 动作掩码
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回两个张量的元组
    if kl_coef <= 0.0:  # 如果KL系数小于等于0
        kl_coef = 0.0  # 将KL系数设为0

    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)  # 计算近似KL散度
    kl_reward = -kl_coef * kl  # 计算KL奖励

    r = r.clamp(min=-10, max=10)  # 将奖励值限制在-10到10之间

    # 下面的代码相当于：
    #
    # last_reward = torch.zeros_like(kl)
    # for i in range(last_reward.size(0)):
    #     for t in reversed(range(last_reward.size(1))):
    #         if action_mask[i][t] > 0.5:
    #             last_reward[i][t] = r[i]
    #             break
    #
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)  # 计算eos索引
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))  # 计算最后的奖励

    reward = last_reward + kl_reward  # 计算总体奖励
    return reward, kl  # 返回总体奖励和近似KL散度

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # 定义从logits计算对数概率的函数
    log_probs = F.log_softmax(logits, dim=-1)  # 计算对数softmax概率
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))  # 按标签提取对应的对数概率
    return log_probs_labels.squeeze(-1)  # 去掉最后一维并返回对数概率

def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = None) -> torch.Tensor:  # 定义计算掩码平均值的函数
    if dim is not None:  # 如果指定了维度
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)  # 在指定维度上计算掩码平均值
    else:
        return (tensor * mask).sum() / mask.sum()  # 在所有维度上计算掩码平均值

def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:  # 定义计算掩码归一化的函数
    tensor = tensor * mask  # 应用掩码
    mean = masked_mean(tensor, mask, dim=dim)  # 计算掩码平均值
    mean_centered = tensor - mean  # 计算均值中心化张量
    var = masked_mean(mean_centered**2, mask, dim=dim)  # 计算掩码方差
    return mean_centered * var.clamp(min=eps).rsqrt()  # 返回归一化张量