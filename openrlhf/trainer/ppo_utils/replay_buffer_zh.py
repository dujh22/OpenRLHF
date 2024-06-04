import random  # 导入随机模块
from abc import ABC  # 从abc模块导入ABC类，用于定义抽象基类
from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
from typing import List, Optional  # 从typing模块导入常用类型提示

import torch  # 导入PyTorch库
import torch.nn.functional as F  # 从PyTorch导入功能函数

from openrlhf.models.utils import masked_mean  # 从openrlhf.models.utils导入masked_mean函数
from .experience_maker import Experience  # 从本地的experience_maker模块导入Experience类

@dataclass
class BufferItem:  # 定义BufferItem类
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    values: (1)
    returns: (1)
    advatanges: (1)
    attention_mask: (S)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor  # 序列张量
    action_log_probs: torch.Tensor  # 动作对数概率张量
    values: torch.Tensor  # 值张量
    returns: torch.Tensor  # 回报张量
    advantages: torch.Tensor  # 优势张量
    attention_mask: Optional[torch.LongTensor]  # 注意力掩码
    action_mask: Optional[torch.BoolTensor]  # 动作掩码
    info: Optional[dict]  # 额外信息字典

def split_experience_batch(experience: Experience) -> List[BufferItem]:  # 定义拆分经验批次的函数
    batch_size = experience.sequences.size(0)  # 获取批次大小
    batch_kwargs = [{} for _ in range(batch_size)]  # 初始化一个空字典列表
    keys = (
        "sequences",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )  # 关键信息的键名
    for key in keys:  # 遍历每个键
        value = getattr(experience, key)  # 获取经验中的对应值
        vals = torch.unbind(value)  # 将其解包为独立的张量
        assert batch_size == len(vals)  # 断言批次大小等于值的长度
        for i, v in enumerate(vals):  # 遍历值
            batch_kwargs[i][key] = v  # 将值赋给相应的字典项

    for i in range(batch_size):  # 遍历批次大小
        batch_kwargs[i]["info"] = {}  # 初始化每个字典项中的info
    for k, v in experience.info.items():  # 遍历经验中的信息字典
        vals = torch.unbind(v)  # 将信息值解包为独立的张量
        assert batch_size == len(vals)  # 断言批次大小等于值的长度
        for i, vv in enumerate(vals):  # 遍历值
            batch_kwargs[i]["info"][k] = vv.item()  # 将值赋给相应的字典项中的info

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]  # 将字典列表转换为BufferItem对象列表
    return items  # 返回BufferItem对象列表

def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left") -> torch.Tensor:  # 定义零填充序列的函数
    assert side in ("left", "right")  # 断言填充方向为“left”或“right”
    max_len = max(seq.size(0) for seq in sequences)  # 获取序列表中的最大长度
    padded_sequences = []  # 初始化一个空的填充序列列表
    for seq in sequences:  # 遍历每个序列
        pad_len = max_len - seq.size(0)  # 计算需要填充的长度
        padding = (pad_len, 0) if side == "left" else (0, pad_len)  # 根据填充方向设置填充大小
        padded_sequences.append(F.pad(seq, padding))  # 将填充后的序列添加到填充序列列表中
    return torch.stack(padded_sequences, dim=0)  # 将填充序列列表堆叠为张量并返回

def make_experience_batch(items: List[BufferItem]) -> Experience:  # 定义生成经验批次的函数
    kwargs = {}  # 初始化一个空字典
    keys = (
        "sequences",
        "action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )  # 关键信息的键名
    for key in keys:  # 遍历每个键
        vals = [getattr(item, key) for item in items]  # 获取每个BufferItem对象中的对应值
        batch_data = zero_pad_sequences(vals, "left")  # 对值进行零填充
        kwargs[key] = batch_data  # 将填充后的值添加到字典中

    kwargs["info"] = {}  # 初始化信息字典
    for key in items[0].info.keys():  # 遍历第一个BufferItem对象中的信息键
        vals = torch.tensor([item.info[key] for item in items])  # 获取每个BufferItem对象中的对应信息值
        kwargs["info"][key] = vals  # 将信息值添加到字典中
    return Experience(**kwargs)  # 创建并返回Experience对象


def remove_padding_in_sequences(items):  # 定义函数，用于去除序列中的填充
    for item in items:  # 遍历每个BufferItem对象
        seq, act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()  # 计算右侧填充大小
        right_pad = None if right_pad == 0 else -right_pad  # 如果没有右侧填充，设为None，否则取负值

        # 计算左侧填充大小
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],  # 截取去掉填充后的序列
            act_log_prob[:right_pad],
            value[:right_pad],
            ret[:right_pad],
            adv[:right_pad],
            att_mask[left_pad:right_pad],
            act_mask[:right_pad],
        )
    return items  # 返回去除填充后的items

class NaiveReplayBuffer(ABC):  # 定义NaiveReplayBuffer类，继承自抽象基类ABC
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(self, sample_batch_size: int, limit: int = 0, cpu_offload: bool = True) -> None:  # 构造函数
        super().__init__()
        self.sample_batch_size = sample_batch_size  # 设置采样批大小
        self.limit = limit  # 设置存储限制，<= 0表示无限制
        self.cpu_offload = cpu_offload  # 是否卸载到CPU
        self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")  # 获取当前CUDA设备
        self.items: List[BufferItem] = []  # 初始化BufferItem列表

    @torch.no_grad()
    def append(self, experience: Experience) -> None:  # 定义添加经验的方法
        if self.cpu_offload:  # 如果需要卸载到CPU
            experience.to_device(torch.device("cpu"))  # 将经验移到CPU
        items = split_experience_batch(experience)  # 拆分经验批次
        items = remove_padding_in_sequences(items)  # 去除填充
        self.items.extend(items)  # 将items扩展到self.items
        if self.limit > 0:  # 如果有限制
            samples_to_remove = len(self.items) - self.limit  # 计算需要移除的样本数量
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]  # 移除多余的样本

    def clear(self) -> None:  # 清空重放缓冲区
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:  # 定义采样方法
        items = random.sample(self.items, self.sample_batch_size)  # 随机采样
        experience = make_experience_batch(items)  # 生成经验批次
        if self.cpu_offload:  # 如果需要卸载到CPU
            experience.to_device(self.target_device)  # 将经验移到目标设备
        return experience  # 返回经验

    def __len__(self) -> int:  # 定义获取缓冲区长度的方法
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:  # 定义获取缓冲区项的方法
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:  # 定义合并函数
        experience = make_experience_batch(batch)  # 生成经验批次
        return experience  # 返回经验批次

    def normalize(self, attribute: str, strategy) -> None:  # 定义归一化方法
        assert attribute == "advantages"  # 断言只对"advantages"进行归一化
        items = []  # 初始化一个空列表
        action_masks = []  # 初始化一个空的动作掩码列表
        for item in self:  # 遍历缓冲区中的每个项
            items.append(getattr(item, attribute))  # 获取对应属性的值
            action_masks.append(item.action_mask)  # 获取动作掩码

        items_vector = torch.cat(items).float().flatten()  # 链接并压平属性值向量
        action_masks_vector = torch.cat(action_masks).flatten()  # 链接并压平动作掩码向量

        # 针对数据并行
        # mean 计算均值
        sum_and_count = torch.tensor([items_vector.sum(), action_masks_vector.sum()], device=items_vector.device)  # 计算总和和总数
        all_sum, all_count = strategy.all_reduce(sum_and_count, "sum")  # 全部归约计算总和和总数
        mean = all_sum / all_count  # 计算均值
        # std 计算标准差
        std = ((items_vector - mean).pow(2) * action_masks_vector).sum()  # 计算标准差
        all_std = strategy.all_reduce(std, "sum")  # 全部归约计算标准差
        rstd = (all_std / all_count).clamp(min=1e-8).rsqrt()  # 计算标准差的倒数平方根

        for i, item in enumerate(self):  # 遍历缓冲区中的每个项
            setattr(item, attribute, (items[i] - mean) * rstd)  # 对属性值进行归一化