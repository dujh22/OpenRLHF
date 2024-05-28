# 导入所需的Python模块和函数
import os
import random
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

# 导入相关的深度学习库
import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

# 导入自定义的模型和工具
from openrlhf.models import Actor
from .deepspeed_utils import (
    _z3_params_to_fetch,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
)

# 定义类型别名，用于清晰表达函数返回值和参数类型
ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]

# 定义一个基于DeepSpeed的训练策略类
class DeepspeedStrategy(ABC):
    """
    用于加速器训练的策略。
    """

    # 初始化函数
    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        zero_stage=2,
        bf16=True,
        args=None,
    ) -> None:
        super().__init__()

        # 设置训练参数和状态
        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.bf16 = bf16
        self.seed = seed
        self.max_norm = max_norm
        self.adam_offload = getattr(args, "adam_offload", False)
        self.zpg = getattr(args, "zpg", 1)
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", "fp32")
        self.disable_trace_cache = getattr(args, "disable_trace_cache", False)

        # 用于记录训练步骤
        self.is_rlhf = False
        self.time_steps = defaultdict(int)

    # 设置随机种子以确保可复现性
    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置分布式训练环境
    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)

        # 初始化分布式训练后端
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size

    # 创建优化器
    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    # 定义反向传播过程
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.backward(loss)

    # 优化器步骤，进行模型更新
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.step()

    # 设置数据加载器
    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
    ):
        if sampler is None:
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
            )

        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    # 移除模型包装，用于获取实际的模型
    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    # 准备模型和优化器，用于训练或评估
    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []  # 初始化一个空列表用于存储返回的模型或模型-优化器对
        self.is_rlhf = is_rlhf  # 设置是否使用强化学习人类反馈的标志
        for arg in models_or_model_optim_pairs:  # 遍历传入的所有模型或模型-优化器对
            if isinstance(arg, tuple):  # 如果参数是一个元组
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'  
                # 断言元组的长度为3，即(model, optimizer, scheduler)对
                ret.append(self._ds_init_train_model(*arg))  # 初始化训练模型并添加到返回列表中
            else:  # 如果参数不是元组
                ret.append(self._ds_init_eval_model(arg))  # 初始化评估模型并添加到返回列表中

        return ret[0] if len(ret) == 1 else ret  # 如果返回列表中只有一个元素，则返回该元素，否则返回整个列表

    # 初始化训练模型配置
    def _ds_init_train_model(self, model, optim, scheduler):
        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_train_config(is_actor)

        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_actor else model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": self.args.local_rank},
            dist_init_required=True,
        )
        if is_actor:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler

    # 获取训练模型的DeepSpeed配置
    def get_ds_train_config(self, is_actor):
        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=self.stage,
            bf16=self.bf16,
            max_norm=self.max_norm,
            zpg=self.zpg,
            grad_accum_dtype=self.grad_accum_dtype,
            disable_trace_cache=self.disable_trace_cache,
        )

        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        train_batch_size = self.train_batch_size
        if self.is_rlhf and is_actor and self.args.pretrain_data is not None:
            train_batch_size *= 2
        ds_config["train_batch_size"] = train_batch_size

        return ds_config

    # 初始化评估模型配置
    def _ds_init_eval_model(self, model):
        is_actor = isinstance(model, Actor)  # 判断模型是否为Actor类的实例
        ds_config = self.get_ds_eval_config(offload=getattr(model, "_offload", False))  # 获取评估配置，判断是否需要offload

        # 使用deepspeed初始化引擎
        engine, *_ = deepspeed.initialize(
            model=model.model if is_actor else model,  # 如果是Actor类实例，则使用其内部的model属性，否则直接使用model
            args={"local_rank": self.args.local_rank},  # 传入本地rank参数
            config=ds_config,  # 传入DeepSpeed配置
            dist_init_required=True,  # 需要进行分布式初始化
        )
        
        if is_actor:  # 如果是Actor类实例
            model.model = engine  # 将初始化后的引擎赋值给Actor的model属性
        else:
            model = engine  # 否则直接赋值给model
        return model  # 返回初始化后的模型

    # 获取评估模型的DeepSpeed配置
    def get_ds_eval_config(self, offload=False):
        ds_config = get_eval_ds_config(offload=offload, stage=self.stage if self.stage == 3 else 0, bf16=self.bf16)
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size

        return ds_config

    # 更新模型的指数移动平均权重
    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                        else:
                            # TODO: 使用预筛选以提高效率
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                                data = param.data.to(device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)

    # 加载模型
    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    # 保存模型
    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        # 如果当前进程是rank 0，创建输出目录（如果不存在）
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        # 解包模型，得到需要保存的模型
        model_to_save = self._unwrap_model(model)

        output_state_dict = {}
        # 遍历模型的所有参数
        for k, v in model_to_save.named_parameters():
            # 获取需要提取的参数
            params_to_fetch = _z3_params_to_fetch([v])
            # 使用DeepSpeed的GatheredParameters上下文管理器来聚集参数
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                # 将参数移动到CPU
                vv = v.data.cpu()
                # 如果当前进程是rank 0，将参数添加到输出状态字典中
                if self.is_rank_0():
                    output_state_dict[k] = vv

        # 如果当前进程是rank 0，保存模型的状态字典
        if self.is_rank_0():
            state_dict = model_to_save.state_dict()

            # 遍历模型的所有缓冲区
            for k, v in model_to_save.named_buffers():
                if k not in state_dict:
                    continue
                # 将缓冲区移动到CPU
                vv = v.data.cpu()
                # 将缓冲区添加到输出状态字典中
                output_state_dict[k] = vv

            # 获取状态字典的键和输出状态字典的键
            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())
            # 确保状态字典的键是输出状态字典键的子集
            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            # 如果模型是PeftModel类型，保存预训练模型到输出目录
            if isinstance(model_to_save, PeftModel):
                model_to_save.save_pretrained(output_dir, **kwargs)
                # 如果stage是3，保存适配器模型状态字典
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin"),
                    )
            else:
                # 否则，直接保存预训练模型和状态字典
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)

            # 保存模型配置到输出目录
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            # 保存分词器到输出目录
            tokenizer.save_pretrained(output_dir)

            # 如果训练模型路径存在，将所有以.py结尾的文件复制到输出目录
            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(os.path.join(train_from_model_path, filename), os.path.join(output_dir, filename))


    # 进行数据归约操作，支持mean、max和sum三种操作
    def all_reduce(self, data, op="mean"):
        # 确保操作符是"mean"、"max"或"sum"之一
        assert op in ("mean", "max", "sum")
        
        # 如果data是字典类型
        if isinstance(data, dict):
            ret = {}
            # 遍历字典的每个键值对，递归调用all_reduce
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            # 如果data不是torch.Tensor类型，将其转换为Tensor
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            # 判断data是否在CPU上
            is_cpu_tensor = data.device.type == "cpu"

            # 如果data在CPU上，将其移动到当前GPU设备上
            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            # 如果操作是"mean"，将data除以世界大小（即进程总数）
            if op == "mean":
                data /= self.world_size
            # 调用分布式的all_reduce操作，根据操作符决定使用最大值或和的方式
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            # 如果data最初在CPU上，将其移回CPU
            if is_cpu_tensor:
                data = data.cpu()
            # 如果最初data不是Tensor，返回其值；否则返回Tensor
            return data.item() if not is_tensor else data


    # 收集所有GPU中的数据
    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    # 打印信息，仅在rank 0上执行
    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    # 检查当前是否为rank 0
    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    # 获取当前的rank编号
    def get_rank(self) -> int:
        return dist.get_rank()

    # 保存检查点
    def save_ckpt(self, model, save_dir, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True):
        if self.is_rank_0():
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            MAX_SIZE = max_mem * 1024 * 1024 * 1024

            while True:
                subdirs = [
                    (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                    for d in os.listdir(save_dir)
                    if os.path.isdir(os.path.join(save_dir, d))
                ]
                subdirs.sort(key=lambda x: x[1])

                total_size = 0
                for subdir, _ in subdirs:
                    for dirpath, dirnames, filenames in os.walk(subdir):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)

                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir, _ = subdirs[0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        assert isinstance(model, deepspeed.DeepSpeedEngine)
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state, save_latest=save_latest)

    # 加载检查点
    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        return model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )
