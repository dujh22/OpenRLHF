import math  # 导入数学模块
import os.path  # 导入os.path模块，用于文件和路径操作
from abc import ABC  # 从abc模块中导入ABC类，用于定义抽象基类
from typing import Any, Callable, Dict, List, Optional, Union  # 从typing模块导入常用类型提示

import ray  # 导入Ray库，用于分布式计算
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch import Tensor  # 从PyTorch中导入Tensor类
from torch.optim import Optimizer  # 从PyTorch中导入Optimizer优化器类
from torch.utils.data import DataLoader, DistributedSampler  # 从PyTorch中导入DataLoader和DistributedSampler
from tqdm import tqdm  # 从tqdm导入进度条库

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, SwitchBalancingLoss, ValueLoss  # 从openrlhf.models导入所需的模型和损失函数
from openrlhf.models.utils import masked_mean  # 从openrlhf.models.utils导入masked_mean函数

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer  # 从本地的ppo_utils模块导入所需的工具

class PPOTrainer(ABC):  # 定义PPOTrainer类，继承自ABC（抽象基类）
    """
        PPO算法的训练器。

    Args:
        strategy (Strategy): 使用的训练策略
        actor (Actor): PPO算法中的actor模型
        critic (nn.Module): PPO算法中的critic模型
        reward_model (nn.Module): RLHF算法中的奖励模型，用于生成句子的奖励
        initial_model (Actor): RLHF算法中的初始模型，用于生成参考logits以限制actor的更新
        actor_optim (Optimizer): actor模型使用的优化器
        critic_optim (Optimizer): critic模型使用的优化器
        kl_coef (float, defaults to 0.1): KL散度损失的系数
        train_batch_size (int, defaults to 8): 训练时使用的批大小
        buffer_limit (int, defaults to 0): 重放缓冲区的最大大小限制
        buffer_cpu_offload (bool, defaults to True): 是否将重放缓冲区卸载到CPU
        eps_clip (float, defaults to 0.2): 策略损失的剪辑系数
        value_clip (float, defaults to 0.4): 值损失的剪辑系数
        experience_batch_size (int, defaults to 8): 经验生成时使用的批大小
        max_epochs (int, defaults to 1): 训练过程中的最大轮数
        tokenier (Callable, optional): 输入标记化时使用的标记器
        sample_replay_buffer (bool, defaults to False): 是否从重放缓冲区中采样
        dataloader_pin_memory (bool, defaults to True): 数据加载器是否固定内存
        callbacks (List[Callback], defaults to []): 训练过程中的回调函数
        generate_kwargs (dict, optional): 模型生成时使用的其他参数
    """

    def __init__(
        self,
        strategy,  # 训练策略
        actor: Actor,  # actor模型
        critic: nn.Module,  # critic模型
        reward_model: nn.Module,  # 奖励模型
        initial_model: Actor,  # 初始模型
        ema_model: Actor,  # 指数移动平均模型
        actor_optim: Optimizer,  # actor模型的优化器
        critic_optim: Optimizer,  # critic模型的优化器
        actor_scheduler,  # actor优化器的调度器
        critic_scheduler,  # critic优化器的调度器
        ema_beta: float = 0.992,  # 指数移动平均的beta值
        init_kl_coef: float = 0.001,  # 初始KL系数
        kl_target: float = None,  # KL目标值
        kl_horizon: int = 10000,  # KL控制的水平线
        ptx_coef: float = 0,  # PTX损失的系数
        micro_train_batch_size: int = 8,  # 微训练批大小
        buffer_limit: int = 0,  # 重放缓冲区大小限制
        buffer_cpu_offload: bool = True,  # 是否卸载重放缓冲区到CPU
        eps_clip: float = 0.2,  # 策略损失的剪辑系数
        value_clip: float = 0.2,  # 值损失的剪辑系数
        micro_rollout_batch_size: int = 8,  # 微回滚批大小
        gradient_checkpointing: bool = False,  # 是否开启梯度检查点
        max_epochs: int = 1,  # 最大训练轮数
        max_norm: float = 1.0,  # 梯度裁剪的最大范数
        tokenizer: Optional[Callable[[Any], dict]] = None,  # 标记器
        prompt_max_len: int = 128,  # 提示的最大长度
        dataloader_pin_memory: bool = True,  # 数据加载器是否固定内存
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,  # 奖励函数
        **generate_kwargs,  # 额外的生成参数
    ) -> None:  # 返回类型为None
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"  # 断言，若使用多个奖励模型，必须指定reward_fn

        super().__init__()  # 调用父类的初始化方法
        self.strategy = strategy  # 设置训练策略
        self.args = strategy.args  # 获取策略的参数
        self.micro_rollout_batch_size = micro_rollout_batch_size  # 设置微回滚批大小
        self.max_epochs = max_epochs  # 设置最大训练轮数
        self.tokenizer = tokenizer  # 设置标记器
        self.generate_kwargs = generate_kwargs  # 设置生成参数
        self.dataloader_pin_memory = dataloader_pin_memory  # 设置数据加载器是否固定内存
        self.max_norm = max_norm  # 设置梯度裁剪的最大范数
        self.ptx_coef = ptx_coef  # 设置PTX损失的系数
        self.micro_train_batch_size = micro_train_batch_size  # 设置微训练批大小
        self.kl_target = kl_target  # 设置KL目标值
        self.prompt_max_len = prompt_max_len  # 设置提示的最大长度
        self.ema_beta = ema_beta  # 设置指数移动平均的beta值
        self.gradient_checkpointing = gradient_checkpointing  # 设置是否开启梯度检查点
        self.reward_fn = reward_fn  # 设置奖励函数

        self.actor = actor  # 设置actor模型
        self.critic = critic  # 设置critic模型
        self.reward_model = reward_model  # 设置奖励模型
        self.initial_model = initial_model  # 设置初始模型
        self.ema_model = ema_model  # 设置指数移动平均模型
        self.actor_optim = actor_optim  # 设置actor模型的优化器
        self.critic_optim = critic_optim  # 设置critic模型的优化器
        self.actor_scheduler = actor_scheduler  # 设置actor优化器的调度器
        self.critic_scheduler = critic_scheduler  # 设置critic优化器的调度器

        self.actor_loss_fn = PolicyLoss(eps_clip)  # 初始化策略损失函数
        self.critic_loss_fn = ValueLoss(value_clip)  # 初始化值损失函数
        self.ptx_loss_fn = GPTLMLoss()  # 初始化GPT语言模型损失函数

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8  # 判断是否使用辅助损失

        if self.kl_target:  # 如果设置了KL目标值
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)  # 使用自适应KL控制器
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)  # 使用固定的KL控制器

        self.experience_maker = NaiveExperienceMaker(
            actor, critic, reward_model, initial_model, tokenizer, prompt_max_len, self.kl_ctl, strategy, reward_fn)  # 初始化经验生成器
        self.replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)  # 初始化重放缓冲区

        self._wandb = None  # 初始化WandB对象为None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():  # 如果使用WandB并且是主进程
            import wandb  # 导入WandB模块

            self._wandb = wandb  # 设置WandB对象
            if not wandb.api.api_key:  # 如果WandB没有API密钥
                wandb.login(key=strategy.args.use_wandb)  # 登录WandB
            wandb.init(
                entity=strategy.args.wandb_org,  # 设置WandB实体
                project=strategy.args.wandb_project,  # 设置WandB项目
                group=strategy.args.wandb_group,  # 设置WandB组
                name=strategy.args.wandb_run_name,  # 设置WandB运行名称
                config=strategy.args.__dict__,  # 设置WandB配置
                reinit=True,  # 重新初始化
            )

            wandb.define_metric("train/global_step")  # 定义训练全局步数指标
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)  # 定义训练相关指标
            wandb.define_metric("eval/epoch")  # 定义评估轮数指标
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)  # 定义评估相关指标

    def fit(
        self,
        prompts_dataloader,  # 提示数据加载器
        pretrain_dataloader,  # 预训练数据加载器
        args,  # 参数
    ) -> None:  # 返回类型为空
        self.prompts_dataloader = prompts_dataloader  # 设置提示数据加载器
        self.pretrain_dataloader = pretrain_dataloader  # 设置预训练数据加载器

        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)  # 计算更新时间步
        global_step = 1  # 初始化全局步数

        # 获取评估和保存步骤
        if args.eval_steps == -1:
            args.eval_steps = prompts_dataloader.__len__() // update_timesteps  # 每个epoch评估一次
        if args.save_steps == -1:
            args.save_steps = float("inf")  # 不保存检查点

        for episode in range(args.num_episodes):  # 遍历每一集
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):  # 如果提示数据加载器是使用的分布式采样器
                self.prompts_dataloader.sampler.set_epoch(episode)  # 设置当前epoch
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),  # 进度条长度为提示数据加载器的长度
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",  # 设置进度条描述
                disable=not self.strategy.is_rank_0(),  # 如果不是主进程则禁用进度条
            )

            for rand_prompts in self.prompts_dataloader:  # 遍历提示数据加载器中的每个样本
                experience = self.experience_maker.make_experience(rand_prompts, **self.generate_kwargs)  # 生成经验
                # 在每个更新步骤打印提示/答案
                if global_step % update_timesteps == 0:
                    output = self.tokenizer.batch_decode(experience.sequences, skip_special_tokens=True)  # 解码输出
                    self.strategy.print(output[0])  # 打印输出
                self.replay_buffer.append(experience)  # 将经验添加到重放缓冲区

                if global_step % update_timesteps == 0:
                    torch.cuda.empty_cache()  # 清空CUDA缓存
                    self.replay_buffer.normalize("advantages", self.strategy)  # 归一化收益
                    status = self.ppo_train()  # 执行PPO训练
                    self.replay_buffer.clear()  # 清空重放缓冲区
                    torch.cuda.empty_cache()  # 再次清空CUDA缓存
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size)  # 更新KL控制器
                    # 记录日志/保存检查点
                    self.save_logs_and_checkpoints(args, global_step // update_timesteps, pbar, status)

                pbar.update()  # 更新进度条
                global_step = global_step + 1  # 增加全局步数

    def ppo_train(self):  # 定义PPO算法的训练函数
        # 重放缓冲区可能一开始是空的，所以我们应该在每次训练中重建它
        dataloader = DataLoader(
            self.replay_buffer,  # 数据加载器使用重放缓冲区
            batch_size=self.replay_buffer.sample_batch_size,  # 设置批大小
            shuffle=True,  # 设置为随机打乱
            drop_last=True,  # 设置丢弃最后的小批量
            pin_memory=self.dataloader_pin_memory,  # 设置是否固定内存
            collate_fn=self.replay_buffer.collate_fn,  # 设置数据集的整合函数
        )
        device = torch.cuda.current_device()  # 获取当前的CUDA设备

        status_list = []  # 用于存储状态的列表
        status_mean = {}  # 用于存储平均状态的字典
        for epoch in range(self.max_epochs):  # 遍历每个训练时代
            pbar = tqdm(
                dataloader,  # 进度条加载数据加载器
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",  # 设置进度条描述
                disable=not self.strategy.is_rank_0(),  # 如果不是主进程则禁用进度条
            )
            for experience in pbar:  # 遍历数据加载器中的每个经验
                experience.to_device(device)  # 将经验移动到设备上
                status = self.training_step(experience)  # 进行一次训练步骤

                # 针对数据并行
                # 加权平均KL值
                status["kl"] *= status["response_length"]  # 按响应长度加权KL值
                status = self.strategy.all_reduce(status)  # 汇总状态
                status["kl"] /= status["response_length"]  # 计算加权平均的KL值

                status_list.append(status)  # 将状态添加到列表中
                short_status = {  # 简化状态用于显示
                    "pg": status["policy_loss"],  # 策略损失
                    "rm": status["reward"],  # 奖励
                    "ret": status["return"],  # 回报
                    "glen": status["response_length"],  # 响应长度
                    "tlen": status["total_length"],  # 总长度
                    "kl": status["kl"],  # KL值
                }
                if "critic_loss" in status:  # 如果状态中包含评论家损失
                    short_status["cri"] = status["critic_loss"]  # 评论家损失
                    short_status["vals"] = status["values"]  # 值

                if "ptx_loss" in status:  # 如果状态中包含PTX损失
                    short_status["ptx"] = status["ptx_loss"]  # PTX损失
                pbar.set_postfix(short_status)  # 在进度条上显示简化的状态

        if status_list:  # 如果状态列表非空
            status_mean = status_list[0]  # 取列表中的第一个状态作为初始平均状态
            for m in status_list[1:]:  # 遍历状态列表中的其余状态
                for k, v in m.items():  # 遍历状态字典中的每个键值对
                    status_mean[k] += v  # 对状态求和
            for k in status_mean.keys():  # 遍历平均状态的每个键
                status_mean[k] /= len(status_list)  # 计算均值
        return status_mean  # 返回平均状态

    def training_step(self, experience: Experience) -> Dict[str, float]:  # 定义训练步骤函数
        status = self.training_step_actor(experience)  # 执行actor的训练步骤
        status.update(self.training_step_critic(experience))  # 更新critic的训练状态
        return status  # 返回训练状态

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:  # 定义actor的训练步骤函数
        self.actor.train()  # 设置actor为训练模式

        num_actions = experience.action_mask.size(1)  # 获取动作的数量
        # 计算动作对数概率和输出
        action_log_probs, output = self.actor(
            experience.sequences, num_actions, attention_mask=experience.attention_mask, return_output=True
        )

        # 计算actor损失
        actor_loss = self.actor_loss_fn(
            action_log_probs,  # 动作对数概率
            experience.action_log_probs,  # 经验中的动作对数概率
            experience.advantages,  # 经验中的优势
            action_mask=experience.action_mask,  # 动作掩码
        )
        # 辅助损失
        if self.aux_loss:
            aux_loss = output.aux_loss  # 获取辅助损失
        else:
            aux_loss = 0  # 如果没有辅助损失，设为0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef  # 总损失为actor损失加上辅助损失乘以系数
        self.strategy.backward(loss, self.actor, self.actor_optim)  # 反向传播损失

        # PTX损失
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)  # 获取预训练数据
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())  # 将inputs移动到当前设备
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())  # 将attention mask移动到当前设备
            label = torch.where(
                attention_mask.bool(),  # 判断attention mask的布尔值
                inputs,  # 如果为True，标签为inputs
                self.ptx_loss_fn.IGNORE_INDEX,  # 如果为False，标签为IGNORE_INDEX
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)  # 获取模型输出
            ptx_log_probs = output["logits"]  # 获取输出中的logits

            # 计算PTX损失
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)  # 计算PTX损失
            # 辅助损失
            if self.aux_loss:
                aux_loss = output.aux_loss  # 获取辅助损失
            else:
                aux_loss = 0  # 如果没有辅助损失，设为0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef  # 总损失为PTX损失加上辅助损失乘以系数
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)  # 反向传播损失

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")  # 执行优化器步骤
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")  # 更新EMA模型

        # 返回状态
        status = {
            "policy_loss": actor_loss.item(),  # 策略损失
        }
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()  # PTX损失
        for k, v in experience.info.items():  # 遍历经验信息
            if k == "kl":
                status[k] = (  # 如果是KL值，计算加权平均
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()  # 否则计算平均值
        return status  # 返回状态

    def training_step_critic(self, experience: Experience) -> Dict[str, float]:  # 定义critic的训练步骤函数
        self.critic.train()  # 设置critic为训练模式

        # 计算值和输出
        values, output = self.critic(
            experience.sequences,  # 输入序列
            action_mask=experience.action_mask,  # 动作掩码
            attention_mask=experience.attention_mask,  # 注意力掩码
            return_output=True,  # 返回输出
        )
        # 计算critic损失
        critic_loss = self.critic_loss_fn(
            values,  # 计算得到的值
            experience.values,  # 经验中的值
            experience.returns,  # 经验中的回报
            action_mask=experience.action_mask,  # 动作掩码
        )
        # 辅助损失
        if self.aux_loss:
            aux_loss = output.aux_loss  # 获取辅助损失
        else:
            aux_loss = 0  # 如果没有辅助损失，设为0
        loss = critic_loss + aux_loss * self.args.aux_loss_coef  # 总损失为critic损失加上辅助损失乘以系数
        self.strategy.backward(loss, self.critic, self.critic_optim)  # 反向传播损失
        self.strategy.optimizer_step(self.critic_optim, self.critic, self.critic_scheduler, name="critic")  # 执行优化器步骤

        # 返回状态
        status = {
            "critic_loss": critic_loss.item(),  # critic损失
            "values": masked_mean(values, experience.action_mask).item(),  # 掩码平均值
        }
        return status  # 返回状态

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):  # 定义保存日志和检查点的函数
        if global_step % args.logging_steps == 0:  # 如果当前步数是记录日志的步数
            # step bar
            step_bar.set_postfix(logs_dict)  # 设置进度条的后缀为日志字典
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():  # 如果使用WandB并且是主进程
                logs = {  # 日志字典
                    "train/%s" % k: v  # 训练日志的键值对
                    for k, v in {
                        **logs_dict,  # 合并日志字典
                        "global_step": global_step,  # 全局步数
                    }.items()  # 遍历键值对
                }
                self._wandb.log(logs)  # 记录日志

        # TODO: 为PPO添加评估机制
        if global_step % args.eval_steps == 0:  # 如果当前步数是评估的步数
            # self.evaluate(self.eval_dataloader, global_step)
            pass  # 暂时跳过评估

        # 保存检查点
        # TODO: 在开发集上保存最好的模型，使用整个开发集上的损失/困惑度/其他作为评价指标
        if global_step % args.save_steps == 0:  # 如果当前步数是保存检查点的步数
            tag = f"global_step{global_step}"  # 标签为全局步数
            self.strategy.save_ckpt(
                self.actor.model,  # 保存actor模型
                os.path.join(args.ckpt_path, "_actor"),  # 路径为保存路径加_actor
                tag,  # 标签为当前步数
                args.max_ckpt_num,  # 最大检查点数
                args.max_ckpt_mem,  # 最大检查点内存
            )
            self.strategy.save_ckpt(
                self.critic,  # 保存critic模型
                os.path.join(args.ckpt_path, "_critic"),  # 路径为保存路径加_critic
                tag,  # 标签为当前步数
                args.max_ckpt_num,  # 最大检查点数
                args.max_ckpt_mem,  # 最大检查点内存
            )