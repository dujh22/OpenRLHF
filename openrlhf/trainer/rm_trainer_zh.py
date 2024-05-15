import math  # 导入数学库
from abc import ABC  # 导入ABC模块用于创建抽象基类

import loralib as lora  # 导入loralib库并简写为lora
import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch库中导入神经网络模块
from torch.optim import Optimizer  # 从PyTorch库中导入优化器模块
from torch.utils.data import DistributedSampler  # 从PyTorch库中导入分布式采样器模块
from tqdm import tqdm  # 导入tqdm库用于显示进度条

from openrlhf.models import LogExpLoss, PairWiseLoss, SwitchBalancingLoss  # 导入自定义的损失函数类

# 奖励模型训练器类，继承自抽象基类ABC
class RewardModelTrainer(ABC):
    """
        在训练奖励模型时使用的训练器。

    参数:
        model (torch.nn.Module): 训练的模型
        strategy (Strategy): 训练时使用的策略
        optim(Optimizer): 训练时使用的优化器
        train_dataset (RewardDataset): 用于训练的数据集
        eval_dataset (RewardDataset): 用于评估的数据集
        batch_size (int, 默认值为1): 训练时的批量大小
        max_epochs (int, 默认值为2): 训练的最大轮数
        optim_kwargs (dict, 默认值为{'lr':1e-4}): 初始化优化器时使用的关键字参数
    """

    # 初始化函数
    def __init__(
        self,
        model,  # 模型
        strategy,  # 训练策略
        optim: Optimizer,  # 优化器
        train_dataloader,  # 训练数据加载器
        eval_dataloader,  # 评估数据加载器
        scheduler,  # 学习率调度器
        tokenizer,  # 分词器
        max_norm=0.5,  # 梯度裁剪的最大范数
        max_epochs: int = 2,  # 最大训练轮数
        loss="sigmoid",  # 损失函数类型
    ) -> None:
        super().__init__()  # 调用父类的初始化函数
        self.strategy = strategy  # 初始化训练策略
        self.epochs = max_epochs  # 设置最大训练轮数
        self.max_norm = max_norm  # 设置梯度裁剪的最大范数
        self.model = model  # 初始化模型
        self.train_dataloader = train_dataloader  # 初始化训练数据加载器
        self.eval_dataloader = eval_dataloader  # 初始化评估数据加载器
        self.scheduler = scheduler  # 初始化学习率调度器
        self.optimizer = optim  # 初始化优化器
        self.tokenizer = tokenizer  # 初始化分词器
        self.args = strategy.args  # 获取策略参数

        # 根据指定的损失函数类型初始化损失函数
        if loss == "sigmoid":
            self.loss_fn = PairWiseLoss()  # 使用PairWiseLoss损失函数
            self.strategy.print("LogSigmoid Loss")  # 打印使用的损失函数类型
        else:
            self.loss_fn = LogExpLoss()  # 使用LogExpLoss损失函数
            self.strategy.print("LogExp Loss")  # 打印使用的损失函数类型

        self.aux_loss = self.args.aux_loss_coef > 1e-8  # 辅助损失系数

        self.margin_loss = self.strategy.args.margin_loss  # 是否使用边缘损失
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss  # 是否使用fp32精度计算损失

        self._wandb = None  # 初始化wandb
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():  # 如果使用wandb且当前为主进程
            import wandb  # 导入wandb库

            self._wandb = wandb  # 初始化wandb
            if not wandb.api.api_key:  # 如果没有wandb API密钥
                wandb.login(key=strategy.args.use_wandb)  # 使用策略参数中的密钥登录wandb
            wandb.init(
                entity=strategy.args.wandb_org,  # 设置wandb实体
                project=strategy.args.wandb_project,  # 设置wandb项目
                group=strategy.args.wandb_group,  # 设置wandb组
                name=strategy.args.wandb_run_name,  # 设置wandb运行名称
                config=strategy.args.__dict__,  # 设置wandb配置
                reinit=True,  # 重新初始化
            )

            # 定义wandb指标
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    # 训练函数
    def fit(self, args):
        # 获取评估和保存步骤
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # 每个epoch评估一次
        if args.save_steps == -1:
            args.save_steps = float("inf")  # 不保存ckpt

        global_step = 1  # 初始化全局步骤计数器
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())  # 显示训练轮数的进度条
        for epoch in range(self.epochs):  # 遍历每个训练轮次
            # 训练
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),  # 获取训练数据的长度
                desc="Train step of epoch %d" % epoch,  # 显示当前轮次的训练步骤
                disable=not self.strategy.is_rank_0(),  # 如果不是主进程则禁用进度条
            )

            if isinstance(self.train_dataloader.sampler, DistributedSampler):  # 如果使用分布式采样器
                self.train_dataloader.sampler.set_epoch(epoch)  # 设置当前轮次

            self.model.train()  # 设置模型为训练模式
            acc_mean = 0  # 初始化平均准确率
            loss_mean = 0  # 初始化平均损失
            for chosen_ids, c_mask, reject_ids, r_mask, margin in self.train_dataloader:  # 遍历训练数据加载器
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())  # 获取选择的ID并移动到当前设备
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())  # 获取选择的掩码并移动到当前设备
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())  # 获取拒绝的ID并移动到当前设备
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())  # 获取拒绝的掩码并移动到当前设备

                if self.margin_loss:  # 如果使用边缘损失
                    margin = torch.tensor(margin).to(torch.cuda.current_device())  # 将边缘移动到当前设备
                else:
                    margin = None  # 否则将边缘设置为None

                chosen_reward, reject_reward, aux_loss = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )  # 前向传播计算奖励和辅助损失

                # 损失函数
                if self.compute_fp32_loss:  # 如果使用fp32精度计算损失
                    chosen_reward = chosen_reward.float()  # 将选择的奖励转换为浮点数
                    reject_reward = reject_reward.float()  # 将拒绝的奖励转换为浮点数

                preference_loss = self.loss_fn(chosen_reward, reject_reward, margin)  # 计算偏好损失
                # mixtral
                if not self.aux_loss:  # 如果不使用辅助损失
                    aux_loss = 0  # 将辅助损失设置为0

                loss = preference_loss + aux_loss * self.args.aux_loss_coef  # 计算总损失
                self.strategy.backward(loss, self.model, self.optimizer)  # 反向传播计算梯度
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)  # 执行优化器步骤

                acc_mean = acc_mean * 0.9 + 0.1 * (chosen_reward > reject_reward).float().mean().item()  # 更新平均准确率
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()  # 更新平均损失
                # 可选的奖励模型信息
                logs_dict = {
                    "preference_loss": preference_loss.item(),  # 偏好损失
                    "chosen_reward": chosen_reward.mean().item(),  # 平均选择奖励
                    "reject_reward": reject_reward.mean().item(),  # 平均拒绝奖励
                    "acc_mean": acc_mean,  # 平均准确率
                    "loss_mean": loss_mean,  # 平均损失
                }
                if self.aux_loss:  # 如果使用辅助损失
                    logs_dict["aux_loss"] = aux_loss.item()  # 添加辅助损失到日志字典
                # 日志/检查点/评估
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)  # 保存日志和检查点

                step_bar.update()  # 更新进度条
                global_step += 1  # 增加全局步骤计数器
            epoch_bar.update()  # 更新训练轮次的进度条

        if self._wandb is not None and self.strategy.is_rank_0():  # 如果使用wandb且当前为主进程
            self._wandb.finish()  # 结束wandb运行

    # 日志/检查点/评估
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}):
        if global_step % args.logging_steps == 0:  # 如果当前步骤为日志记录步骤
            # 步骤条
            logs_dict = self.strategy.all_reduce(logs_dict)  # 聚合所有进程的日志字典
            step_bar.set_postfix(logs_dict)  # 设置步骤条的后缀为日志字典

            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):  # 如果使用wandb且当前为主进程且当前步骤为累积梯度步骤
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}  # 构建训练日志
                self._wandb.log(logs)  # 记录wandb日志

        # 评估
        if global_step % args.eval_steps == 0:  # 如果当前步骤为评估步骤
            self.evaluate(self.eval_dataloader, global_step)  # 评估模型
        # 保存检查点
        if global_step % args.save_steps == 0:  # 如果当前步骤为保存步骤
            tag = f"global_step{global_step}"  # 构建检查点标签
            self.strategy.save_ckpt(self.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)  # 保存检查点

    # 评估函数
    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),  # 获取评估数据的长度
            desc="Eval stage of steps %d" % steps,  # 显示当前步骤的评估阶段
            disable=not self.strategy.is_rank_0(),  # 如果不是主进程则禁用进度条
        )
        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():  # 禁用梯度计算
            acc = 0  # 初始化准确率
            rewards = []  # 初始化奖励列表
            loss_sum = 0  # 初始化损失总和
            for chosen_ids, c_mask, reject_ids, r_mask, margin in eval_dataloader:  # 遍历评估数据加载器
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())  # 获取选择的ID并移动到当前设备
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())  # 获取选择的掩码并移动到当前设备
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())  # 获取拒绝的ID并移动到当前设备
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())  # 获取拒绝的掩码并移动到当前设备
                margin = torch.tensor(margin).to(torch.cuda.current_device())  # 将边缘移动到当前设备

                chosen_reward, reject_reward, _ = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask
                )  # 前向传播计算奖励

                loss = self.loss_fn(chosen_reward, reject_reward, margin)  # 计算损失

                rewards += [chosen_reward.flatten(), reject_reward.flatten()]  # 将奖励添加到奖励列表
                acc += (chosen_reward > reject_reward).float().mean().item()  # 计算准确率
                loss_sum += loss.item()  # 计算损失总和
                step_bar.update()  # 更新进度条

            acc_mean = acc / self.eval_dataloader.__len__()  # 计算平均准确率
            loss_mean = loss_sum / self.eval_dataloader.__len__()  # 计算平均损失

            rewards = torch.cat(rewards).float()  # 拼接奖励列表并转换为浮点数
            rewards = self.strategy.all_gather(rewards)  # 聚合所有进程的奖励
            reward_mean = torch.mean(rewards)  # 计算奖励的均值
            reward_std = torch.std(rewards).clamp(min=1e-8)  # 计算奖励的标准差并限制最小值为1e-8

            # 保存均值和标准差
            self.strategy.print("Set reward mean std")  # 打印设置信息
            unwrap_model = self.strategy._unwrap_model(self.model)  # 获取未包装的模型
            unwrap_model.config.mean = reward_mean.item()  # 设置奖励均值
            unwrap_model.config.std = reward_std.item()  # 设置奖励标准差

            bar_dict = {
                "eval_loss": loss_mean,  # 评估损失
                "acc_mean": acc_mean,  # 平均准确率
                "reward_mean": reward_mean.item(),  # 奖励均值
                "reward_std": reward_std.item(),  # 奖励标准差
            }
            logs = self.strategy.all_reduce(bar_dict)  # 聚合所有进程的日志字典
            step_bar.set_postfix(logs)  # 设置步骤条的后缀为日志字典

            histgram = torch.histogram(rewards.cpu(), bins=10, range=(-10, 10), density=True) * 2  # 计算奖励的直方图
            self.strategy.print("histgram")  # 打印直方图信息
            self.strategy.print(histgram)  # 打印直方图

            if self._wandb is not None and self.strategy.is_rank_0():  # 如果使用wandb且当前为主进程
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}  # 构建评估日志
                self._wandb.log(logs)  # 记录wandb日志
        self.model.train()  # 重置模型状态为训练模式

    # 拼接前向传播函数
    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask):
        """在给定的一批输入上运行给定的模型，将选择和拒绝的输入拼接在一起。

        这样做是为了避免进行两次前向传播，因为对于FSDP来说这样更快。
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)  # 拼接输入
        all_values, output = model(input_ids, attention_mask=att_masks, return_output=True)  # 模型前向传播
        chosen_rewards = all_values[: chosen_ids.shape[0]]  # 获取选择的奖励
        rejected_rewards = all_values[chosen_ids.shape[0] :]  # 获取拒绝的奖励
        aux_loss = output.aux_loss if "aux_loss" in output else []  # 获取辅助损失
        return chosen_rewards, rejected_rewards, aux_loss  # 返回选择的奖励、拒绝的奖励和辅助损失

    # 拼接输入函数
    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """将选择和拒绝的输入拼接成一个张量。

        参数:
            batch: 一批数据。必须包含键'chosen_input_ids'和'rejected_input_ids'，这些是形状为(batch_size, sequence_length)的张量。

        返回:
            一个包含拼接输入的字典，键为'concatenated_input_ids'。
        """

        # 填充到指定长度的函数
        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:  # 如果张量的长度大于等于指定长度
                return tensor  # 返回原始张量
            else:
                pad_size = list(tensor.shape)  # 获取张量的形状
                pad_size[dim] = length - tensor.size(dim)  # 计算填充长度
                # 左侧填充
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )  # 拼接填充张量和原始张量

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])  # 计算最大长度
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),  # 填充选择的ID
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),  # 填充拒绝的ID
            ),
            dim=0,
        )  # 拼接选择的ID和拒绝的ID
        max_length = max(c_mask.shape[1], r_mask.shape[1])  # 计算掩码的最大长度
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)  # 拼接掩码
        return inputs_ids, att_masks  # 返回拼接后的输入和掩码
