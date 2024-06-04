import argparse  # 导入argparse模块，用于解析命令行参数
import itertools  # 导入itertools模块，用于高效的循环迭代
import math  # 导入math模块，提供数学运算函数
import os  # 导入os模块，提供操作系统接口
from copy import deepcopy  # 从copy模块导入deepcopy，用于深拷贝对象
from datetime import datetime  # 从datetime模块导入datetime，用于处理日期和时间

import torch  # 导入torch模块，用于深度学习框架PyTorch
from transformers.trainer import get_scheduler  # 从transformers.trainer模块导入get_scheduler，用于获取学习率调度器

from openrlhf.datasets import PromptDataset, SFTDataset  # 从openrlhf.datasets模块导入PromptDataset和SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression  # 从openrlhf.models模块导入Actor类和get_llm_for_sequence_regression函数
from openrlhf.trainer import PPOTrainer  # 从openrlhf.trainer模块导入PPOTrainer类
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer  # 从openrlhf.utils模块导入blending_datasets, get_strategy和get_tokenizer函数


def train(args):
    # 配置策略
    strategy = get_strategy(args)  # 获取训练策略
    strategy.setup_distributed()  # 设置分布式训练

    # 配置模型
    # 加载huggingface模型
    actor = Actor(
        args.pretrain,  # 预训练模型路径
        use_flash_attention_2=args.flash_attn,  # 是否使用闪电注意力2
        bf16=args.bf16,  # 是否使用bfloat16精度
        load_in_4bit=args.load_in_4bit,  # 是否以4位精度加载模型
        lora_rank=args.lora_rank,  # LoRA秩参数
        lora_alpha=args.lora_alpha,  # LoRA alpha参数
        target_modules=args.target_modules,  # 目标模块
        lora_dropout=args.lora_dropout,  # LoRA dropout率
        ds_config=strategy.get_ds_train_config(is_actor=True),  # 获取分布式训练配置
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())  # 将actor模型移到GPU上

    critic = get_llm_for_sequence_regression(
        args.reward_pretrain,  # 预训练奖励模型路径
        "critic",  # 模型类型为critic
        normalize_reward=args.normalize_reward,  # 是否归一化奖励
        use_flash_attention_2=args.flash_attn,  # 是否使用闪电注意力2
        bf16=args.bf16,  # 是否使用bfloat16精度
        load_in_4bit=args.load_in_4bit,  # 是否以4位精度加载模型
        lora_rank=args.lora_rank,  # LoRA秩参数
        lora_alpha=args.lora_alpha,  # LoRA alpha参数
        target_modules=args.target_modules,  # 目标模块
        lora_dropout=args.lora_dropout,  # LoRA dropout率
        ds_config=strategy.get_ds_train_config(is_actor=False),  # 获取分布式训练配置（非actor）
    )
    reward_model = get_llm_for_sequence_regression(
        args.reward_pretrain,  # 预训练奖励模型路径
        "reward",  # 模型类型为reward
        normalize_reward=args.normalize_reward,  # 是否归一化奖励
        use_flash_attention_2=args.flash_attn,  # 是否使用闪电注意力2
        bf16=args.bf16,  # 是否使用bfloat16精度
        load_in_4bit=args.load_in_4bit,  # 是否以4位精度加载模型
        ds_config=strategy.get_ds_train_config(is_actor=False),  # 获取分布式训练配置（非actor）
    )

    # 配置tokenizer
    tokenizer = get_tokenizer(
        args.pretrain,  # 预训练模型路径
        actor.model,  # actor模型
        "left",  # 对齐方式为左对齐
        strategy,  # 分布式训练策略
        use_fast=not args.disable_fast_tokenizer  # 是否使用快速tokenizer
    )
    get_tokenizer(
        args.reward_pretrain,  # 预训练奖励模型路径
        critic,  # critic模型
        "left",  # 对齐方式为左对齐
        strategy,  # 分布式训练策略
        use_fast=not args.disable_fast_tokenizer  # 是否使用快速tokenizer
    )
    get_tokenizer(
        args.reward_pretrain,  # 预训练奖励模型路径
        reward_model,  # reward模型
        "left",  # 对齐方式为左对齐
        strategy,  # 分布式训练策略
        use_fast=not args.disable_fast_tokenizer  # 是否使用快速tokenizer
    )

    strategy.print(actor)  # 打印actor模型信息
    strategy.print(critic)  # 打印critic模型信息

    # 加载参考actor的权重
    initial_model = Actor(
        args.pretrain,  # 预训练模型路径
        use_flash_attention_2=args.flash_attn,  # 是否使用闪电注意力2
        bf16=args.bf16,  # 是否使用bfloat16精度
        load_in_4bit=args.load_in_4bit,  # 是否以4位精度加载模型
        ds_config=strategy.get_ds_eval_config(offload=False),  # 获取评估配置（不卸载）
    )
    get_tokenizer(
        args.pretrain,  # 预训练模型路径
        initial_model.model,  # 初始actor模型
        "left",  # 对齐方式为左对齐
        strategy  # 分布式训练策略
    )

    strategy.print("reward normalization status: {}".format(args.normalize_reward))  # 打印奖励归一化状态
    strategy.print("mean: {}, std {}".format(reward_model.mean, reward_model.std))  # 打印奖励模型的均值和标准差

    if args.enable_ema:
        ema_model = Actor(
            args.pretrain,  # 预训练模型路径
            use_flash_attention_2=args.flash_attn,  # 是否使用闪电注意力2
            bf16=args.bf16,  # 是否使用bfloat16精度
            load_in_4bit=args.load_in_4bit,  # 是否以4位精度加载模型
            ds_config=strategy.get_ds_eval_config(offload=True),  # 获取评估配置（卸载）
        )
    else:
        ema_model = None  # 如果不启用EMA，设置ema_model为None

    # 配置优化器
    # 为 actor 配置优化器，设置学习率、beta 参数和权重衰减
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
    )
    # 为 critic 配置优化器，设置学习率、beta 参数和权重衰减
    critic_optim = strategy.create_optimizer(
        critic, lr=args.critic_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
    )

    # 准备数据集
    # 将不同来源的数据集混合
    prompts_data = blending_datasets(
        args.prompt_data,             # 提示数据集路径
        args.prompt_data_probs,       # 各数据集的权重
        strategy,                     # 策略对象
        args.seed,                    # 随机种子
        max_count=args.max_samples,   # 最大样本数量
        return_eval=False,            # 不返回评估数据
    )
    # 根据最大样本数选择数据
    prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    # 创建数据集实例，使用指定的 tokenizer 和输入模板
    prompts_dataset = PromptDataset(
        prompts_data,                 # 提示数据
        tokenizer,                    # 分词器
        strategy,                     # 策略对象
        input_template=args.input_template  # 输入模板
    )
    # 配置数据加载器
    prompts_dataloader = strategy.setup_dataloader(
        prompts_dataset,               # 提示数据集
        args.micro_rollout_batch_size, # 批量大小
        True,                          # 是否打乱数据
        True                           # 是否在每个 epoch 后重新打乱
    )

    # 如果有预训练数据
    if args.pretrain_data:
        # 将不同来源的预训练数据集混合
        pretrain_data = blending_datasets(
            args.pretrain_data,            # 预训练数据集路径
            args.pretrain_data_probs,      # 各数据集的权重
            strategy,                      # 策略对象
            args.seed,                     # 随机种子
            return_eval=False,             # 不返回评估数据
        )
        # 设定预训练的最大长度
        pretrain_max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        # 创建预训练数据集实例
        pretrain_dataset = SFTDataset(
            pretrain_data.select(range(min(len(pretrain_data), args.max_epochs * len(prompts_dataset)))), # 选择数据
            tokenizer,                  # 分词器
            pretrain_max_len,           # 预训练的最大长度
            strategy,                   # 策略对象
            pretrain_mode=True,         # 预训练模式
        )
        # 配置预训练数据加载器，循环迭代
        pretrain_dataloader = itertools.cycle(
            iter(
                strategy.setup_dataloader(
                    pretrain_dataset,          # 预训练数据集
                    args.micro_train_batch_size, # 批量大小
                    True,                       # 是否打乱数据
                    True,                       # 是否在每个 epoch 后重新打乱
                    pretrain_dataset.collate_fn # 数据整理函数
                )
            )
        )
    else:
        # 如果没有预训练数据，设定为 None
        pretrain_dataloader = None

    # 配置调度器
    num_update_steps_per_episodes = (
        int(len(prompts_dataloader) * (args.micro_rollout_batch_size / args.micro_train_batch_size))
        * args.max_epochs
        // strategy.accumulated_gradient
    ) # 计算每个周期的更新步数

    # 计算最大训练步数
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    # 配置 actor 的学习率调度器，使用余弦退火调度策略
    actor_scheduler = get_scheduler(
        "cosine",                              # 调度器类型
        actor_optim,                           # 优化器
        num_warmup_steps=math.ceil(max_steps * 0.03), # 预热步数，通常是总步数的 3%
        num_training_steps=max_steps,          # 总训练步数
    )

    # 配置 critic 的学习率调度器，使用余弦退火调度策略
    critic_scheduler = get_scheduler(
        "cosine",                              # 调度器类型
        critic_optim,                          # 优化器
        num_warmup_steps=math.ceil(max_steps * 0.03), # 预热步数，通常是总步数的 3%
        num_training_steps=max_steps,          # 总训练步数
    )

    # 梯度检查点
    if args.gradient_checkpointing:
        # 启用 actor 的梯度检查点，使用指定参数
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        # 启用 critic 的梯度检查点，使用指定参数
        critic.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # 准备模型/优化器...
    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),   # 准备 actor 模型、优化器和调度器
        (critic, critic_optim, critic_scheduler), # 准备 critic 模型、优化器和调度器
        reward_model,                             # 奖励模型
        initial_model,                            # 初始模型
        is_rlhf=True,                             # 指定为强化学习
    )

    # 如果有 EMA 模型
    if ema_model:
        ema_model._offload = True                 # 启用 EMA 模型的卸载功能
        ema_model = strategy.prepare(ema_model, is_rlhf=True) # 准备 EMA 模型

    # 加载检查点
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path) # 打印加载检查点的信息

    # 创建保存路径，如果不存在则创建
    os.makedirs(args.save_path, exist_ok=True)

    # 配置训练器
    trainer = PPOTrainer(
        strategy,                         # 策略对象
        actor,                            # actor 模型
        critic,                           # critic 模型
        reward_model,                     # 奖励模型
        initial_model,                    # 初始模型
        ema_model,                        # EMA 模型
        actor_optim,                      # actor 优化器
        critic_optim,                     # critic 优化器
        actor_scheduler,                  # actor 调度器
        critic_scheduler,                 # critic 调度器
        max_epochs=args.max_epochs,       # 最大训练周期数
        micro_train_batch_size=args.micro_train_batch_size, # 训练批量大小
        micro_rollout_batch_size=args.micro_rollout_batch_size, # 展开批量大小
        gradient_checkpointing=args.gradient_checkpointing,     # 是否启用梯度检查点
        tokenizer=tokenizer,              # 分词器
        prompt_max_len=args.prompt_max_len, # 提示最大长度
        value_clip=args.value_clip,       # 值剪裁参数
        eps_clip=args.eps_clip,           # Epsilon 剪裁参数
        gamma=args.gamma,                 # 折扣因子
        lambd=args.lambd,                 # Lambda 参数
        init_kl_coef=args.init_kl_coef,   # 初始 KL 系数
        kl_target=args.kl_target,         # KL 目标值
        ema_beta=0.992,                   # EMA 的 Beta 参数
        ptx_coef=args.ptx_coef,           # PTX 系数
        max_norm=args.max_norm,           # 最大范数
        # GPT 生成相关参数
        do_sample=True,                   # 是否进行采样
        max_new_tokens=args.generate_max_len, # 最大新生成的 token 数量
        max_length=args.max_len,          # 最大长度
        temperature=args.temperature,     # 采样温度
        top_p=args.top_p,                 # Top-p 采样参数
        pad_token_id=tokenizer.pad_token_id, # 填充 token 的 ID
        eos_token_id=tokenizer.eos_token_id, # 结束 token 的 ID
    )

    # 训练模型
    trainer.fit(
        prompts_dataloader,              # 提示数据加载器
        pretrain_dataloader,             # 预训练数据加载器
        args,                            # 训练参数
    )

    # 仅在rank0上保存模型检查点
    strategy.save_model(
        ema_model if args.enable_ema else actor, # 保存 EMA 模型或 actor 模型
        tokenizer,                               # 分词器
        args.save_path,                          # 保存路径
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument("--prompt_data", type=str, default=None)  # 提示数据路径
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",  # 数据集的采样概率
    )
    parser.add_argument("--pretrain_data", type=str, default=None)  # 预训练数据路径
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",  # 数据集的采样概率
    )
    parser.add_argument("--pretrain", type=str, default=None)  # 预训练路径
    parser.add_argument("--reward_pretrain", type=str, default=None)  # 奖励预训练路径
    parser.add_argument("--save_path", type=str, default="./ckpt")  # 保存路径
    parser.add_argument("--save_steps", type=int, default=-1)  # 保存间隔步数
    parser.add_argument("--logging_steps", type=int, default=1)  # 日志记录间隔步数
    parser.add_argument("--eval_steps", type=int, default=-1)  # 评估间隔步数
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")  # 检查点路径
    parser.add_argument("--max_ckpt_num", type=int, default=3)  # 最大检查点数量
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)  # 最大检查点内存（GB）
    parser.add_argument("--num_episodes", type=int, default=1)  # 训练回合数
    parser.add_argument("--rollout_batch_size", type=int, default=512)  # 展开批量大小
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)  # 微展开批量大小
    parser.add_argument("--max_epochs", type=int, default=1)  # 最大训练周期数
    parser.add_argument("--prompt_max_len", type=int, default=1024)  # 提示最大长度
    parser.add_argument("--generate_max_len", type=int, default=1024)  # 生成最大长度
    parser.add_argument("--max_len", type=int, default=None)  # 最大长度
    parser.add_argument("--max_samples", type=int, default=100000)  # 最大样本数
    parser.add_argument("--max_norm", type=float, default=1.0)  # 最大范数
    parser.add_argument("--l2", type=float, default=0.0)  # L2正则化系数
    parser.add_argument("--ptx_coef", type=float, default=0.05)  # 预训练系数
    parser.add_argument("--eps_clip", type=float, default=0.2)  # Epsilon剪裁系数
    parser.add_argument("--value_clip", type=float, default=0.2)  # 值剪裁系数
    parser.add_argument("--lambd", type=float, default=0.95)  # Lambda参数
    parser.add_argument("--gamma", type=float, default=1)  # 折扣因子
    parser.add_argument("--micro_train_batch_size", type=int, default=4)  # 微训练批量大小
    parser.add_argument("--train_batch_size", type=int, default=128)  # 训练批量大小
    parser.add_argument("--load_checkpoint", action="store_true", default=False)  # 是否加载检查点
    parser.add_argument("--normalize_reward", action="store_true", default=False)  # 是否标准化奖励
    parser.add_argument("--top_p", type=float, default=1.0)  # Top-p采样参数
    parser.add_argument("--temperature", type=float, default=1.0)  # 采样温度
    parser.add_argument("--seed", type=int, default=42)  # 随机种子

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")  # 本地rank
    parser.add_argument("--zero_stage", type=int, default=2)  # ZeRO阶段
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)  # 梯度检查点
    parser.add_argument("--bf16", action="store_true", default=False)  # 使用 bf16 精度
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)  # actor 学习率
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)  # critic 学习率
    parser.add_argument("--kl_target", type=float, default=None)  # KL目标值
    parser.add_argument("--init_kl_coef", type=float, default=0.02)  # 初始KL系数
    # 使 EMA 成为可选功能
    parser.add_argument("--enable_ema", action="store_true", help="Enable EMA checkpoint for the model.")  # 启用EMA检查点
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")  # ZeRO++最大分区大小
    parser.add_argument("--adam_offload", action="store_true", default=False)  # 启用Adam卸载
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)  # 在GPU上初始化actor
    parser.add_argument("--flash_attn", action="store_true", default=False)  # 启用闪存注意力
    parser.add_argument("--aux_loss_coef", type=float, default=0)  # 辅助损失系数
    parser.add_argument("--grad_accum_dtype", type=str, default=None)  # 梯度累积数据类型
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)  # 禁用跟踪缓存
    parser.add_argument("--load_in_4bit", action="store_true", default=False)  # 以4位加载
    parser.add_argument("--lora_rank", type=int, default=0)  # LoRA等级
    parser.add_argument("--lora_alpha", type=int, default=16)  # LoRA alpha
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")  # 目标模块
    parser.add_argument("--lora_dropout", type=float, default=0)  # LoRA dropout
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")  # 输入模板
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")  # 使用再进入的梯度检查点
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)  # 禁用快速分词器

    # 自定义数据集键名
    parser.add_argument("--input_key", type=str, default=None)  # 输入键名

    # wandb参数
    parser.add_argument("--use_wandb", type=str, default=None)  # 是否使用wandb
    parser.add_argument("--wandb_org", type=str, default=None)  # wandb组织
    parser.add_argument("--wandb_group", type=str, default=None)  # wandb组
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")  # wandb项目名称
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),  # wandb运行名称
    )

    args = parser.parse_args()  # 解析命令行参数
    train(args)  # 调用train函数进行训练
