# 这段代码是一个使用 Hugging Face Transformers 和自定义组件来训练奖励模型的 Python 脚本。

import argparse  # 导入命令行参数解析模块
import math  # 导入数学运算模块
import os  # 导入操作系统相关模块
from collections import OrderedDict  # 导入有序字典类
from datetime import datetime  # 导入日期时间模块

from transformers.trainer import get_scheduler  # 导入获取学习率调度器的函数

from openrlhf.datasets import RewardDataset  # 导入奖励数据集类
from openrlhf.models import get_llm_for_sequence_regression  # 导入获取序列回归模型的函数
from openrlhf.trainer import RewardModelTrainer  # 导入奖励模型训练器类
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer  # 导入工具函数

def train(args):
    # 配置训练策略
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # 配置模型
    model = get_llm_for_sequence_regression(
        args.pretrain,
        "reward",
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        init_value_head=True,
    )

    # 配置分词器
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # 打印模型信息
    strategy.print(model)

    # 配置优化器
    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=args.l2)

    # 准备数据和数据集
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=5000000,
        stopping_strategy="all_exhausted",
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    train_dataset = RewardDataset(train_data, tokenizer, args.max_len, strategy, input_template=args.input_template)
    eval_dataset = RewardDataset(eval_data, tokenizer, args.max_len, strategy, input_template=args.input_template)

    # 配置数据加载器
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset, args.micro_train_batch_size, True, False, eval_dataset.collate_fn
    )

    # 配置学习率调度器
    num_update_steps_per_epoch = len(train_dataloader) * args.max_epochs // strategy.accumulated_gradient
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)
    scheduler = get_scheduler(
        "cosine",
        optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
    )

    # 启用梯度检查点
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # 准备训练策略
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # 加载检查点
    if args.load_checkpoint:
        strategy.print("Load checkpoint: ", args.save_path)

    # 创建保存路径
    os.makedirs(args.save_path, exist_ok=True)

    # 创建并启动训练器
    trainer = RewardModelTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        max_epochs=args.max_epochs,
        loss=args.loss,
    )
    trainer.fit(args)

    # 保存模型检查点


    strategy.save_model(model, tokenizer, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 添加各种命令行参数

    # 预训练模型的名称或路径
    parser.add_argument("--pretrain", type=str, default="bigscience/bloomz-1b7")

    # 训练和评估使用的数据集名称
    parser.add_argument("--dataset", type=str, default="Dahoas/full-hh-rlhf")

    # 数据集采样概率
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")

    # 模型保存路径
    parser.add_argument("--save_path", type=str, default="./ckpt")

    # 每多少步保存一次模型，-1 表示不自动保存
    parser.add_argument("--save_steps", type=int, default=-1)

    # 每多少步记录一次日志
    parser.add_argument("--logging_steps", type=int, default=1)

    # 每多少步进行一次评估
    parser.add_argument("--eval_steps", type=int, default=-1)

    # 检查点保存路径
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_rm")

    # 最多保存多少个检查点
    parser.add_argument("--max_ckpt_num", type=int, default=3)

    # 检查点文件总大小限制（单位GB）
    parser.add_argument("--max_ckpt_mem", type=int, default=1000)

    # 训练的最大周期数
    parser.add_argument("--max_epochs", type=int, default=1)

    # 训练时每个微批次的大小
    parser.add_argument("--micro_train_batch_size", type=int, default=8)

    # 训练时批量大小
    parser.add_argument("--train_batch_size", type=int, default=128)

    # 最大样本数量
    parser.add_argument("--max_samples", type=int, default=1000000)

    # 是否加载检查点
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # 梯度裁剪的最大范数
    parser.add_argument("--max_norm", type=float, default=1.0)

    # 输入序列的最大长度
    parser.add_argument("--max_len", type=int, default=512)

    # L2 正则化系数
    parser.add_argument("--l2", type=float, default=0.0)

    # 损失函数类型
    parser.add_argument("--loss", type=str, default="sigmoid")

    # 是否启用梯度检查点
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # 随机种子
    parser.add_argument("--seed", type=int, default=42)

    # DeepSpeed 训练时的本地排名
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")

    # DeepSpeed 的 ZeRO 阶段
    parser.add_argument("--zero_stage", type=int, default=2)

    # 是否使用 bfloat16 训练
    parser.add_argument("--bf16", action="store_true", default=False)

    # 学习率
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # ZeRO++的最大分区大小
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")

    # 是否将 Adam 优化器的状态卸载到CPU
    parser.add_argument("--adam_offload", action="store_true", default=False)

    # 是否使用 Flash Attention 优化
    parser.add_argument("--flash_attn", action="store_true", default=False)

    # 是否在 FP32 精度下计算损失
    parser.add_argument("--compute_fp32_loss", action="store_true", default=False)

    # 是否使用边际损失
    parser.add_argument("--margin_loss", action="store_true", default=False)

    # 辅助损失系数
    parser.add_argument("--aux_loss_coef", type=float, default=0)

    # 梯度累积的数据类型
    parser.add_argument("--grad_accum_dtype", type=str, default=None)

    # 是否禁用跟踪缓存
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)

    # 是否在4bit精度下加载模型
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # LoRA的秩
    parser.add_argument("--lora_rank", type=int, default=0)

    # LoRA的α值
    parser.add_argument("--lora_alpha", type=int, default=16)

    # LoRA的dropout比率
    parser.add_argument("--lora_dropout", type=float, default=0)

    # 目标模块类型
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")

    # 输入模板
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")

    # 梯度检查点是否使用 reentrant
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")

    # 是否禁用快速分词器
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # 数据集关键字
    parser.add_argument("--prompt_key", type=str, default=None)
    parser.add_argument("--chosen_key", type=str, default=None)
    parser.add_argument("--rejected_key", type=str, default=None)

    # Weights and Biases (wandb) 集成配置
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_rm")
    parser.add_argument("--wandb_run_name", type=str, default="rm_%s" % datetime.now().strftime("%m%dT%H:%M"))



    args = parser.parse_args()
    train(args)
