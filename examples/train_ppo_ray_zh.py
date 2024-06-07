import argparse  # 导入 argparse 模块，用于解析命令行参数
from datetime import datetime  # 导入 datetime 模块，用于处理日期和时间
from typing import List  # 导入 List 类型，用于类型注解

import ray  # 导入 ray 模块，用于分布式计算
import torch  # 导入 torch 模块，用于深度学习
from ray.util.placement_group import placement_group  # 从 ray.util.placement_group 导入 placement_group

# 从 openrlhf.trainer.ray 导入多个类和函数
from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
)
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer  # 从 openrlhf.utils 导入多个实用函数

# 注意：这是用于多个奖励模型的奖励函数，请用你自己的函数替换它！
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)  # 将多个奖励张量堆叠并在第0维求和

def _validate_args(args):
    actor_world_size = args.actor_num_nodes * args.actor_num_gpus_per_node  # 计算 actor 的世界大小
    critic_world_size = args.critic_num_nodes * args.critic_num_gpus_per_node  # 计算 critic 的世界大小

    # 断言 actor_world_size 是 2 的幂
    assert (
        actor_world_size & (actor_world_size - 1)
    ) == 0, f"actor_world_size must be power of 2, got {actor_world_size}"
    # 断言 critic_world_size 是 2 的幂
    assert (
        critic_world_size & (critic_world_size - 1)
    ) == 0, f"critic_world_size must be power of 2, got {critic_world_size}"
    # 断言 actor_world_size 可以被 critic_world_size 整除
    assert (
        actor_world_size % critic_world_size == 0
    ), f"actor_world_size must be divisible by critic_world_size, got {actor_world_size} and {critic_world_size}"

    # 断言在使用 ZeRO-3 阶段时，vLLM 引擎的数量大于 0
    assert args.zero_stage != 3 or args.vllm_num_engines > 0, f"ZeRO-3 is only supported when vLLM enabled"


def train(args):
    _validate_args(args)  # 验证输入参数

    # 配置策略
    strategy = get_strategy(args)

    # 如果需要共置，显式地为 actor 和参考模型创建 placement group
    pg = None
    if args.colocate_actor_ref:
        assert (
            args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
        ), f"num_nodes 和 num_gpus_per_node 必须相同，当共置 actor 和参考模型时."

        # 为每个节点分配 GPU 和 CPU 资源
        bundles = [
            {"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node}
            for _ in range(args.actor_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")  # 创建 placement group
        ray.get(pg.ready())  # 等待 placement group 准备就绪

    # 注意：为什么在共置模型时不给每个 actor 分配 0.5 个 GPU？
    # 假设我们有一个节点有 4 个 GPU，并且每个模型的 num_gpus_per_node 是 4。
    # 如果我们为 actor 和参考模型分别分配 0.5 个 GPU，那么 GPU 分配将是
    #   |actor|actor|actor|actor|  ref | ref  | ref  | ref |
    #   |GPU0 |GPU0 |GPU1 |GPU1 | GPU2 | GPU2 | GPU3 | GPU3 |
    #
    # 因此，0.75/0.25 GPU 是一种让 Ray 在所有 GPU 上均匀分布所有模型的巧妙方法。
    #   |actor| ref  |actor| ref  |actor| ref  |actor|ref  |
    #   |GPU0 | GPU0 |GPU1 | GPU1 |GPU2 | GPU2 |GPU3 | GPU3 |
    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    ref_model = PPORayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ReferenceModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.25 if pg else 1,
    )

    # 如果需要共置，显式地为 critic 和奖励模型创建 placement group
    pg = None
    if args.colocate_critic_reward:
        assert (
            args.critic_num_nodes == args.reward_num_nodes
            and args.critic_num_gpus_per_node == args.reward_num_gpus_per_node
        ), f"num_nodes 和 num_gpus_per_node 必须相同，当共置 critic 和奖励模型时."

        # 为每个节点分配 GPU 和 CPU 资源
        bundles = [
            {"GPU": args.critic_num_gpus_per_node, "CPU": args.critic_num_gpus_per_node}
            for _ in range(args.critic_num_nodes)
        ]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")  # 创建 placement group
        ray.get(pg.ready())  # 等待 placement group 准备就绪

    critic_model = PPORayActorGroup(
        args.critic_num_nodes,
        args.critic_num_gpus_per_node,
        CriticModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    # 多个奖励模型
    reward_pretrains = args.reward_pretrain.split(",")  # 分割奖励预训练模型列表
    reward_models = []
    for _ in reward_pretrains:
        reward_models.append(
            PPORayActorGroup(
                args.reward_num_nodes,
                args.reward_num_gpus_per_node,
                RewardModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )
        )

    # 初始化参考、奖励和演员模型
    refs = []
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
        refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))

    # 初始化用于文本生成的 vLLM 引擎
    vllm_engines = None
    if args.vllm_num_engines is not None:
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines, args.vllm_tensor_parallel_size, args.pretrain, args.seed
        )

    # critic 调度器的初始化依赖于 max_step，所以我们必须在 actor 之后初始化 critic
    # TODO: 使用第一个奖励模型作为 critic 模型
    max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
    refs.extend(critic_model.async_init_model_from_pretrained(strategy, reward_pretrains[0], max_steps))
    ray.get(refs)  # 等待所有模型初始化完成

    # 训练 actor 和 critic 模型
    refs = actor_model.async_fit_actor_model(
        critic_model, ref_model, reward_models, reward_fn=reward_fn, vllm_engines=vllm_engines
    )
    ray.get(refs)  # 等待训练完成

    # 保存模型
    ray.get(actor_model.async_save_actor_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建 ArgumentParser 对象，用于解析命令行参数
    
    # 添加命令行参数及其说明
    parser.add_argument("--ref_num_nodes", type=int, default=1, help="参考模型的节点数")
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8, help="每个节点的 GPU 数量（参考模型）")
    parser.add_argument("--reward_num_nodes", type=int, default=1, help="奖励模型的节点数")
    parser.add_argument("--reward_num_gpus_per_node", type=int, default=8, help="每个节点的 GPU 数量（奖励模型）")
    parser.add_argument(
        "--colocate_actor_ref",
        action="store_true",
        default=False,
        help="是否将参考模型和 actor 共置，如果为 True，它们将共享相同的 GPU。",
    )

    parser.add_argument("--actor_num_nodes", type=int, default=1, help="actor 模型的节点数")
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8, help="每个节点的 GPU 数量（actor 模型）")
    parser.add_argument("--critic_num_nodes", type=int, default=1, help="critic 模型的节点数")
    parser.add_argument("--critic_num_gpus_per_node", type=int, default=8, help="每个节点的 GPU 数量（critic 模型）")
    parser.add_argument(
        "--colocate_critic_reward",
        action="store_true",
        default=False,
        help="是否将 critic 和奖励模型共置，如果为 True，它们将共享相同的 GPU。",
    )

    # 可选的 vLLM 用于文本生成
    parser.add_argument("--vllm_num_engines", type=int, default=None, help="vLLM 引擎的数量")
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM 引擎的张量并行大小，用于多 GPU 推理",
    )

    parser.add_argument("--prompt_data", type=str, default=None)
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="数据集的采样概率",
    )
    parser.add_argument("--pretrain_data", type=str, default=None)
    parser.add_argument(
        "--pretrain_data_probs",
        type=str,
        default="1.0",
        help="数据集的采样概率",
    )
    parser.add_argument("--pretrain", type=str, default=None)  # 预训练模型路径
    parser.add_argument("--reward_pretrain", type=str, default=None)  # 奖励模型的预训练路径
    parser.add_argument("--save_path", type=str, default="./ckpt")  # 模型保存路径
    parser.add_argument("--num_episodes", type=int, default=1)  # 训练的总轮数
    parser.add_argument("--rollout_batch_size", type=int, default=512)  # rollout 的批量大小
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)  # 微 rollout 的批量大小
    parser.add_argument("--max_epochs", type=int, default=1)  # 最大训练轮数
    parser.add_argument("--prompt_max_len", type=int, default=1024)  # 提示的最大长度
    parser.add_argument("--generate_max_len", type=int, default=1024)  # 生成文本的最大长度
    parser.add_argument("--max_len", type=int, default=None)  # 文本的最大长度
    parser.add_argument("--max_samples", type=int, default=100000)  # 最大样本数
    parser.add_argument("--max_norm", type=float, default=1.0)  # 最大归一化值
    parser.add_argument("--l2", type=float, default=0.0)  # L2 正则化系数
    parser.add_argument("--ptx_coef", type=float, default=0.05)  # 预训练系数
    parser.add_argument("--eps_clip", type=float, default=0.2)  # 剪裁系数
    parser.add_argument("--value_clip", type=float, default=0.2)  # 值函数剪裁系数
    parser.add_argument("--lambd", type=float, default=0.95)  # lambda 值
    parser.add_argument("--gamma", type=float, default=1)  # 折扣因子
    parser.add_argument("--micro_train_batch_size", type=int, default=4)  # 微训练批量大小
    parser.add_argument("--train_batch_size", type=int, default=128)  # 训练批量大小
    parser.add_argument("--load_checkpoint", action="store_true", default=False)  # 是否加载检查点
    parser.add_argument("--normalize_reward", action="store_true", default=False)  # 是否归一化奖励
    parser.add_argument("--top_p", type=float, default=1.0)  # 采样的 top-p 值
    parser.add_argument("--seed", type=int, default=42)  # 随机种子

    parser.add_argument("--local_rank", type=int, default=-1, help="deepspeed 的 local_rank")  # deepspeed 的 local_rank
    parser.add_argument("--zero_stage", type=int, default=2)  # ZeRO 优化的阶段
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)  # 是否启用梯度检查点
    parser.add_argument("--bf16", action="store_true", default=False)  # 是否启用 bfloat16
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)  # actor 模型的学习率
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6)  # critic 模型的学习率
    parser.add_argument("--kl_target", type=float, default=None)  # KL 散度目标
    parser.add_argument("--init_kl_coef", type=float, default=0.02)  # 初始 KL 系数
    parser.add_argument("--enable_ema", action="store_true", help="为模型启用 EMA 检查点。")  # 启用 EMA 检查点
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ 最大分区大小")  # ZeRO++ 最大分区大小
    parser.add_argument("--adam_offload", action="store_true", default=False)  # 是否启用 Adam offload
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)  # 是否启用参考奖励 offload
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)  # 是否在 GPU 上初始化 actor 模型
    parser.add_argument("--flash_attn", action="store_true", default=False)  # 是否启用 flash attention
    parser.add_argument("--aux_loss_coef", type=float, default=0)  # 辅助损失系数
    parser.add_argument("--grad_accum_dtype", type=str, default=None)  # 梯度累积的数据类型
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)  # 是否禁用 trace 缓存
    parser.add_argument("--load_in_4bit", action="store_true", default=False)  # 是否以 4 位加载模型
    parser.add_argument("--lora_rank", type=int, default=0)  # LoRA 的秩
    parser.add_argument("--lora_alpha", type=int, default=16)  # LoRA 的 alpha 值
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")  # 目标模块
    parser.add_argument("--lora_dropout", type=float, default=0)  # LoRA 的 dropout 比率
    parser.add_argument("--input_template", type=str, default="Human:\n{}\nAssistant:\n")  # 输入模板
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true")  # 是否使用再入式梯度检查点
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)  # 是否禁用快速 tokenizer

    # 自定义数据集键名
    parser.add_argument("--input_key", type=str, default=None)  # 输入键名

    # 评估
    parser.add_argument("--eval_steps", type=int, default=-1)  # 评估步骤数
    parser.add_argument("--save_steps", type=int, default=-1)  # 保存步骤数
    parser.add_argument("--logging_steps", type=int, default=1)  # 日志记录步骤数

    # wandb 参数
    parser.add_argument("--use_wandb", type=str, default=None)  # 是否使用 wandb
    parser.add_argument("--wandb_org", type=str, default=None)  # wandb 组织
    parser.add_argument("--wandb_group", type=str, default=None)  # wandb 组
    parser.add_argument("--wandb_project", type=str, default="openrlhf_train_ppo")  # wandb 项目名
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="ppo_%s" % datetime.now().strftime("%m%dT%H:%M"),  # wandb 运行名
    )

    # 性能调优
    parser.add_argument("--perf", action="store_true", default=False)  # 性能调优选项

    args = parser.parse_args()  # 解析命令行参数
    train(args)  # 调用训练函数
