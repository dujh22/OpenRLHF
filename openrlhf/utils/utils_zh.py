import os  # 导入操作系统模块
from pathlib import Path  # 从 pathlib 模块导入 Path 类

from datasets import Dataset, interleave_datasets, load_dataset  # 从 datasets 模块导入 Dataset、interleave_datasets 和 load_dataset
from transformers import AutoTokenizer  # 从 transformers 模块导入 AutoTokenizer

from openrlhf.utils import DeepspeedStrategy  # 从 openrlhf.utils 导入 DeepspeedStrategy

# 默认填充标记
DEFAULT_PAD_TOKEN = "[PAD]"
# 默认结束标记
DEFAULT_EOS_TOKEN = "</s>"
# 默认开始标记
DEFAULT_BOS_TOKEN = "<s>"
# 默认未知标记
DEFAULT_UNK_TOKEN = "<unk>"

# 获取分词器的函数
def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    # 设置填充的对齐方向
    tokenizer.padding_side = padding_side
    # 注意：当启用 vLLM 时，不要调整分词器的嵌入大小，否则词汇表大小会与 vLLM 不匹配
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        # 如果分词器的填充标记为空，则使用结束标记
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer  # 返回分词器

# 获取策略的函数
def get_strategy(args):
    # 创建 DeepspeedStrategy 实例
    strategy = DeepspeedStrategy(
        # 设置随机种子，默认值为 42
        seed=getattr(args, "seed", 42),
        # 设置梯度裁剪的最大范数，默认值为 1.0
        max_norm=getattr(args, "max_norm", 1.0),
        # 设置微训练批次大小，默认值为 1
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        # 设置训练批次大小，默认值为 128
        train_batch_size=getattr(args, "train_batch_size", 128),
        # 设置零阶段参数（用于 ZeRO 优化器）
        zero_stage=args.zero_stage,
        # 设置是否使用 bf16 精度，默认值为 True
        bf16=getattr(args, "bf16", True),
        # 将 args 参数传递给策略
        args=args,
    )
    # 返回创建的策略对象
    return strategy

# 混合数据集的函数
def blending_datasets(
    datasets,  # 数据集名称，以逗号分隔
    probabilities,  # 数据集的概率，以逗号分隔
    strategy=None,  # 策略
    seed=42,  # 随机种子，默认值为 42
    max_count=5000000,  # 最大样本数，默认值为 5000000
    return_eval=True,  # 是否返回评估集，默认值为 True
    stopping_strategy="first_exhausted",  # 停止策略，默认值为 "first_exhausted"
):
    datasets = datasets.split(",")  # 将数据集名称转换为列表
    probabilities = list(map(float, probabilities.split(",")))  # 将概率转换为浮点数列表
    assert len(probabilities) == len(datasets)  # 确保数据集数量与概率数量相同

    train_data_list = []  # 训练数据列表
    eval_data_list = []  # 评估数据列表
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()  # 去掉数据集名称前后的空格
        dataset_subfold_list = dataset.split("@")  # 分割子文件夹列表
        strategy.print(f"dataset: {dataset}")  # 打印数据集名称
        # 判断是否为本地目录或常见的本地文件
        if os.path.isdir(os.path.join(os.getcwd(), dataset)) or dataset.endswith(
            (".json", ".jsonl", ".csv", ".parquet", ".txt")
        ):
            if dataset.endswith((".json", ".jsonl", ".csv", ".parquet", ".txt")):
                files = dataset  # 文件路径
                data_type = os.path.splitext(files)[1][1:]  # 文件类型
            else:
                path = Path(dataset)  # 数据集路径
                script = [str(file.resolve()) for file in Path(path).rglob("*.py")]  # 找到所有 Python 脚本
                extensions = ("*.json", "*.jsonl", "*.csv", "*.parquet", "*.txt")  # 支持的文件扩展名
                files = [str(file) for ext in extensions for file in Path(path).rglob(ext)]  # 找到所有支持的文件
                strategy.print(f"script: {script}")  # 打印脚本路径
                strategy.print(f"files: {files}")  # 打印文件路径
                # 对于目录，使用 Python 脚本或第一个文件类型
                data_type = script[0] if len(script) == 1 else os.path.splitext(files[0])[1][1:]
            # 重新格式化数据类型
            if data_type in ["json", "jsonl"]:
                data_type = "json"  # 将 json 和 jsonl 统一为 json
            elif data_type == "txt":
                data_type = "text"  # 将 txt 转换为 text
            elif data_type.endswith(".py"):
                # 加载带有 Python 脚本的本地目录
                files = None
            if data_type.endswith(".py"):
                strategy.print(f"load {dataset} with script {data_type}")  # 打印加载脚本的信息
            else:
                strategy.print(f"load {files} from {dataset}")  # 打印加载文件的信息
            data = load_dataset(data_type, data_files=files)  # 加载数据集
        elif len(dataset_subfold_list) == 2:
            dataset = dataset_subfold_list[0]  # 数据集名称
            subfold = dataset_subfold_list[1]  # 子文件夹名称
            data = load_dataset(dataset, data_dir=subfold.strip())  # 加载子文件夹中的数据集
        elif len(dataset_subfold_list) == 1:
            dataset = dataset_subfold_list[0]  # 数据集名称
            data = load_dataset(dataset)  # 加载数据集
        else:
            raise Exception(f"Dataset Name {dataset}: Format error")  # 抛出格式错误异常

        if "train" in data:
            train_data_list.append(data["train"].select(range(min(max_count, len(data["train"])))))  # 选择训练数据
        else:
            train_data_list.append(data.select(range(min(max_count, len(data)))))  # 选择数据

        if return_eval:
            if "test" in data:
                eval_data = data["test"].select(range(min(int(max_count * 0.1), len(data["test"]))))  # 选择测试数据
            elif "validation" in data:
                eval_data = data["validation"].select(range(min(int(max_count * 0.1), len(data["validation"]))))  # 选择验证数据
            elif "train" in data:
                eval_data = data["train"].select(range(min(int(max_count * 0.1), int(len(data["train"]) * 0.01))))  # 选择训练数据作为评估数据
            else:
                eval_data = data.select(range(min(int(max_count * 0.1), int(len(data) * 0.001))))  # 选择数据作为评估数据
            eval_data_list.append(eval_data)  # 添加评估数据到列表

    # 合并数据集
    if strategy.is_rank_0():
        print(train_data_list)  # 打印训练数据列表

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )  # 混合训练数据集
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )  # 混合评估数据集
        return train_dataset, eval_dataset  # 返回训练和评估数据集
    else:
        return train_dataset  # 返回训练数据集
