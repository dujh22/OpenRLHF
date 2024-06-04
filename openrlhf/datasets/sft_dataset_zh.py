from typing import Callable
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences  # 导入自定义的存在且不为空检查函数以及零填充序列函数

# 数据预处理函数
def preprocess_data(data, input_template=None, input_key=None, output_key=None):
    system_prompt = None  # 初始化系统提示为 None

    # 如果提供了输入键和输出键，从自定义数据集中提取相应值
    if input_key and output_key:
        prompt = data[input_key]  # 获取输入键对应的值
        response = data[output_key]  # 获取输出键对应的值
    else:
        # 处理 pvduy/sharegpt_alpaca_oa_vicuna_format 数据集
        if exist_and_not_none(data, "prompt") and exist_and_not_none(data, "label"):
            prompt = data["prompt"].replace("USER:", "").replace("ASSISTANT:", "")
            response = data["label"].replace("</s>", "")
        # 处理 Open-Orca/OpenOrca 数据集
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            system_prompt = data["system_prompt"]
            prompt = data["question"]
            response = data["response"]
        # 处理 MaziyarPanahi/WizardLM_evol_instruct_V2_196k 和 jondurbin/airoboros-3.2 数据集
        elif exist_and_not_none(data, "conversations"):

            # 处理对话数据的内部函数
            def process_conversations(lll):
                result = []
                for l in lll:
                    if "human" in l["from"]:
                        result.append(input_template.format(l["value"]))
                    elif "system" in l["from"]:
                        nonlocal system_prompt
                        system_prompt = l["value"]
                    else:
                        result.append(l["value"] + "\n")
                return "".join(result)

            # 获取并处理对话数据
            prompt = process_conversations(data["conversations"][:-1])
            response = data["conversations"][-1]["value"]
            input_template = None  # 如果已经处理，不再使用输入模板修改
        # 处理 batch_inference.py 数据集
        elif exist_and_not_none(data, "input") and exist_and_not_none(data, "output"):
            prompt = data["input"]
            response = data["output"]
            input_template = None  # 如果已经处理，不再使用输入模板修改
        else:
            # 如果数据集格式未知，抛出错误
            raise ValueError("Unknown SFT dataset")

    # 如果有输入模板，使用模板格式化提示
    if input_template:
        prompt = input_template.format(prompt)

    # 如果存在系统提示，添加到提示的开头
    if system_prompt:
        prompt = system_prompt + "\n" + prompt
    return prompt, response

# 定义 SFT 模型的数据集类
class SFTDataset(Dataset):
    """
    SFT 模型的数据集

    参数:
        dataset: SFT 模型的数据集
        tokenizer: SFT 模型的分词器
        max_length: 输入的最大长度
    """

    def __init__(
        self,
        dataset,  # 输入数据集
        tokenizer: Callable,  # 分词器
        max_length: int,  # 输入的最大长度
        strategy,  # SFT 模型的策略
        input_template="Human:\n{}\nAssistant:\n",  # 输入格式化模板
        pretrain_mode=False,  # 预训练模式开关
    ) -> None:
        super().__init__()
        self.prompts = []  # 初始化提示列表
        self.responses = []  # 初始化响应列表
        self.prompt_ids_lens = []  # 初始化提示ID长度列表
        self.tokenizer = tokenizer  # 分词器
        self.strategy = strategy  # 策略
        self.pretrain_mode = pretrain_mode  # 预训练模式
        self.max_length = max_length  # 最大长度
        input_key = getattr(self.strategy.args, "input_key", None)  # 从策略参数中获取自定义输入键
        output_key = getattr(self.strategy.args, "output_key", None)  # 从策略参数中获取自定义输出键

        # 处理数据集中的每条数据
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):  # 仅在rank 0节点上展示进度条
            prompt, response = preprocess_data(data, None if pretrain_mode else input_template, input_key, output_key)

            if not self.pretrain_mode:
                # 将提示进行分词和编码
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()  # 获取提示的长度

            else:
                prompt_ids_len = 0  # 预训练模式下，提示长度为0

            # 过滤掉长度大于最大长度的样本（2个答案的长度）
            if not self.pretrain_mode:
                if prompt_ids_len >= self.max_length - 2:
                    continue
                if not prompt or not response:
                    continue

            # 将有效的提示和响应添加到列表中
            self.prompt_ids_lens.append(prompt_ids_len)
            self.prompts.append(prompt)
            self.responses.append(response)

    # 数据集的长度
    def __len__(self):
        length = len(self.prompts)  # 返回数据集长度
        return length

    # 根据索引获取数据
    def __getitem__(self, idx):
        prompt_ids_len = self.prompt_ids_lens[idx]  # 获取提示的长度
        prompt = self.prompts[idx]  # 获取提示
        response = self.responses[idx]  # 获取响应

        input_token = self.tokenizer(
            prompt + response + " " + self.tokenizer.eos_token,  # 拼接提示和响应，并添加结束标记
            max_length=self.max_length,  # 设置最大长度
            padding=False,  # 不进行填充
            truncation=True,  # 进行截断
            return_tensors="pt",  # 返回 PyTorch 张量
        )
        info = {"input": prompt, "output": response}  # 保存输入和输出信息

        # 避免 EOS_token 截断
        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True
        return prompt_ids_len, input_token["input_ids"], input_token["attention_mask"], info

    # 合并多个样本到一个批次内
    def collate_fn(self, item_list):
        prompt_ids_lens = []
        input_ids = []
        attention_masks = []
        infos = {"input": [], "output": []}

        # 遍历样本列表
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            prompt_ids_lens.append(prompt_ids_len)  # 添加提示长度
            input_ids.append(input_id)  # 添加输入ID
            attention_masks.append(attention_mask)  # 添加attention_mask
            infos["input"].append(info["input"])  # 添加原始输入信息
            infos["output"].append(info["output"])  # 添加原始输出信息

        # 右侧零填充输入ID序列和attention_mask
        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        return prompt_ids_lens, input_ids, attention_masks, infos