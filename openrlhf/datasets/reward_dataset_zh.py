from typing import Callable  # 导入Callable类型用于类型注解
import torch  # 导入PyTorch库
from torch.utils.data import Dataset  # 从PyTorch的数据工具中导入Dataset类
from tqdm import tqdm  # 导入tqdm库用于显示进度条
from .utils import exist_and_not_none, zero_pad_sequences  # 从当前模块的utils文件导入两个工具函数

def preprocess_data(data, input_template=None, prompt_key=None, chosen_key=None, rejected_key=None) -> str:
    system_prompt = None  # 初始化系统提示为None

    # 自定义数据集
    if chosen_key and rejected_key:
        if prompt_key:
            prompt = data[prompt_key]  # 提取prompt_key对应的数据
        else:
            prompt = ""  # 如果没有prompt_key，设为空字符串
            input_template = None  # 不再使用输入模板
        chosen = data[chosen_key]  # 提取chosen_key对应的数据
        reject = data[rejected_key]  # 提取rejected_key对应的数据
    else:
        # Anthropic/hh-rlhf 或 tasksource/oasst1_pairwise_rlhf_reward 数据集
        if exist_and_not_none(data, "chosen") and exist_and_not_none(data, "rejected"):
            prompt = data["prompt"] if exist_and_not_none(data, "prompt") else ""  # 提取prompt，如果存在
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman:\n").replace("assistant:", "\nAssistant:\n")
                    + "\nAssistant:\n"
                )  # 替换prompt中的角色标签
            chosen = data["chosen"]  # 提取chosen数据
            reject = data["rejected"]  # 提取rejected数据
            input_template = None  # 不再使用输入模板
        # lmsys/chatbot_arena_conversations 数据集
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))  # 格式化用户内容
                    else:
                        result.append(l["content"] + "\n")  # 添加其他角色内容
                return "".join(result)

            prompt = ""  # 初始化prompt为空
            chosen = data["conversation_a"] if data["winner"] == "model_a" else data["conversation_b"]  # 根据winner选择
            reject = data["conversation_b"] if data["winner"] == "model_a" else data["conversation_a"]  # 相反选择
            chosen = process_chatbot_arena_conversations(chosen)  # 处理chosen对话
            reject = process_chatbot_arena_conversations(reject)  # 处理reject对话
            input_template = None  # 不再使用输入模板
        # openai/webgpt_comparisons 数据集
        elif exist_and_not_none(data, "answer_0") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]  # 提取问题的完整文本
            chosen = data["answer_0"] if data["score_0"] > data["score_1"] else data["answer_1"]  # 根据得分选择
            reject = data["answer_1"] if data["score_0"] > data["score_1"] else data["answer_0"]  # 相反选择
        else:
            raise ValueError("Unknown reward dataset")  # 如果数据集类型未知，抛出异常

    # 边际损失
    margin = data["margin"] if exist_and_not_none(data, "margin") else 0  # 提取边际损失，默认值为0

    # 输入模板
    if input_template:
        prompt = input_template.format(prompt)  # 应用输入模板格式化prompt

    if system_prompt:
        prompt = system_prompt + "\n" + prompt  # 如果有系统提示，添加到prompt前
    return prompt, chosen, reject, margin  # 返回处理后的数据

class RewardDataset(Dataset):
    """
    用于奖励模型的数据集

    参数:
        dataset: 用于奖励模型的数据集
        self.tokenizer: 用于奖励模型的分词器
        self.max_length: 输入的最大长度
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template="Human:\n{}\nAssistant:\n",
        is_dpo=False,
    ) -> None:
        super().__init__()
        self.is_dpo = is_dpo  # 初始化DPO标志

        self.prompts = []  # 初始化prompts列表
        self.chosens = []  # 初始化chosens列表
        self.rejects = []  # 初始化rejects列表
        if self.is_dpo:
            self.prompt_ids_lens = []  # 初始化prompt_ids_lens列表
        else:
            self.margins = []  # 初始化margins列表

        self.tokenizer = tokenizer  # 设置分词器
        self.strategy = strategy  # 设置策略
        self.max_length = max_length  # 设置最大长度
        self.is_dpo = is_dpo  # 设置DPO标志

        prompt_key = getattr(self.strategy.args, "prompt_key", None)  # 获取prompt_key
        chosen_key = getattr(self.strategy.args, "chosen_key", None)  # 获取chosen_key
        rejected_key = getattr(self.strategy.args, "rejected_key", None)  # 获取rejected_key

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, margin = preprocess_data(
                data, input_template, prompt_key, chosen_key, rejected_key
            )

            # prompt_ids_len 用于prompt掩码
            if self.is_dpo:
                prompt_token = self.tokenizer(
                    prompt,
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_ids_len = prompt_token["attention_mask"].int().sum().item()  # 计算prompt的有效长度
                # 过滤掉长度超过最大值的样本（答案长度留出2）
                if prompt_ids_len >= self.max_length - 2:
                    continue
                else:
                    self.prompt_ids_lens.append(prompt_ids_len)  # 添加有效长度到列表
            else:
                self.margins.append(margin)  # 添加边际损失到列表

            self.prompts.append(prompt)  # 添加prompt到列表
            self.chosens.append(chosen)  # 添加chosen到列表
            self.rejects.append(reject)  # 添加reject到列表

    def __len__(self):
        length = len(self.chosens)  # 数据集的长度
        return length

    def __getitem__(self, idx):
        prompt, chosen, reject = self.prompts[idx], self.chosens[idx], self.rejects[idx]  # 获取对应索引的数据
        if self.is_dpo:
            extra = self.prompt_ids_lens[idx]  # 获取对应的prompt长度
        else:
            extra = self.margins[idx]  # 获取对应的边际损失

        chosen = prompt + chosen + " " + self.tokenizer.eos_token  # 添加结束标记到chosen
        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        reject = prompt + reject + " " + self.tokenizer.eos_token  # 添加结束标记到reject
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        # 避免EOS_token截断
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            extra,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        rejects_masks = []
        extras = []
        for chosen_id, chosen_mask, reject_id, rejects_mask, extra in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            rejects_masks.append(rejects_mask)
            extras.append(extra)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)  # 填充chosen_ids
        chosen_masks = zero_pad_sequences(chosen_masks)  # 填充chosen_masks
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)  # 填充reject_ids
        rejects_masks = zero_pad_sequences(rejects_masks)  # 填充rejects_masks
        return chosen_ids, chosen_masks, reject_ids, rejects_masks, extras  # 返回填充后的数据
