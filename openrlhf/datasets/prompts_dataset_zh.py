from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none  # 导入自定义的存在且非空检查函数

def preprocess_data(data, input_template=None, input_key=None) -> str:
    system_prompt = None  # 初始化系统提示为 None

    # custom dataset
    if input_key:
        prompt = data[input_key]  # 如果有自定义的输入键，则从数据中提取该键对应的值作为提示
    else:
        # 处理不同的数据集格式，根据数据的不同字段进行相应的处理
        # Dahoas/full-hh-rlhf
        if exist_and_not_none(data, "prompt"):
            prompt = data["prompt"]
            # tasksource/oasst1_pairwise_rlhf_reward
            if prompt.startswith("prompter:"):
                prompt = (
                    prompt.replace("prompter:", "\nHuman:\n").replace("assistant:", "\nAssistant:\n")
                    + "\nAssistant:\n"
                )
            input_template = None  # 如果经过处理，不再用输入模板修改
        # Open-Orca/OpenOrca
        elif exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
            system_prompt = data["system_prompt"]
            prompt = data["question"]
        # lmsys/chatbot_arena_conversations
        elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):
            
            # 定义一个处理lmsys/chatbot_arena_conversations数据集的内部函数
            def process_chatbot_arena_conversations(lll):
                result = []
                for l in lll:
                    if "user" in l["role"]:
                        result.append(input_template.format(l["content"]))
                    else:
                        result.append(l["content"] + "\n")
                return "".join(result)

            # 获取对话并处理每条消息
            prompt = data["conversation_a"][:-1]
            prompt = process_chatbot_arena_conversations(prompt)
            input_template = None  # 如果经过处理，不再用输入模板修改
        # openai/webgpt_comparisons
        elif exist_and_not_none(data, "question") and exist_and_not_none(data, "answer_1"):
            prompt = data["question"]["full_text"]
        else:
            # 如果数据集格式未知，抛出错误
            raise ValueError("Unknown prompts dataset")

    # 如果有输入模板，使用模板格式化提示
    if input_template:
        prompt = input_template.format(prompt)

    # 如果存在系统提示，添加到提示的开头
    if system_prompt:
        prompt = system_prompt + "\n" + prompt
    return prompt

class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        strategy: strategy for PPO model, contains distributed training information
        input_template: template for formatting the input
    """

    def __init__(
        self,
        dataset,  # 数据集
        tokenizer,  # 分词器
        strategy,  # PPO模型的策略
        input_template="Human:\n{}\nAssistant:\n",  # 输入模板，默认是 "Human:\n{}\nAssistant:\n"
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)  # 从策略参数中获取自定义的输入键

        self.prompts = []
        # 处理数据集中每条数据
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):  # 仅在rank 0节点上展示进度条
            prompt = preprocess_data(data, input_template, input_key)  # 预处理每条数据
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)  # 返回数据集长度
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]  # 返回指定索引的数据