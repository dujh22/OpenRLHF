from typing import Tuple  # 从typing模块导入Tuple类型

import numpy as np  # 导入NumPy库，取别名为np

class AdaptiveKLController:  # 定义自适应KL控制器类
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):  # 类的构造函数
        self.value = init_kl_coef  # 初始化KL系数
        self.target = target  # 设置目标值
        self.horizon = horizon  # 设置水平线

    def update(self, current, n_steps):  # 定义更新函数
        target = self.target  # 获取目标值
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)  # 计算比例误差，并限制在-0.2到0.2之间
        mult = 1 + proportional_error * n_steps / self.horizon  # 计算比例增益
        self.value *= mult  # 更新KL系数

class FixedKLController:  # 定义固定KL控制器类
    """Fixed KL controller."""

    def __init__(self, kl_coef):  # 类的构造函数
        self.value = kl_coef  # 初始化KL系数

    def update(self, current, n_steps):  # 定义更新函数
        pass  # 固定KL控制器不进行更新操作