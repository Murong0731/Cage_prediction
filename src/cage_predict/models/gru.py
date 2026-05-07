"""GRU（门控循环单元）模型构建器 —— 三层 GRU + Dropout。

GRU 是 LSTM 的简化变体，它将 LSTM 的遗忘门和输入门合并为"更新门"，
同时用"重置门"替代 LSTM 的输出门。因此 GRU 只有两个门，参数更少，
训练速度更快，在许多任务上能达到与 LSTM 相近的效果。

GRU 的两个门：
  - 重置门 (Reset Gate)：控制如何将新输入与之前的记忆结合
  - 更新门 (Update Gate)：控制保留多少之前的记忆

本模块构建与原始论文一致的 3 层 GRU 架构：
  GRU(25) → Dropout → GRU(100) → Dropout → GRU(100) → Dropout → Dense(1)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_gru_model(
    input_shape: tuple[int, int],
    units: list[int] | None = None,
    dropout: float = 0.3,
    activation: str = "tanh",
    learning_rate: float = 0.01,
):
    """构建与原始代码一致的三层 GRU 模型。

    架构：GRU(25) → Dropout → GRU(100) → Dropout → GRU(100) → Dropout → Dense(1)

    参数
    ----------
    input_shape : (时间步数, 特征数) —— 输入数据的形状。
    units : 每层 GRU 的单元数，默认为 [25, 100, 100]。
            第一层 25 个单元提取基础特征，后两层 100 个单元学习高级模式。
    dropout : 层间 Dropout 比例，默认 0.3（即随机丢弃 30% 的神经元）。
    activation : 激活函数，默认 "tanh"（双曲正切，输出范围 [-1, 1]）。
    learning_rate : Adam 优化器的初始学习率。

    返回
    -------
    model : 编译好的 Keras Sequential 模型。
    """
    from keras.layers import Activation, Dense, Dropout, GRU
    from keras.models import Sequential

    from .lstm import _build_adam

    if units is None:
        units = [25, 100, 100]

    model = Sequential()
    # 第一层 GRU：return_sequences=True 将完整序列传给下一层
    model.add(GRU(units[0], activation=activation, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    # 第二层 GRU：继续输出完整序列
    model.add(GRU(units[1], activation=activation, return_sequences=True))
    model.add(Dropout(dropout))
    # 第三层 GRU：只输出最后一步的结果
    model.add(GRU(units[2], activation=activation))
    model.add(Dropout(dropout))
    # 全连接输出层
    model.add(Dense(1))
    model.add(Activation("tanh"))
    model.compile(loss="mse", optimizer=_build_adam(learning_rate))
    logger.info("已构建 GRU 模型: units=%s, activation=%s", units, activation)
    return model
