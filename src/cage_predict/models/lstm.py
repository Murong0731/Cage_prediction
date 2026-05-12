"""LSTM（长短期记忆网络）模型构建器 —— 用于时间序列运动和力预测。

LSTM 是循环神经网络 (RNN) 的改进版本，通过"门控机制"解决了传统 RNN
在长序列上的梯度消失问题。它能够学习时间序列中的长期依赖关系。

LSTM 单元包含三个门：
  - 遗忘门 (Forget Gate)：决定丢弃哪些旧信息
  - 输入门 (Input Gate)：决定存储哪些新信息
  - 输出门 (Output Gate)：决定输出哪些信息

本模块保留原始论文中的模型架构：
  - num_layers=1 (单层): LSTM(25) → Dense(1) 输出层
  - num_layers=3 (三层): LSTM(25) → Dropout → LSTM(100) → Dropout → LSTM(100) → Dropout → Dense(1)

多层 LSTM 中，前两层的 return_sequences=True 表示输出完整的时间序列给下一层，
最后一层的 return_sequences=False（默认）表示只输出最后一个时间步的结果。

兼容 TensorFlow 2.4–2.15（Keras 2 API）。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _build_adam(learning_rate: float = 0.001):
    """构建 Adam 优化器，兼容 lr/learning_rate 两种参数名。

    Keras 不同版本中 Adam 的学习率参数名不同：
      - 旧版本使用 lr= （已弃用但仍可用）
      - 新版本使用 learning_rate=
    本函数先尝试新参数名，失败后回退到旧参数名。

    clipnorm=1.0 对梯度全局范数进行裁剪，防止高学习率下
    LSTM/GRU 出现梯度爆炸（NaN loss）。
    """
    from keras.optimizers import Adam

    try:
        return Adam(learning_rate=learning_rate, clipnorm=1.0)
    except TypeError:
        return Adam(lr=learning_rate, clipnorm=1.0)


def build_lstm_model(
    input_shape: tuple[int, int],
    num_layers: int = 3,
    units: list[int] | None = None,
    dropout: float = 0.3,
    activation: str = "tanh",
    learning_rate: float = 0.001,
):
    """构建与原始论文架构一致的 LSTM 模型。

    参数
    ----------
    input_shape : (时间步数, 特征数) —— 输入数据的形状。
                  例如 (10, 2) 表示每个样本包含10个时间步，每步有2个特征。
    num_layers : LSTM 层数，1 为单层，3 为深层。
    units : 每层 LSTM 的单元数列表。默认值匹配原始代码：
            三层时为 [25, 100, 100]，单层时为 [25]。
    dropout : 层间的 Dropout 比例（0 到 1 之间）。
              Dropout 随机丢弃一部分神经元的输出，防止过拟合。
    activation : LSTM 的激活函数，通常为 "tanh"（双曲正切函数）。
    learning_rate : Adam 优化器的初始学习率。

    返回
    -------
    model : 编译好的 Keras Sequential 模型。
    """
    from keras.layers import Activation, Dense, Dropout, LSTM
    from keras.models import Sequential

    if units is None:
        units = [25, 100, 100] if num_layers == 3 else [25]

    model = Sequential()

    if num_layers == 1:
        # 单层 LSTM：不需要 return_sequences，直接输出最后一步
        model.add(LSTM(units[0], activation=activation, input_shape=input_shape))
    elif num_layers >= 2:
        # 多层 LSTM：除最后一层外，都需要 return_sequences=True
        # 这样每个时间步都会输出，供下一层 LSTM 继续处理
        model.add(LSTM(units[0], activation=activation, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        # 中间的隐藏层
        for u in units[1:-1]:
            model.add(LSTM(u, activation=activation, return_sequences=True))
            model.add(Dropout(dropout))
        # 最后一层 LSTM：return_sequences=False（默认），只输出最后一步
        model.add(LSTM(units[-1], activation=activation))
        model.add(Dropout(dropout))
    else:
        raise ValueError(f"num_layers 必须 >= 1，但收到了 {num_layers}")

    # 全连接输出层：将 LSTM 的输出映射为单个预测值
    model.add(Dense(1))
    model.add(Activation("tanh"))
    # 编译模型：使用均方误差 (MSE) 作为损失函数，Adam 作为优化器
    model.compile(loss="mse", optimizer=_build_adam(learning_rate))
    logger.info("已构建 LSTM 模型: num_layers=%d, units=%s, activation=%s", num_layers, units, activation)
    return model
