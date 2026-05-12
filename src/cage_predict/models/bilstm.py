"""BiLSTM（双向长短期记忆网络）模型构建器。

BiLSTM 在标准 LSTM 的基础上增加了一个反向的 LSTM 层，使得模型
可以同时利用过去和未来的信息进行预测。

工作原理：
  正向 LSTM：从左到右处理序列 (t-5 → t-4 → ... → t)
  反向 LSTM：从右到左处理序列 (t → t-1 → ... → t-5)
  合并方式：将两个方向的输出拼接 (concat) 在一起

这种双向结构让模型在每个时间步都能看到"上下文"信息，
对于需要全局理解序列的任务特别有效。

本模块的架构（原始论文）：
  BiLSTM(25) → LeakyReLU(α=0.3) → Dense(1) → tanh
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_bilstm_model(
    input_shape: tuple[int, int],
    lstm_units: int = 25,
    batch_size: int = 24,
    learning_rate: float = 0.001,
):
    """构建双向 LSTM 模型（原始论文架构）。

    架构：BiLSTM(25) → LeakyReLU(0.3) → Dense(1) → tanh

    LeakyReLU 的作用：
      标准 ReLU 在输入为负时输出为 0（神经元"死亡"），
      LeakyReLU 在输入为负时输出一个很小的负值（α * 输入），
      保持了梯度流动，避免了神经元永久失活的问题。

    参数
    ----------
    input_shape : (时间步数, 特征数)。
    lstm_units : LSTM 单元数，默认 25。
    batch_size : 批量大小，用于 batch_input_shape（固定批量训练）。
                 注意：使用固定 batch_size 后，预测时也必须使用相同大小的批量。
    learning_rate : Adam 优化器的学习率。

    返回
    -------
    model : 编译好的 Keras Sequential 模型。
    """
    from keras.layers import (
        Activation,
        Bidirectional,
        Dense,
        LeakyReLU,
        LSTM,
    )
    from keras.models import Sequential

    from .lstm import _build_adam

    # batch_input_shape 指定了固定的批量大小和输入形状
    batch_input_shape = (batch_size, input_shape[0], input_shape[1])

    model = Sequential()
    # Bidirectional 包装器：将 LSTM 正向和反向各运行一次，然后拼接结果
    model.add(
        Bidirectional(
            LSTM(lstm_units, batch_input_shape=batch_input_shape, stateful=False),
            merge_mode="concat",  # 将正反向输出拼接，使输出维度翻倍
        )
    )
    # LeakyReLU 激活：α=0.3 表示负半轴的斜率为 0.3
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dense(1))              # 全连接层，输出单个预测值
    model.add(Activation("tanh"))    # tanh 将输出限定在 [-1, 1]，配合归一化数据
    model.compile(loss="mse", optimizer=_build_adam(learning_rate))
    logger.info("已构建 BiLSTM 模型: units=%d, batch_size=%d", lstm_units, batch_size)
    return model
