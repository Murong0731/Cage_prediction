"""BPNN（反向传播神经网络）模型构建器。

BPNN 是最经典的前馈神经网络，信息从输入层经过多个隐藏层单向传播到输出层，
通过反向传播算法更新权重。

本模块支持两种构建方式：
  - Sequential API（Model_NN 风格）：层按顺序堆叠，代码简洁
  - Functional API（Model_NN1 风格）：使用 Input 层定义，更灵活（第4.2节使用）

默认架构：3 个隐藏层，每层 150 个神经元，使用 tanh 激活函数。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_bpnn_model(
    input_dim: int,
    hidden_units: list[int] | None = None,
    activation: str = "tanh",
    learning_rate: float = 0.01,
    use_functional_api: bool = False,
):
    """构建前馈 BPNN 模型。

    参数
    ----------
    input_dim : 输入特征的数量。例如，如果使用 Surge、Heave、Pitch 三个特征，
                则 input_dim=3。
    hidden_units : 各隐藏层的神经元数量列表，默认 [150, 150, 150]。
                   每个数字对应一个 Dense 层。
    activation : 隐藏层的激活函数，默认 "tanh"。
                 tanh 函数输出范围 [-1, 1]，适合归一化到该范围的数据。
    learning_rate : Adam 优化器的学习率。
    use_functional_api : 如果为 True，使用 Keras Functional API（Model_NN1 风格）；
                         如果为 False，使用 Sequential API（Model_NN 风格）。

    返回
    -------
    model : 编译好的 Keras 模型。
    """
    from keras.layers import Dense, Input
    from keras.models import Model, Sequential

    from .lstm import _build_adam

    if hidden_units is None:
        hidden_units = [150, 150, 150]

    if use_functional_api:
        # Functional API 风格：显式定义输入层，适合多输入/多输出的复杂架构
        input_layer = Input(shape=(input_dim,))
        x = input_layer
        for units in hidden_units:
            x = Dense(units, activation=activation)(x)
        output_layer = Dense(1, activation="linear")(x)  # 线性输出（回归任务）
        model = Model(inputs=input_layer, outputs=output_layer)
    else:
        # Sequential API 风格：层按顺序堆叠，代码更简洁
        model = Sequential()
        model.add(Dense(hidden_units[0], activation=activation, input_shape=(input_dim,)))
        for units in hidden_units[1:]:
            model.add(Dense(units, activation=activation))
        model.add(Dense(1, activation="linear"))  # 输出层：单个值，线性激活

    model.compile(loss="mse", optimizer=_build_adam(learning_rate))
    logger.info("已构建 BPNN 模型: hidden_units=%s, activation=%s, functional=%s",
                hidden_units, activation, use_functional_api)
    return model
