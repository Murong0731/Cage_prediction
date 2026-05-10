"""CNN-BiLSTM-Attention（卷积-双向LSTM-注意力）模型构建器。

这是第5章的核心模型，结合了三种深度学习技术：

  1. Conv1D（一维卷积）：
     - 在时间维度上滑动卷积核，提取局部时序模式
     - 类似于移动平均，但可以学习复杂的非线性模式
     - kernel_size=1 意味着对每个时间步的多个特征进行融合

  2. BiLSTM（双向LSTM）：
     - 从正反两个方向处理序列，捕捉长期依赖关系
     - 返回完整序列 (return_sequences=True)，供注意力层使用

  3. Attention（注意力机制）：
     - 自动学习哪些时间步更重要
     - 通过 Softmax 计算每个时间步的权重
     - 加权合并所有时间步的信息

整体架构：
  Conv1D → Dropout → BiLSTM → Dropout → Attention → Flatten → Dense(1)

Attention 机制使用的是 attention_3d_block2（Permute 模式），
兼容 TensorFlow 2.x Keras API。

注意：旧版 attention_3d_block（Dense 变体）已被移除，
因为它使用了已废弃的 keras.layers.merge 且存在缺失的 K 导入。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def attention_3d_block2(inputs, single_attention_vector: bool = False):
    """注意力机制 —— 使用 Permute + Dense + Multiply 实现（推荐版本）。

    这是注意力机制的一种简洁实现，通过以下步骤计算注意力权重：

    步骤：
      1. Permute((2,1))：交换维度，将 (batch, time, features) 变为 (batch, features, time)
      2. Dense(time_steps, softmax)：对每个特征在所有时间步上学习注意力得分并归一化
      3. Permute((2,1))：再交换回来，恢复原始维度顺序
      4. Multiply：将注意力权重与原始输入逐元素相乘

    这样，重要时间步的信号被放大，不重要时间步的信号被抑制。

    参数
    ----------
    inputs : Keras 张量，形状为 (batch_size, time_steps, features)。
    single_attention_vector : 如果为 True，在所有特征间共享一个注意力向量。

    返回
    -------
    output : 加权后的 Keras 张量，形状与 inputs 相同。
    """
    import keras.backend as K
    from keras.layers import (
        Dense,
        Lambda,
        Multiply,
        Permute,
        RepeatVector,
    )

    time_steps = K.int_shape(inputs)[1]  # 时间步数
    input_dim = K.int_shape(inputs)[2]   # 每个时间步的特征维度

    # 步骤1: 转置，变成 (batch, features, time) 以便对时间步计算注意力
    a = Permute((2, 1))(inputs)

    # 步骤2: 全连接层学习每个时间步的注意力得分
    a = Dense(time_steps, activation="tanh")(a)

    # 如果使用单注意力向量：对所有特征取平均，再复制回原来的维度
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)  # 沿特征维度取平均
        a = RepeatVector(input_dim)(a)               # 复制回 input_dim 维

    # 步骤3: 转置回来 → 得到注意力权重
    a_probs = Permute((2, 1))(a)

    # 步骤4: 将注意力权重与原始输入相乘
    output = Multiply()([inputs, a_probs])
    return output


def build_attention_model(
    input_shape: tuple[int, int],
    conv_filters: int = 64,
    conv_kernel: int = 1,
    lstm_units: int = 64,
    lstm_activation: str = "tanh",
    dropout_rate: float = 0.1,
    attention_mode: str = "permute",
    learning_rate: float = 0.01,
):
    """构建 CNN-BiLSTM-Attention 模型。

    架构流程：
      输入 (batch, time_steps, features)
        ↓
      Conv1D（一维卷积，提取局部特征）
        ↓
      Dropout（防止过拟合）
        ↓
      BiLSTM（双向LSTM，捕捉长期依赖）
        ↓
      Attention（自适应加权各时间步）
        ↓
      Flatten（展平为向量）
        ↓
      Dense(1)（输出单个预测值）

    参数
    ----------
    input_shape : (时间步数, 输入特征维度)。
    conv_filters : Conv1D 的卷积核数量，默认 64。更多的核可以提取更多样的特征，
                   但也增加计算成本。
    conv_kernel : Conv1D 的卷积核大小，默认 1。
                  kernel_size=1 表示逐特征点的全连接卷积。
    lstm_units : BiLSTM 的单元数，默认 64。
    lstm_activation : LSTM 的激活函数，默认 "tanh"。
    dropout_rate : Conv1D 后的 Dropout 比例，默认 0.1（10%）。
    attention_mode : 注意力模式，固定为 "permute"（TF 2.x 兼容）。
    learning_rate : Adam 优化器的学习率。

    返回
    -------
    model : 编译好的 Keras Model。
    """
    from keras.layers import (
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        Input,
        LSTM,
    )
    from keras.models import Model

    from .lstm import _build_adam

    # 输入层
    inputs = Input(shape=input_shape)

    # 第一步：一维卷积提取局部时序模式
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel, activation="relu")(inputs)
    x = Dropout(dropout_rate)(x)

    # 第二步：双向 LSTM 捕捉长期依赖关系
    lstm_out = Bidirectional(
        LSTM(lstm_units, activation=lstm_activation, return_sequences=True),
        merge_mode="concat",  # 正反向拼接，输出维度 = lstm_units * 2
    )(x)
    lstm_out = Dropout(dropout_rate)(lstm_out)

    # 第三步：注意力机制 —— 自动学习各时间步的重要性权重
    attn = attention_3d_block2(lstm_out)

    # 第四步：展平并输出
    attn = Flatten()(attn)      # 将 3D 张量展平为 2D
    output = Dense(1)(attn)     # 全连接输出层
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss="mse", optimizer=_build_adam(learning_rate))
    logger.info(
        "已构建 Attention 模型: conv_filters=%d, lstm_units=%d, attention_mode=%s, lr=%f",
        conv_filters, lstm_units, attention_mode, learning_rate,
    )
    return model
